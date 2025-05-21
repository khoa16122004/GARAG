from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI
from .util import f1

import lightning.pytorch as pl

import os
import math
import torch
import logging

cls_mapping = {
    "Llama-7b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-7b-chat-hf"),
    "Llama-13b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-13b-chat-hf"),
    "Mistral-7b": (MistralForCausalLM, AutoTokenizer, True, "Mistral-7B-Instruct-v0.2"),
    "vicuna-7b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-7b-v1.5"),
    "vicuna-13b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-13b-v1.5"),
    "gemma-7b": (AutoModelForCausalLM, AutoTokenizer, True, "gemma-7b-it")
}

logger = logging.getLogger(__name__)

save_keys = [
    "question", "doc_id", "question", "answers"
]

def load_reader(opt):
    if opt.reader == "chatgpt":
        return Reader_GPT(opt)
    elif opt.is_vllm:
        return Reader_vLLM(opt)
    else:
        return Reader(opt)

def _load_model(opt):
    reader_name = opt.reader
    if reader_name in cls_mapping:
        return cls_mapping[reader_name]
    else:
        NotImplementedError()

class Reader(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        model_cls, tokenizer_cls, self.is_decoder, hf_name = _load_model(opt)
        self.model = model_cls.from_pretrained(os.path.join(opt.model_dir, hf_name)).to("cuda:0")
        self.tokenizer = tokenizer_cls.from_pretrained(os.path.join(opt.model_dir, hf_name))
        self.generate_kwargs = dict(
            max_new_tokens=opt.max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True
        )
        if self.is_decoder:
            self.tokenizer.padding_side = "left"
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model.generate(input_ids=input_ids.to(self.model.device), attention_mask=attention_mask.to(self.model.device), **self.generate_kwargs)
        preds = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return preds
    
    # def get_loss(self, input_ids, attention_mask):


    def _cal_label_prob(self, probs, labels):
        result = []
        for prob, label in zip(probs, labels):
            print("Prob: ", prob.shape)
            print("Label: ", label)
            mask = label > 0
            prob, label = prob[mask], label[mask]
            log_softmax = torch.nn.functional.log_softmax(prob, dim=-1)
            # from IPython import embed; embed(); exit(0)
            nll = -log_softmax.gather(1, label.unsqueeze(0).transpose(0, 1))
            # nll = -log_softmax.gather(1, label.unsqueeze(1)).squeeze(1)


            avg_nll = torch.sum(nll, dim=0) * -1
            result.append(float(torch.exp(avg_nll / float(label.shape[0]))))
        return result
    
    def get_scores(self, input_ids, label_ids):
        if input_ids.shape[1] != label_ids.shape[1]:
            min_len = min(input_ids.shape[1], label_ids.shape[1])
            input_ids = input_ids[:, :min_len]
            label_ids = label_ids[:, :min_len]

        outputs = self.model(input_ids=input_ids.to(self.model.device), labels=label_ids.to(self.model.device))
        scores = self._cal_label_prob(outputs.logits, label_ids.to(self.model.device))

        return scores
    
    def get_tokenizer(self):
        return self.tokenizer

class Reader_vLLM(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        _, tokenizer_cls, _, hf_name = _load_model(opt)
        self.model = LLM(model=os.path.join(opt.model_dir, hf_name), gpu_memory_utilization=0.70, kv_cache_dtype="fp8_e5m2")
        self.tokenizer = tokenizer_cls.from_pretrained(os.path.join(opt.model_dir, hf_name))
        self.gen_sampling = SamplingParams(temperature=1, top_p=1, max_tokens=30)
        self.score_sampling = SamplingParams(temperature=1, top_p=1, prompt_logprobs=0, max_tokens=1)

    def _cal_label_prob(self, outputs, labels):
        labels = [input_id[1:] for input_id in self.tokenizer(labels).input_ids]
        probs = [output.prompt_logprobs for output in outputs]
        result = []
        for prob, label in zip(probs, labels):
            prs = []
            for pr, l in zip(prob[-1 * len(label):], label):
                k,v = list(pr.items())[0]
                assert k == l
                prs.append(v)
            avg_nll = sum(prs)
            result.append(math.exp(avg_nll)/len(label))
        return result

    def forward(self, inputs):
        preds= [output.outputs[0].text.strip() for output in self.model.generate(inputs, use_tqdm=False, sampling_params=self.gen_sampling)]
        return preds
    
    def get_scores(self, inputs, labels):
        outputs = self.model.generate(inputs, use_tqdm=False, sampling_params=self.score_sampling)
        scores = self._cal_label_prob(outputs, labels)
        return scores
    
    def get_tokenizer(self):
        return self.tokenizer
    
class Reader_GPT(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        OPENAI_API_KEY = opt.openai_key
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.system_prompt = "You are a QA assistant. Read the document and answer the question. Your answer should be concise and short phrase, not sentence."
    

    def _cal_label_prob(self, outputs, labels):
        raise NotImplementedError
    
    def forward(self, contexts, question):
        preds = []
        for context in contexts:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Document: {}\nQuestion: {}".format(context, question)}
                ],
                logprobs=True
            )
            preds.append(completion.choices[0].message.content)
        return preds
    
    def get_scores(self, contexts, question, answers):
        from math import exp

        scores = []
        for context in contexts:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                n=10,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Document: {}\nQuestion: {}".format(context, question)}
                ],
                logprobs=True
            )
            score = 0
            for choice in completion.choices:
                pred = choice.message.content
                if f1(answers, pred) > 0.5:
                    for token in choice.logprobs.content:
                        score += token.logprob
                    score = exp(score)
                    break
            scores.append(score)
        return scores
    
    def get_tokenizer(self):
        raise NotImplementedError

class Read_Module(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        if opt.is_vllm:
            self.model = Reader(opt)
        else:
            self.model = Reader_vLLM(opt)
        self.is_vllm = opt.is_vllm
        logger.info("Model Load Done")

    # def forward(self, input_ids, attention_mask):
    #     preds = self.model(input_ids, attention_mask)
    #     return preds

    def predict_step(self, batch, batch_idx):
        if self.is_vllm:
            preds = self.model(batch['inputs'])
        else:
            preds = self.model(batch['input_ids'], batch['attention_mask'])
        result = self._process_output(preds, batch)
        return result
    
    def _process_output(self, preds, batch):
        keys = list(batch.keys())
        result = []
        for i in range(len(preds)):
            instance = {}
            for key in keys:
                if not isinstance(batch[key][i],torch.Tensor) and key in save_keys:
                    instance[key] = batch[key][i]
            instance["pred"] = preds[i]
            result.append(instance)
        # result = [{
        #     "question": batch["question"][i],
        #     "context": batch["context"][i],
        #     "answers": batch["answers"][i],
        #     "pred": preds[i],
        # }  for i in range(len(preds))]
        return result
    