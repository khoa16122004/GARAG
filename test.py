import random
from src.option import Options
from src.util import init_logger, timestr
from src.task import ReaderDataset, evaluate
from src.attacker import build_attack
from textattack.augmentation import Augmenter
from textattack.attack_args import AttackArgs

import tqdm
import os
import json
import logging

from textattack.metrics.quality_metrics import Perplexity, USEMetric
from textattack.shared import AttackedText, utils

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    option = Options("attack")
    opt, message = option.parse(timestr()) # create 
    logger = init_logger(opt)
    logger.info(message)
    logger.info("The name of experiment is {}".format(opt.name))
    logger.info("Attack type is {}".format(opt.method))

    # Load dataset
    dataset = ReaderDataset(opt) # get document dataset
    
    # Build attack and dataset
    attack, dataset = build_attack(opt, dataset)

    for i, d in enumerate(dataset):
        answers = d["answers"]     # list of answers     
        question = d["question"]   # question
        ctxs = d["ctxs"]           # list of contexts   
        q_id = i                   # id
        texts = [ctx["context"] for ctx in ctxs]
        
        golds_preds = attack.goal_function.generate(texts, question)
        
        scores = attack.goal_function.eval(texts, question, answers[0])
        
        # Gộp lại để sort
        results = []
        for ctx, score, golds_pred in zip(ctxs, scores, golds_preds):
            results.append({
                "context": ctx["context"],
                "score": score,
                "gold_pred": golds_pred
            })
        
        # Sort theo score[0] giảm dần
        results_sorted = sorted(results, key=lambda x: x["score"][0], reverse=True)
        
        for item in results_sorted:
            print("="*40)
            print("Score:", item["score"])
            print("Gold Pred:", item["gold_pred"])
            print("Context:", item["context"])
        break

    print("Attacker and dataset done")

if __name__=="__main__":
    main()