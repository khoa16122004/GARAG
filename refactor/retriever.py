import faiss
import torch
from typing import List, Tuple
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dataset import TriviaQA


class DPRRetriever:
    def __init__(self, dataset_dir: str = "chunks" , index_dir: str = "index", model_name: str = "facebook/dpr-question_encoder-single-nq-base", use_gpu: bool = False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.q_encoder = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
        self.contexts = []
        self.index = None
        self.id_map = []
        self.index_dir = index_dir
        self.dataset_dir = dataset_dir
        self.index = None

    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.q_tokenizer(query, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.q_encoder(**inputs).pooler_output
        return embeddings.cpu().numpy()

    def build_index(self, index: int, documents: List[str], batch_size: int = 32):
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(self.device)

        self.contexts = documents
        self.id_map = list(range(len(documents)))

        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            inputs = ctx_tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                emb = ctx_encoder(**inputs).pooler_output
            all_embeddings.append(emb.cpu().numpy())

        embeddings_matrix = np.vstack(all_embeddings).astype('float32')

        index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        faiss.normalize_L2(embeddings_matrix)
        index.add(embeddings_matrix)
        
        output_path = os.path.join(self.index_dir, f"{index}.faiss")
        faiss.write_index(index, output_path)
        print(f"Index saved to {output_path}")
    
    def load_index(self, index: int):
        self.index = faiss.read_index(os.path.join(self.index_dir, f"{index}.faiss"))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        if not self.index:
            raise ValueError("Index not loaded. Please load the index before retrieval.")

        query_vec = self.encode_query(query)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            doc_id = self.id_map[idx]
            content = self.contexts[doc_id]
            results.append((doc_id, float(score), content))
        return results

retriever = DPRRetriever()

a.build_index(0)