import json
import os
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

class TriviaQA:
    def __init__(self, json_path: str, data_dir: str):
        self.json_path = json_path
        self.data_dir = data_dir
        self.documents_files = []
        self.queries = []
        self.answers = []
        self.load_data()
        
    def load_data(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)['Data']
            
            for item in json_data:
                self.queries.append(item.get('Question', ''))
                gt_files = []
                for document in item.get('EntityPages', []):
                    filename = document.get('Filename', '')
                    gt_files.append(filename)
                self.documents_files.append(gt_files)
                            
                self.answers.append(item.get('Answer', '').get('Value'))

    def yield_data(self):
        for query, document, answer in zip(self.queries, self.documents_files, self.answers):
            yield query, document, answer
            
    def __getitem__(self, index):
        if index < 0 or index >= len(self.queries):
            raise IndexError("Index out of range")
        return self.queries[index], self.documents_files[index], self.answers[index]

    def write_chunks(self, index, output_dir="chunks"):
        index_output_dir = os.path.join(output_dir, f"{index}")
        os.makedirs(index_output_dir, exist_ok=True)
    
        question, documents, answer = self.__getitem__(index)
        document_paths = [os.path.join(self.data_dir, document) for document in documents]
        for document_path in document_paths:
            with open(document_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = sentence_based_chunk(text)
                for i, chunk in enumerate(chunks):
                    with open(os.path.join(index_output_dir, f"chunk_{i}.txt"), 'w', encoding='utf-8') as chunk_file:
                        chunk_file.write(chunk)
    
        
    
def sentence_based_chunk(text, max_tokens=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


    
    
# dataset = TriviaQA(r"D:\GARAG\triviaqa-rc\qa\wikipedia-dev.json", r"D:\GARAG\triviaqa-rc\evidence\wikipedia")
# dataset.write_chunks(0)