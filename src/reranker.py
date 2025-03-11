from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_core.documents import Document
from collections.abc import Iterable
import numpy as np


class ReRanker:
    def __init__(self, model_path, cache_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            # torch_dtype=torch.float16,
            cache_dir=cache_dir,
            device_map="auto",
            max_memory={
                # 0: torch.cuda.mem_get_info(0)[0]L, 
                1: torch.cuda.mem_get_info(1)[0], 
                2: torch.cuda.mem_get_info(2)[0],
                3: torch.cuda.mem_get_info(3)[0],
                4: torch.cuda.mem_get_info(4)[0],
                },
        ).eval()
        
    @property
    def device(self):
        return self.model.device
    
    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)
    
    def rerank(self, query: str, texts: Iterable[str], k: int) -> list[str]:
        '''
        query = 'what is panda?'
        texts = ['hi', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
        '''
        pairs = [[query, t] for t in texts]
        scores = self.compute_scores(pairs)
        rank = scores.argpartition(-k)[-k:]
        rank.sort()
        return np.array(texts)[rank].tolist()
    
    
    def rerank_documents(self, query: str, docs: Iterable[Document], k: int) -> list[Document]:
        '''
        query = 'what is panda?'
        docs = [Document, Document, ...]
        '''
        pairs = [[query, d.page_content] for d in docs]
        scores = self.compute_scores(pairs)
        rank = scores.argpartition(-k)[-k:]
        rank.sort()
        return np.array(docs)[rank].tolist()
    
    
    @torch.inference_mode()
    def compute_scores(self, pairs: list[list[str, str]]) -> np.ndarray:
        '''
        pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
        '''
        inputs = self.tokenizer.__call__(pairs, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy()
        return scores  