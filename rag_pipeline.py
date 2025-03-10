from langchain_core.documents import Document

from llm import LLM
from retriever import Retriever
from reranker import ReRanker


class RAG_Pipeline:
    def __init__(
        self,
        llm: LLM, 
        retriever: Retriever, 
        reranker: ReRanker = None,
        user_prompt: str = '',
        retriv_k: int = 100,
        retriv_n: int = 0,
        rerank_k: int = 10,
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        
        assert "{query}" in user_prompt and "{contexts}" in user_prompt, \
            "The user_prompt must contain '{query}' and '{contexts}' placeholders."
        self.user_prompt = user_prompt
        
        self.retriv_k = retriv_k
        self.retriv_n = retriv_n
        self.rerank_k = rerank_k
        
        
    def run_one_query(self, query: str, merge_docs=False) -> dict:
        '''
        Run one query.
        '''
        docs = self.retriever.retrieve_topk(query, self.retriv_k)
        
        if self.reranker:
            docs = self.reranker.rerank_documents(query, docs, self.rerank_k)
            
        if self.retriv_n > 0:
            docs = self.retriever.get_topN(docs, self.retriv_n)
        
        docs = self.remove_duplicate(docs)
        
        if merge_docs:
            rag_input = self.merge_overlap_docs(docs)
        else:
            rag_input = self.docs_to_text(docs)
        
        query = self.user_prompt.format(contexts=rag_input, query=query)
        llm_output = self.llm(query)
        return {'llm_output': llm_output, 'contexts': [d.page_content for d in docs]}
        
    
    def merge_overlap_docs(self, docs: list[Document]) -> str:
        '''
        Sort chunks and merge them if they contain overlapping contents.
        '''
        docs.sort(key=lambda d: d.metadata['data_id'])
        merged_string = docs[0].page_content
        for i in range(len(docs[:-1])):
            merged_string = self.merge_strings(merged_string, docs[i+1].page_content)
        return merged_string
        
        
    def merge_strings(self, s1: str, s2: str) -> str:
        def find_overlap(s1, s2):
            # Find the maximum possible overlap length
            max_overlap_len = min(len(s1), len(s2))
            
            # Iterate from the longest possible overlap to the shortest
            for i in range(max_overlap_len, 0, -1):
                if s1[-i:].lower() == s2[:i].lower():  # Case-insensitive comparison
                    return i
            return 0  # No overlap found
        
        overlap_len = find_overlap(s1, s2)
        merged_string = s1 + s2[overlap_len:]
        return merged_string
    
    
    def remove_duplicate(self, docs: list[Document]) -> list[Document]:
        '''
        Remove duplicated chunks.
        '''
        docs = {doc.page_content: doc for doc in docs}
        docs = list(docs.values())
        return docs


    def docs_to_text(self, docs: list[Document]) -> str:
        '''
        Combine documents to a single string for LLM's input.
        '''
        return ''.join([d.page_content for d in docs])