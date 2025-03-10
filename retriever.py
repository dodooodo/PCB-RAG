from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class Retriever:
    def __init__(self, model_name, db_folder_path=None, cache_folder=None):   
        self.embeddings = HuggingFaceEmbeddings(
            model_name = model_name,
            # model_kwargs = {"device": "cuda"},
            # 
            encode_kwargs = {'normalize_embeddings': True, 'batch_size': 1},
            # 
            cache_folder = cache_folder,    
        )
        if db_folder_path:
            self.db = FAISS.load_local(db_folder_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.db = None
        
    @property
    def device(self):
        return self.embeddings.client.device
    
    # def retrieve(
    #         self, 
    #         query: str, 
    #         k: int, 
    #         n: int, 
    #         max_char_length=7000,
    #     ) -> list[str]:
    #     '''
    #         Retrieve topK sentences -> get topN contexts from topK
    #     '''
    #     docs = self.retrieve_topk(query, k)
    #     docs = self.get_topN(docs, n)
    #     docs = self.remove_duplicated_doc_content(docs)
    #     contexts = [doc.page_content for doc in docs]
        
    #     def truncate(contexts, max_char_length):
    #         n_contexts = len(contexts)
    #         contexts_word_length = 0
    #         for i in range(n_contexts):
    #             contexts_word_length += len(contexts[i])
    #             if contexts_word_length > max_char_length:
    #                 return contexts[:i]
    #         return contexts
    #     return truncate(contexts, max_char_length)
    
    
    def retrieve_topk(self, query: str, k: int) -> list[Document]:
        '''
            get topk results through similarity search
        '''
        if k > 0:
            docs = self.db.search(query, 'similarity', k=k)
        else:
            docs = [Document('')]
        return docs
    
    
    def get_topN(self, topk_docs: list[Document], n: int) -> list[Document]:
        '''
            get topN from topK results
        '''
        if n > 0:
            docs = []
            for doc in topk_docs:
                docs += self.get_contexts(doc, n)
                
        elif n == 0:
            docs = topk_docs
            
        else:
            raise Exception(f'n = "{n}" is not an integer > 0')
            
        return docs
    
    
    # def get_contexts(self, doc: Document, n: int = 5) -> list[Document]:
    #     '''        
    #         modified from langchain_community.vectorstores.faiss 
    #             -> class FAISS -> func similarity_search_with_score_by_vector
    #     '''
    #     center_index = doc.metadata['seq_num']
    #     docs = []
    #     for i in range(center_index-n-1, center_index+n):
    #         id = self.db.index_to_docstore_id[i]
    #         doc = self.db.docstore.search(id)
    #         docs.append(doc)
    #     return docs
    
    def get_contexts(self, doc: Document, n: int) -> list[Document]:
        '''        
            Get contexts of the document.
        '''
        center_data_id = doc.metadata['data_id']
        post_id = doc.metadata['post_id']
        docs = []
        for data_id in range(center_data_id-n-1, center_data_id+n):
            if data_id < 0:
                continue
            id = self.db.index_to_docstore_id[data_id]
            doc = self.db.docstore.search(id)
            if doc.metadata['post_id'] == post_id:
                docs.append(doc)
        return docs
    
    
    # def remove_duplicated_doc_content(self, docs: list[Document]) -> list[Document]:
    #     docs = {doc.page_content: doc for doc in docs}
    #     docs = list(docs.values())
    #     return docs


