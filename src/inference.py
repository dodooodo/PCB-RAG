import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:21"

import warnings
warnings.filterwarnings("ignore")

import json, torch, gc
from opencc import OpenCC
from argparse import ArgumentParser
from tqdm import tqdm

from llm import LLM
from retriever import Retriever
from reranker import ReRanker
from rag_pipeline import RAG_Pipeline


EMBEDDING_MODEL_PATH = r"BAAI/bge-m3"
RERANKER_ID = 'BAAI/bge-reranker-v2-m3'
CACHE_DIR = 'cache_dir'
CC = OpenCC('t2s')


def get_rag_pipeline(db_path, llm, retriv_k, retriv_n, rerank_k):
    SYSTEM_PROMPT = CC.convert("你是一個專業的PCB印製電路板製造工程師，你只會講中文")
    USER_PROMPT = CC.convert("請閱讀以下內容:\n'''{contexts}'''\n根據你讀到的內容用中文回答以下問題，回答盡量簡短:\n{query}")

    retriever = Retriever(EMBEDDING_MODEL_PATH, db_path, cache_folder=CACHE_DIR)
    reranker = ReRanker(RERANKER_ID, cache_dir=CACHE_DIR)
    llm = LLM(llm, system_prompt=SYSTEM_PROMPT, cache_dir=CACHE_DIR)
    
    return RAG_Pipeline(llm, retriever, reranker, USER_PROMPT, retriv_k, retriv_n, rerank_k)


def run_pcb40(db_path, output_dir, llm, retriv_k, retriv_n, rerank_k):
    print(f'retriv_k: {retriv_k}, retriv_n: {retriv_n}, rerank_k: {rerank_k}')
    
    questions = list(json.load(open('question.json')).values())
    groud_truth = list(json.load(open('ground_truth.json')).values())
    
    rag = get_rag_pipeline(db_path, llm=llm, retriv_k=retriv_k, retriv_n=retriv_n, rerank_k=rerank_k)
    fname = "{}/k{}n{}rk{}-beam6.json".format(output_dir, retriv_k, retriv_n, rerank_k)
    if os.path.exists(fname):
        with open(fname) as fp:
            i = len(fp.readlines())
        fp = open(fname, "a")
    else:
        fp = open(fname, "w")
        i = 0
    
    for q, g in tqdm(zip(questions[i:], groud_truth[i:])):
        output = rag.run_one_query(query=q, merge_docs=True)
        contexts = output['contexts']
        result = output['llm_output']
        print(len(''.join(contexts)))
        
        rd = {}
        rd['question'] = CC.convert(q)
        rd['answer'] = CC.convert(result)
        rd['ground_truth'] = CC.convert(g)
        rd['contexts'] = contexts
        
        # print(result, flush=True)
        print(json.dumps(rd, ensure_ascii=False), file=fp, flush=True)
        
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = ArgumentParser()
    parser.add_argument('--db_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--llm')
    parser.add_argument('--retriv_k', type=int)
    parser.add_argument('--retriv_n', type=int)
    parser.add_argument('--rerank_k', type=int)
    args = parser.parse_args()
    run_pcb40(**vars(args))
    

if __name__ == "__main__":
    main()