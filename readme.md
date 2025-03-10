## Create Dense Vector Database
Run `create_chunks_embeddings_with_boundary.ipynb`

The format of `data_title_content.json`:
```
{'data': [
    {
        'title': '以工艺窗口建模探索路径...',
        'content': '\n作者：泛林集团 ...'
    },
    {
        'title': 'X-FAB在制造工艺...',
        'content': '\n中国北京，2023...'
    },
    ...
]}
```
## Run RAG Pipeline
Run `rag.sh`.

## Evaluation (RAGAS)
Run `ragas.sh`. It runs `run_ragas_openai-slow.py` which will evaluate the answer one-by-one. The evaluation results will be saved seperately and also the average score. During the evalutation `temp.json` is created.