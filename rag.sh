OUTPUT_DIR="rag_output/mistral-0.3/sentence";
LLM="mistralai/Mistral-7B-Instruct-v0.3";
DB_PATH="data/embeddings/bge-sentence-with-boundary";

# python inference.py --db_path $DB_PATH --output_dir $OUTPUT_DIR --llm $LLM --retriv_k 100 --retriv_n 10 --rerank_k 5;



OUTPUT_DIR="rag_output/mistral-0.3/chunk512";
DB_PATH="data/embeddings/bge-chunk512-overlap25%-boundary";

python inference.py --db_path $DB_PATH --output_dir $OUTPUT_DIR --llm $LLM --retriv_k 50 --retriv_n 0 --rerank_k 10;