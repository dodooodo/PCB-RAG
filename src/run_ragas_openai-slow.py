import os
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_ORGANIZATION'] = ''
from ragas.metrics import (
    context_recall,
    context_relevancy,
    context_precision,
    answer_correctness,
    faithfulness,
    answer_relevancy
)
from ragas import evaluate, RunConfig
from datasets import load_dataset, Dataset
import json
from argparse import ArgumentParser
import numpy as np
import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--output_detailed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    ragas_dataset = load_dataset('json', data_files=args.input)
    data = ragas_dataset['train']
    fp_output_detailed = open(args.output_detailed, 'w')

    results = []
    for i in range(40):
        data_point = data[i]
        
        json.dump(data_point, open('temp.json', 'w'))
        data_point = load_dataset('json', data_files='temp.json')['train']

        metrics=[
                answer_correctness, 
                answer_relevancy,
                context_recall,
                context_relevancy,
                context_precision,
                faithfulness,
            ]

        result = evaluate(
            data_point,
            metrics=metrics,
            raise_exceptions=False,
            is_async=True,
            # run_config=RunConfig(timeout=120),
        )
        results.append(result)
        
        print(json.dumps(result, ensure_ascii=False), file=fp_output_detailed, flush=True)
        
        # time.sleep(10)

    result = {key:np.nanmean([r[key] for r in results]) for key in results[0].keys()}
    print(result)

    json.dump(result, open(args.output, "w"))