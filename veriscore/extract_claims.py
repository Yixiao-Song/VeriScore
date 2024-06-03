"""
This script is written to extract claims from the model responses.
model generations: /data/yixiao/atomic_claims/data/model_generation_decomposition/model_generations
"""

import os
import json
import argparse
from tqdm import tqdm
from .claim_extractor import ClaimExtractor

input_file_names = ['Mistral-7B-Instruct-v0.1',
                    'Mistral-7B-Instruct-v0.2',
                    'Mixtral-8x7B-Instruct-v0.1',
                    'Mixtral-8x22B-Instruct-v0.1',
                    'gpt-4-0125-preview',
                    'gpt-3.5-turbo-1106',
                    'gpt-3.5-turbo-0613',
                    'claude-3-opus-20240229',
                    'claude-3-sonnet-20240229',
                    'claude-3-haiku-20240307',
                    'dbrx-instruct',
                    'OLMo-7B-Instruct', ]

abstain_responses = ["I'm sorry, I cannot fulfill that request.",
                     "I'm sorry, I can't fulfill that request.",
                     "I'm sorry, but I cannot fulfill that request.",
                     "I'm sorry, but I can't fulfill that request.",
                     "Sorry, but I can't fulfill that request.",
                     "Sorry, I can't do that."]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./data')
    parser.add_argument("--cache_dir", type=str, default='./data/cache')
    parser.add_argument("--model_name", type=str, default="gpt-4-0125-preview")
    args = parser.parse_args()

    input_file_name = "".join(args.input_file.split('.')[:-1])

    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, "r") as f:
        data = [json.loads(x) for x in f.readlines() if x.strip()]

    # initialize objects
    model_name = args.model_name
    claim_extractor = ClaimExtractor(model_name, args.cache_dir)

    output_dir = args.output_dir
    output_file = f"claims_{input_file_name}.jsonl"
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w") as f:
        for dict_item in tqdm(data):
            # get necessary info
            question = dict_item["question"]
            response = dict_item["response"]
            prompt_source = dict_item["prompt_source"]
            model = dict_item["model"]

            # skip abstained responses
            if response.strip() in abstain_responses:
                output_dict = {"question": question.strip(),
                               "response": response.strip(),
                               "abstained": True,
                               "prompt_source": prompt_source,
                               "model": model, }
                f.write(json.dumps(output_dict) + "\n")
                continue

            # extract claims
            snippet_lst, claim_lst_lst, all_claim_lst, prompt_tok_cnt, response_tok_cnt = claim_extractor.qa_scanner_extractor(
                question, response)

            # write output
            output_dict = {"question": question.strip(),
                           "prompt_source": prompt_source,
                           "response": response.strip(),
                           "prompt_tok_cnt": prompt_tok_cnt,
                           "response_tok_cnt": response_tok_cnt,
                           "model": model,
                           "abstained": False,  # "abstained": False, "abstained": True
                           "claim_lst_lst": claim_lst_lst,
                           "all_claim_lst": all_claim_lst
                           }
            f.write(json.dumps(output_dict) + "\n")