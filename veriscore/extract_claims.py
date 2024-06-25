"""
This script is written to extract claims from the model responses.
"""

import os
import json
import argparse
from tqdm import tqdm
from .claim_extractor import ClaimExtractor

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
    parser.add_argument("--use_external_model", action='store_true')
    args = parser.parse_args()

    input_file_name = "".join(args.input_file.split('.')[:-1])

    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, "r") as f:
        data = [json.loads(x) for x in f.readlines() if x.strip()]

    # initialize objects
    model_name = args.model_name
    claim_extractor = ClaimExtractor(model_name, args.cache_dir, args.use_external_model)

    output_dir = args.output_dir
    output_file = f"claims_{input_file_name}.jsonl"
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w") as f:
        for dict_item in tqdm(data):
            # get necessary info

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

            if "question" in dict_item and dict_item["question"]:
                question = dict_item["question"]
                snippet_lst, claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = claim_extractor.qa_scanner_extractor(
                    question, response)
            else:
                question = ''
                snippet_lst, claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = claim_extractor.non_qa_scanner_extractor(
                    response)

            # write output
            output_dict = {"question": question.strip(),
                           "prompt_source": prompt_source,
                           "response": response.strip(),
                           "prompt_tok_cnt": prompt_tok_cnt,
                           "response_tok_cnt": response_tok_cnt,
                           "model": model,
                           "abstained": False,  # "abstained": False, "abstained": True
                           "claim_list": claim_list,
                           "all_claims": all_claims
                           }
            f.write(json.dumps(output_dict) + "\n")
    print(f"extracted claims are saved at {output_path}")