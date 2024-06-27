import os
import json
import argparse

from collections import defaultdict

from tqdm import tqdm

from veriscore import utils
from .claim_verifier import ClaimVerifier

if __name__ == '__main__':
    """
    add argparse for label_n and model_name
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./data')
    parser.add_argument("--cache_dir", type=str, default='./data/cache')
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--label_n", type=int, default=3, choices=[2, 3])
    parser.add_argument("--search_res_num", type=int, default=10)
    parser.add_argument("--use_external_model", action='store_true')
    args = parser.parse_args()

    model_name = args.model_name
    label_n = args.label_n

    input_file_name = "".join(args.input_file.split('.')[:-1])
    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, "r") as f:
        input_data = [json.loads(line) for line in f if line.strip()]
    demon_dir = os.path.join(args.data_dir, 'demos')
    claim_verifier = ClaimVerifier(model_name=model_name, label_n=label_n,
                                   cache_dir=args.cache_dir, demon_dir=demon_dir,
                                   use_external_model=args.use_external_model)

    output_dir = os.path.join(args.output_dir, 'model_output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'verification_{input_file_name}_{label_n}.jsonl'
    output_path = os.path.join(output_dir, output_file)

    model_domain_triplet_dict = defaultdict(
        lambda: defaultdict(list))  # triplet = [[supported, total, # of sentences], ...]

    total_prompt_tok_cnt = 0
    total_resp_tok_cnt = 0

    with open(output_path, "w") as f:
        for dict_item in tqdm(input_data):
            model_name = dict_item['model']
            domain = dict_item['prompt_source']
            claim_search_results = dict_item["claim_search_results"]

            if dict_item['abstained']:
                f.write(json.dumps(dict_item) + "\n")
                continue

            claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt = claim_verifier.verifying_claim(
                claim_search_results, search_res_num=args.search_res_num)
            dict_item["claim_verification_result"] = claim_verify_res_dict

            f.write(json.dumps(dict_item) + "\n")

            total_prompt_tok_cnt += prompt_tok_cnt
            total_resp_tok_cnt += response_tok_cnt

            ## for VeriScore calculation
            triplet = [0, 0, 0]
            triplet[1] = len(dict_item['all_claims'])
            triplet[2] = len(dict_item['claim_list'])
            if not dict_item['claim_search_results']:
                triplet[0] = 0
            else:
                for claim_veri_res in dict_item['claim_verification_result']:
                    if claim_veri_res['verification_result'] == "supported":
                        triplet[0] += 1
            model_domain_triplet_dict[domain][model_name].append(triplet)


    print(f"claim verification is done! saved to {output_path}")
    utils.get_veriscore(model_domain_triplet_dict)
    print(f"Total cost: {total_prompt_tok_cnt * 10 / 1e6 + total_resp_tok_cnt * 30 / 1e6}")
