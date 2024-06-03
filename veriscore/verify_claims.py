import os
import json
import argparse
from tqdm import tqdm
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
    parser.add_argument("--search_res_num", type=int, default=5)
    args = parser.parse_args()

    model_name = args.model_name
    label_n = args.label_n

    input_file_name = "".join(args.input_file.split('.')[:-1])
    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, "r") as f:
        input_data = [json.loads(line) for line in f if line.strip()]
    demon_dir = os.path.join(args.data_dir, 'demos')
    claim_verifier = ClaimVerifier(model_name=model_name, label_n=label_n,
                                   cache_dir=args.cache_dir, demon_dir=demon_dir)

    output_dir = os.path.join(args.output_dir, 'model_output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'verification_{input_file_name}_{model_name}_{label_n}.jsonl'
    output_path = os.path.join(output_dir, output_file)

    total_prompt_tok_cnt = 0
    total_resp_tok_cnt = 0
    with open(output_path, "w") as f:
        for dict_item in tqdm(input_data):
            claim_snippets_dict = dict_item["claim_snippets_dict"]
            claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt = claim_verifier.verifying_claim(
                claim_snippets_dict, search_res_num=args.search_res_num)
            json.dump(claim_verify_res_dict, f)
            f.write("\n")
            total_prompt_tok_cnt += prompt_tok_cnt
            total_resp_tok_cnt += response_tok_cnt

    print(f"Total cost: {total_prompt_tok_cnt * 10 / 1e6 + total_resp_tok_cnt * 30 / 1e6}")
