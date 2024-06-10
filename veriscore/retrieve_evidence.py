import os
import pdb
import json
import random
import argparse
from tqdm import tqdm
from .search_API import SearchAPI

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./data')
    args = parser.parse_args()

    """
    Get search results for each claim
    Store as a dictionary {claim: {"search_results": [search_results]}}
    """
    # initialize search api
    fetch_search = SearchAPI()

    input_file_name = "".join(args.input_file.split('.')[:-1])
    input_path = os.path.join(args.data_dir, args.input_file)

    # read in extracted claim data
    with open(input_path, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    output_dir = args.output_dir
    output_file = f"evidence_{input_file_name}.jsonl"
    output_path = os.path.join(output_dir, output_file)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with open(output_path, "w") as f:
        for dict_item in tqdm(data):
            if dict_item['abstained']:
                f.write(json.dumps(dict_item) + "\n")
                continue

            claim_lst = dict_item["all_claims"]
            if claim_lst == ["No verifiable claim."]:
                dict_item["claim_search_results"] = []
                f.write(json.dumps(dict_item) + "\n")
                continue
            claim_snippets = fetch_search.get_snippets(claim_lst)
            dict_item["claim_search_results"] = claim_snippets

            f.write(json.dumps(dict_item) + "\n")
            f.flush()
