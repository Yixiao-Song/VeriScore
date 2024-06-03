"""
This script is written to extract claims from the model responses.
model generations: /data/yixiao/atomic_claims/data/model_generation_decomposition/model_generations
"""

import os
import json
import argparse
import spacy
from tqdm import tqdm
from .claim_extractor import ClaimExtractor
from .search_API import SearchAPI
from .claim_verifier import ClaimVerifier


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

class VeriScorer(object):
    def __init__(self,
                 model_name_extraction='gpt-4-0125-preview',
                 model_name_verification='gpt-4o',
                 data_dir='./data',
                 cache_dir='./data/cache',
                 output_dir='./data_cache',
                 label_n=3,
                 search_res_num=5):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.spacy_nlp = spacy.load('en_core_web_sm')

        self.system_message_extraction = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Wikipedia)"

        self.claim_extractor = ClaimExtractor(model_name_extraction, cache_dir=self.cache_dir)

        self.fetch_search = SearchAPI()

        demon_dir = os.path.join(self.data_dir, 'demos')
        self.model_name_verification = model_name_verification
        self.claim_verifier = ClaimVerifier(model_name=model_name_verification, label_n=label_n,
                                         cache_dir=self.cache_dir, demon_dir=demon_dir)
        self.label_n = label_n
        self.search_res_num = search_res_num

    def get_veriscore(self, data, input_file_name):

        ### extract claims ###
        output_file = f"claims_{input_file_name}.jsonl"
        output_path = os.path.join(self.output_dir, output_file)

        extracted_claims = []
        with open(output_path, "w") as f:
            for dict_item in tqdm(data):
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
                    extracted_claims.append(output_dict)
                    continue

                # extract claims
                snippet_lst, claim_lst_lst, all_claim_lst, prompt_tok_cnt, response_tok_cnt = self.claim_extractor.qa_scanner_extractor(
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
                extracted_claims.append(output_dict)

        print(f"claim extraction is done! saved to {output_path}")

        output_file = f"evidence_{input_file_name}.jsonl"
        output_path = os.path.join(self.output_dir, output_file)
        searched_evidence_dict = []
        with open(output_path, "w") as f:
            for dict_item in tqdm(extracted_claims):
                if dict_item['abstained']:
                    f.write(json.dumps(dict_item) + "\n")
                    searched_evidence_dict.append(dict_item)
                    continue

                claim_lst = dict_item["all_claim_lst"]
                if claim_lst == ["No verifiable claim."]:
                    dict_item["claim_snippets_dict"] = []
                    f.write(json.dumps(dict_item) + "\n")
                    searched_evidence_dict.append(dict_item)
                    continue
                claim_snippets = self.fetch_search.get_snippets(claim_lst)
                dict_item["claim_snippets_dict"] = claim_snippets
                searched_evidence_dict.append(dict_item)
                f.write(json.dumps(dict_item) + "\n")
                f.flush()
        print(f"evidence searching is done! saved to {output_path}")

        output_dir = os.path.join(args.output_dir, 'model_output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'verification_{input_file_name}_{self.model_name_verification}_{self.label_n}.jsonl'
        output_path = os.path.join(output_dir, output_file)

        total_prompt_tok_cnt = 0
        total_resp_tok_cnt = 0
        with open(output_path, "w") as f:
            for dict_item in tqdm(searched_evidence_dict):
                claim_snippets_dict = dict_item["claim_snippets_dict"]
                claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt = self.claim_verifier.verifying_claim(
                    claim_snippets_dict, search_res_num=args.search_res_num)
                json.dump(claim_verify_res_dict, f)
                f.write("\n")
                total_prompt_tok_cnt += prompt_tok_cnt
                total_resp_tok_cnt += response_tok_cnt

        print(f"claim verification is done! saved to {output_path}")
        print(f"Total cost: {total_prompt_tok_cnt * 10 / 1e6 + total_resp_tok_cnt * 30 / 1e6}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./data')
    parser.add_argument("--cache_dir", type=str, default='./data/cache')
    parser.add_argument("--model_name_extraction", type=str, default="gpt-4-0125-preview")
    parser.add_argument("--model_name_verification", type=str, default="gpt-4o")
    parser.add_argument("--label_n", type=int, default=3, choices=[2, 3])
    parser.add_argument("--search_res_num", type=int, default=5)
    args = parser.parse_args()


    vs = VeriScorer(model_name_extraction=args.model_name_extraction,
                    model_name_verification=args.model_name_verification,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    cache_dir=args.cache_dir,
                    label_n=args.label_n,
                    search_res_num=args.search_res_num)

    input_file_name = "".join(args.input_file.split('.')[:-1])
    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, "r") as f:
        data = [json.loads(x) for x in f.readlines() if x.strip()]

    vs.get_veriscore(data, input_file_name)