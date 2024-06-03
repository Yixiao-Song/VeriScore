import os
import pdb
import json
from .get_response import GetResponse


class ClaimVerifier():
    def __init__(self, model_name, label_n=2, cache_dir="./data/cache/", demon_dir="data/demos/"):
        cache_dir = os.path.join(cache_dir, model_name)
        os.makedirs(cache_dir, exist_ok=True)
        self.model_name = model_name
        self.label_n = label_n
        self.cache_file = os.path.join(cache_dir, "claim_verification_cache.json")
        self.demon_path = os.path.join(demon_dir, 'few_shot_examples.jsonl')
        self.get_model_response = GetResponse(cache_file=self.cache_file,
                                              model_name=model_name,
                                              max_tokens=1000,
                                              temperature=0)
        self.system_message = "You are a helpful assistant who can verify the truthfulness of a claim against reliable external world knowledge."
        self.prompt_initial_temp = self.get_initial_prompt_template()

    def get_instruction_template(self):
        prompt_temp = ''
        if self.label_n == 2 and "claude" in self.model_name:
            prompt_temp = open("./prompt/verification_instruction_claude_binary.txt", "r").read()
        elif self.label_n == 3 and "claude" in self.model_name:
            prompt_temp = open("./prompt/verification_instruction_claude_trinary.txt", "r").read()
        elif self.label_n == 2 and "claude" not in self.model_name:
            prompt_temp = open("./prompt/verification_instruction_binary.txt", "r").read()
        elif self.label_n == 3 and "claude" not in self.model_name:
            prompt_temp = open("./prompt/verification_instruction_trinary.txt", "r").read()
        return prompt_temp

    def get_initial_prompt_template(self):
        prompt_temp = self.get_instruction_template()
        with open(self.demon_path, "r") as f:
            example_data = [json.loads(line) for line in f if line.strip()]
        element_lst = []
        for dict_item in example_data:
            claim = dict_item["claim"]
            search_result_str = dict_item["search_result"]
            human_label = dict_item["human_label"]
            if self.label_n == 2:
                if human_label == "support":
                    human_label = "Supported."
                else:
                    human_label = "Unsupported."
            if "claude" in self.model_name:
                element_lst.extend([search_result_str, claim, human_label])
            else:
                element_lst.extend([claim, search_result_str, human_label])

        prompt_few_shot = prompt_temp.format(*element_lst)

        if "claude" in self.model_name:
            self.your_task = "Your task:\n\n{search_results}\n\nClaim: {claim}\n\nTask: Given the search results above, is the claim supported or unsupported? Mark your decision with ### signs.\n\nYour decision:"
        else:
            self.your_task = "Your task:\n\nClaim: {claim}\n\n{search_results}\n\nYour decision:"

        """
        choose system message
        """
        if self.label_n == 2:
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported by the search results or not."
        elif self.label_n == 3:
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported or contradicted by the search results, or whether there is no enough information to make a judgement."

        return prompt_few_shot

    def verifying_claim(self, claim_snippets_dict, search_res_num=5):
        """
        search_snippet_lst = [{"title": title, "snippet": snippet, "link": link}, ...]
        """
        assert search_res_num <= 9, "search_res_num should be less than or equal to 9"

        prompt_tok_cnt, response_tok_cnt = 0, 0
        claim_verify_res_dict = {}
        for claim, search_snippet_lst in claim_snippets_dict.items():
            search_res_str = ""
            search_cnt = 1
            for search_dict in search_snippet_lst:
                search_res_str += f'Search result {search_cnt}\nTitle: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                search_cnt += 1

            prompt_tail = self.your_task.format(
                claim=claim,
                search_results=search_res_str.strip(),
            )
            prompt = f"{self.prompt_initial_temp}\n\n{prompt_tail}"
            response, prompt_tok_num, response_tok_num = self.get_model_response.get_response(self.system_message,
                                                                                              prompt)
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num
            claim_verify_res_dict[claim] = {"search_results": search_res_str,
                                            "response": response}
        return claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt
