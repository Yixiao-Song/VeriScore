import os
import pdb
import json
import tiktoken
from openai import OpenAI
from anthropic import Anthropic


class GetResponse():
    # OpenAI model list: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    # Claude3 model list: https://www.anthropic.com/claude
    # "claude-3-opus-20240229", gpt-4-0125-preview
    def __init__(self, cache_file, model_name="gpt-4-0125-preview", max_tokens=1000, temperature=0):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_file = cache_file

        # invariant variables
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        # model keys
        if "gpt" in model_name:
            self.key = os.getenv("OPENAI_API_KEY_PERSONAL")
            self.client = OpenAI(api_key=self.key)
            self.seed = 1130
        elif "claude" in model_name:
            self.key = os.getenv("CLAUDE_API_KEY")
            self.client = Anthropic(api_key=self.key)
        # cache related
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 1
        self.print_interval = 20

    # Returns the response from the model given a system message and a prompt text.
    def get_response(self, system_message, prompt_text, cost_estimate_only=False):
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        if cost_estimate_only:
            # count tokens in prompt and response
            response_tokens = 0
            return None, prompt_tokens, response_tokens

        # check if prompt is in cache; if so, return from cache
        cache_key = prompt_text.strip()
        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key], 0, 0

        # Example message: "You are a helpful assistant who can extract verifiable atomic claims from a piece of text."
        if "gpt" in self.model_name:
            message = [{"role": "system", "content": system_message},
                       {"role": "user", "content": prompt_text}]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
            )
            response_content = response.choices[0].message.content.strip()
        elif "claude" in self.model_name:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response_content = response.content[0].text.strip()

        # update cache
        self.cache_dict[cache_key] = response_content.strip()
        self.add_n += 1

        # save cache every save_interval times
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        if self.add_n % self.print_interval == 0:
            print(f"Saving # {self.add_n} cache to {self.cache_file}...")

        response_tokens = len(self.tokenizer.encode(response_content))
        return response_content, prompt_tokens, response_tokens

    # Returns the number of tokens in a text string.
    def tok_count(self, text: str) -> int:
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

    def save_cache(self):
        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # load a json file
                cache = json.load(f)
                print(f"Loading cache from {self.cache_file}...")
        else:
            cache = {}
        return cache
