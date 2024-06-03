import os
from ast import literal_eval
import pdb
import json
import requests
from tqdm import tqdm


class SearchAPI():
    def __init__(self):
        # invariant variables
        self.serper_key = os.getenv("SERPER_KEY_PRIVATE")
        self.url = "https://google.serper.dev/search"
        self.headers = {'X-API-KEY': self.serper_key,
                        'Content-Type': 'application/json'}
        # cache related
        self.cache_file = "data/cache/search_cache.json"
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 10

    def get_snippets(self, claim_lst):
        text_claim_snippets_dict = {}
        for query in claim_lst:
            search_result = self.get_search_res(query)
            if "statusCode" in search_result:  # and search_result['statusCode'] == 403:
                print(search_result['message'])
                exit()
            if "organic" in search_result:
                organic_res = search_result["organic"]
            else:
                organic_res = []

            search_res_lst = []
            for item in organic_res:
                title = item["title"] if "title" in item else ""
                snippet = item["snippet"] if "snippet" in item else ""
                link = item["link"] if "link" in item else ""

                search_res_lst.append({"title": title,
                                       "snippet": snippet,
                                       "link": link})
            text_claim_snippets_dict[query] = search_res_lst
        return text_claim_snippets_dict

    def get_search_res(self, query):
        # check if prompt is in cache; if so, return from cache
        cache_key = query.strip()
        if cache_key in self.cache_dict:
            # print("Getting search results from cache ...")
            return self.cache_dict[cache_key]

        payload = json.dumps({"q": query})
        response = requests.request("POST",
                                    self.url,
                                    headers=self.headers,
                                    data=payload)
        response_json = literal_eval(response.text)

        # update cache
        self.cache_dict[query.strip()] = response_json
        self.add_n += 1

        # save cache every save_interval times
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        return response_json

    def save_cache(self):
        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        cache = self.load_cache().items()
        for k, v in cache:
            self.cache_dict[k] = v
        print(f"Saving search cache ...")
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # load a json file
                cache = json.load(f)
                print(f"Loading cache ...")
        else:
            cache = {}
        return cache
