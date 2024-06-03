import os
import regex
import pdb
import json
import spacy
from tqdm import tqdm
from .get_response import GetResponse


class ClaimExtractor():
    def __init__(self, model_name, cache_dir="./data/cache/"):
        cache_dir = os.path.join(cache_dir, model_name)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"claim_extraction_cache.json")
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.get_model_response = GetResponse(cache_file=self.cache_file,
                                              model_name=model_name,
                                              max_tokens=1000,
                                              temperature=0)
        self.system_message = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Wikipedia)"

    def non_qa_scanner_extractor(self, response):
        """
        Given a model output
        - split by \n into paragraphs
        - split the paragraphs into sentences using spaCy
        - go para by para, always add the first sent of the para into context1
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """
        # split response into paras & clean out empty strings
        paragraph_lst = [x.strip() for x in response.split("\n") if x.strip() != ""]

        all_facts_lst = []
        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0
        for para in paragraph_lst:
            # split the text into sentences using spaCy
            sentences = self.get_sentence(para)
            for i in range(len(sentences)):
                lead_sent = sentences[0]  # 1st sentence of the para
                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])

                # if the para is not long
                if len(sentences) <= 5:
                    snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
                # if the para is long, add lead sentence to context1
                else:
                    snippet = f"{lead_sent.strip()} {context1.strip()} {sentence.strip()} {context2.strip()}".strip()

                # call fact_extractor on each snippet
                facts, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(),
                                                                              qa_input=False)

                # update token counts
                prompt_tok_cnt += prompt_tok_num
                response_tok_cnt += response_tok_num

                if facts == None:
                    continue

                # deduplication
                for fact in facts:
                    if fact.strip() == "":
                        continue
                    # cases where GPT returns its justification
                    elif fact.startswith("Note:"):
                        continue
                    elif fact not in all_facts_lst:
                        all_facts_lst.append(fact)

        print(f"Returning facts and token counts for the whole response ...")
        if all_facts_lst == None:
            return None, prompt_tok_cnt, response_tok_cnt
        else:
            return all_facts_lst, prompt_tok_cnt, response_tok_cnt

    def qa_scanner_extractor(self, question, response):
        """
        Given a model output to a question
        - split the response into sentences using spaCy
        - snippet = question (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """
        all_facts_lst = []
        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0
        sentences = self.get_sentence(response)

        # new return values
        snippet_lst = []
        fact_lst_lst = []
        for i in range(len(sentences)):
            context1 = " ".join(sentences[max(0, i - 3):i])
            sentence = f"<SOS>{sentences[i].strip()}<EOS>"
            context2 = " ".join(sentences[i + 1:i + 2])

            snippet = f"Question: {question.strip()}\nResponse: {context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            # new return value
            snippet_lst.append(snippet)

            # call fact_extractor on each tesnippetxt
            facts, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(), qa_input=True)

            # update token counts
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            if facts == None:
                fact_lst_lst.append([None])
                continue

            # deduplication
            fact_lst = []
            for fact in facts:
                if fact.strip() == "":
                    continue
                # cases where GPT returns its justification
                elif fact.startswith("Note:"):
                    continue
                elif fact not in all_facts_lst:
                    all_facts_lst.append(fact.strip())
                fact_lst.append(fact.strip())
            fact_lst_lst.append(fact_lst)
        print(f"Returning facts and token counts for the whole response ...")

        return snippet_lst, fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt

    def get_sentence(self, text):
        # use spaCy to split the text into sentences
        return [x.text.strip() for x in self.spacy_nlp(text).sents]

    def get_prompt_template(self, qa_input):
        if qa_input:
            prompt_template = open("./prompt/extraction_qa_template.txt", "r").read()
        else:
            prompt_template = open("./prompt/extraction_non_qa_template.txt", "r").read()
        return prompt_template

    def fact_extractor(self, snippet, sentence, qa_input=False):
        """
        snippet = (context1) <SOS>sentence<EOS> (context2)
        sentence = the sentence to be focused on
        """
        prompt_template = self.get_prompt_template(qa_input)  # qa_prompt_temp if qa_input else non_qa_prompt_temp
        prompt_text = prompt_template.format(snippet=snippet, sentence=sentence)
        response, prompt_tok_cnt, response_tok_cnt = self.get_model_response.get_response(self.system_message,
                                                                                          prompt_text)

        if "No verifiable claim." in response:
            return None, prompt_tok_cnt, response_tok_cnt
        else:
            # remove itemized list
            facts = [x.strip().replace("- ", "") for x in response.split("\n")]
            # remove numbers in the beginning
            facts = [regex.sub(r"^\d+\.?\s", "", x) for x in facts]
            return facts, prompt_tok_cnt, response_tok_cnt
