import os
import regex
import pdb
import json
import spacy
from tqdm import tqdm
from .get_response import GetResponse


class ClaimExtractor():
    def __init__(self, model_name, cache_dir="./data/cache/", use_external_model=False):
        self.model = None
        if os.path.isdir(model_name) or use_external_model:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            self.model = self.model.to("cuda")
            self.alpaca_prompt = open("./prompt/extraction_alpaca_template.txt", "r").read()
        else:
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f"claim_extraction_cache.json")
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=1000,
                                                  temperature=0)
            self.system_message = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Wikipedia)"

        self.spacy_nlp = spacy.load('en_core_web_sm')

    def non_qa_scanner_extractor(self, response, cost_estimate_only=False):
        """
        Given a model output
        - split the response into sentences using spaCy
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """
        sentences = self.get_sentence(response)

        all_facts_lst = []
        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0

        # new return values
        snippet_lst = []
        fact_lst_lst = []

        for i, sentence in enumerate(sentences):
            if self.model:
                input = response.strip()
                snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            else:
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

            snippet_lst.append(snippet)

            # call fact_extractor on each snippet
            facts, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(),
                                                                          qa_input=False,
                                                                          cost_estimate_only=cost_estimate_only)

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

    def qa_scanner_extractor(self, question, response, cost_estimate_only=False):
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
        for i, sentence in enumerate(sentences):
            if self.model:
                input = f"Questions:\n{question.strip()}\nResponse:\n{response.strip()}"
                snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            else:
                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])

                snippet = f"Question: {question.strip()}\nResponse: {context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            # new return value
            snippet_lst.append(snippet)

            # call fact_extractor on each tesnippetxt
            facts, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(), qa_input=True,
                                                                          cost_estimate_only=cost_estimate_only)

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

    def fact_extractor(self, snippet, sentence, qa_input=False, cost_estimate_only=False):
        """
        snippet = (context1) <SOS>sentence<EOS> (context2)
        sentence = the sentence to be focused on
        """

        if self.model:
            formatted_input = self.alpaca_prompt.format(snippet, "")
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=1000, use_cache=True)
            output_str = ' '.join(self.tokenizer.batch_decode(outputs))
            # print(output_str)

            clean_output = output_str.split("### Response:")[-1].strip().replace("</s>", "")
            if not clean_output or "No verifiable claim." in clean_output:
                return None, 0, 0
            claims = [x.strip() for x in clean_output.split("\n")]
            return claims, 0, 0
        else:
            ### prompting base approach via API call
            prompt_template = self.get_prompt_template(qa_input)  # qa_prompt_temp if qa_input else non_qa_prompt_temp
            prompt_text = prompt_template.format(snippet=snippet, sentence=sentence)
            response, prompt_tok_cnt, response_tok_cnt = self.get_model_response.get_response(self.system_message,
                                                                                              prompt_text,
                                                                                              cost_estimate_only)
            if not response or "No verifiable claim." in response:
                return None, prompt_tok_cnt, response_tok_cnt
            else:
                # remove itemized list
                claims = [x.strip().replace("- ", "") for x in response.split("\n")]
                # remove numbers in the beginning
                claims = [regex.sub(r"^\d+\.?\s", "", x) for x in claims]
                return claims, prompt_tok_cnt, response_tok_cnt
