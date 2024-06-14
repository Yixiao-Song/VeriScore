# VeriScore
This is introduction for pip package of VeriScore. It have two type of method to extract claims by 
1) Prompting 
2) Fine-tuned model

**We have a preliminary [Colab notebook](https://colab.research.google.com/drive/14cJsd5xu-paXb1ld72kF3WA97qzcyEn1?authuser=1#scrollTo=uhfwyPWBUojR) for demo** 

VeriScore consists of three parts 1) `claim extraction` 2) `evidence searching` and 3) `claim verification`.
We provide an end-to-end pipeline to obtain the VeriScore, along with each of its components individually.

**You can choose between a prompting-based approach and a fine-tuned model-based approach using the `model_name` option. If you specify the path to the checkpoint of a fine-tuned model, it will automatically access the local model to perform inference. If the model name is not in a directory format, it will use an API call instead.**

## Install
1. Make a new Python 3.9+ environment using `virtualenv` or `conda`.
2. Install `veriscore` pacakge using `pip`
3. Download `en_core_web_sm` using `spacy` library
4. Our code supports inference using fine-tuned models based on the Unsloth library. To use this feature, you need to install the [Unsloth](https://github.com/unslothai/unsloth) library.
```
pip install --upgrade veriscore
python -m spacy download en_core_web_sm
```

## Setup environment before running code
1. Download `prompt` folder that have txt file of prompt template (you can see `prompt` folder VeriScore's repository)
2. Set OpenAI or Claude API key to environment variable of `bash` for prompting approach
```
export OPENAI_API_KEY_PERSONAL={your_openai_api_key}
export CLAUDE_API_KEY={your_claude_api_key}
```
3. Set SERPER API key to environment variable of `bash` for searching evidence
```
export SERPER_KEY_PRIVATE={your_serper_api_key}
```
4. For the prompt-based approach, you need to set `data_dir/demos/` with [few-shot examples](https://github.com/Yixiao-Song/VeriScore/blob/main/data/demos/few_shot_examples.jsonl).

## Running VeriScore using a command line
This is an end-to-end pipeline for running VeriScore.
```
 python3 -m veriscore.veriscore --data_dir {data_dir} --input_file {input_file} --model_name_extraction {model_name_extraction} --model_name_verification {model_name_verification}
```
* `data_dir`: Directory containing input data. `./data` by default.
* `input_file`: Name of input data file. It should be `jsonl` format where each line contains
    * `question`: query to ask
    * `response`: generated response from of `question`
    * `model`: name of model generate response
    * `prompt_source`: name of dataset provide `question` like FreshQA
* `model_name_extraction`: Name of model used for claim extraction. `gpt-4-0125-preview` by default.
* `model_name_verification`: Name of model used for claim verification. `gpt-4o` by default.

Other optional flags:

* `output_dir`: Directory for saving ouptut data. `./data` by default.
* `cache_dir`: Directory for saving cache data. `./data/cache` by default.
* `label_n`: This is type of label for claim verification. It could be `2` (binary) ro `3` (trinary)
    * `2`: query to ask binary `supported` and `unsupported`
    * `3`: query to ask trinay `labels—supported`, `contradicted`, and `inconclusive`
* `search_res_num`: A Hyperparameter for number of search result. `5` by default.

Saving output: 
`input_file_name` is file name removed `jsonl` from `—-input_file`
* `extracted claims` will be saved to `output_dir/claims_{input_file_name}.jsonl`.
* `searched evidence` will be saved to `output_dir/evidence_{input_file_name}.jsonl`.
* `verified claims` will be saved to `output_dir/model_output/verification_{input_file_name}.jsonl`.


## Running individual part using a command line
`Claim extraction`:
```
 python3 -m veriscore.extract_claims --data_dir {data_dir} --input_file {input_file} --model_name {model_name} 
```
* `input_file`: Name of input data file. It should be `jsonl` format where each line contains
    * `question`: query to ask
    * `response`: generated response from of `question`
    * `model`: name of model generate response
    * `prompt_source`: name of dataset provide `question` like FreshQA
* `model_name`: Name of model used for claim extraction. `gpt-4-0125-preview` by default.
output:
```dictionary
 {
  "question": question.strip(),
  "prompt_source": prompt_source,
  "response": response.strip(),
  "prompt_tok_cnt": prompt_tok_cnt,
  "response_tok_cnt": response_tok_cnt,
  "model": model,
  "abstained": False,  
  "claim_list": list of claims for each snippet,
  "all_claims": list of all claims
 }
```
`Evidence searching`:
```
 python3 -m veriscore.retrieve_evidence --data_dir {data_dir} --input_file {input_file}
```
* `input_file`: Name of input data file. It should be `jsonl` format where each line contains the keys of the output dictionary from the `Claim extraction`.
output:
```dictionary
 {
  ...
  "claim_snippets_dict": dictionary for claim and list of searched evidence. each evidence have dictionary of {"title": title, "snippet": snippet, "link": link}
 }
```

`Claim verification`:
```
 python3 -m veriscore.verify_claims --data_dir {data_dir} --input_file {input_file} --model_name {model_name}
```
* `input_file`: Name of input data file. It should be `jsonl` format where each line contains the keys of the output dictionary from the `Evidence searching`.
output:
```dictionary
 {
  ...
  "response": result of claim verification
  "clean_output": post-processed label
 }
```
