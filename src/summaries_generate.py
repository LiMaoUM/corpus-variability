import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
import preprocessor as p
import pandas as pd

import re


def remove_parentheses_and_tags(text):
    # First, remove the content within parentheses
    no_parentheses = re.sub(r"\([^()]*\)", "", text)
    # Then, remove the '</s>' tags from the string
    no_tags = re.sub(r"</s>", "", no_parentheses)
    no_newlines = re.sub(r"\n", "", no_tags)
    no_inst = re.sub(r"[/INST] ", "", no_newlines)
    return re.sub(r"```", "", no_inst)


randoms = pd.read_csv(
    "/nfs/turbo/isr-fconrad1/projects/corpus-variability/data/randoms.csv"
)
citizens = pd.read_csv(
    "/nfs/turbo/isr-fconrad1/projects/corpus-variability/data/citizens.csv"
)

# Load the model
peft_model_id = "/nfs/turbo/isr-fconrad1/model/LLaMa2TS-0.1.0"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_8bit=True,
    device_map="auto",
)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(
    "/nfs/turbo/isr-fconrad1/model/LLaMa2TS-0.1.0"
)
tokenizer.pad_token_id = tokenizer.eos_token_id

idx = 0
sequences_list = []
randoms["Message"] = randoms["Message"].apply(p.clean)
for i in tqdm(range(100)):
    edx = idx + 50
    documents = list(randoms.loc[idx:edx, "Message"])
    tweets = ""
    i = 1
    for text in documents:
        tweets += str(i) + "-[" + text + "] "
        i += 1
    messages = f"""<s>[INST] <</SYS>>\nI would like you to help me by summarizing a group of tweets, delimited by triple backticks, and each tweet is labeled by a number in a given format: number-[tweet]. Give me a comprehensive summary in a concise paragraph and as you generate each sentence, provide the identifying number of tweets on which that sentence is based:\n<</SYS>>\n\n
                      ``` {tweets} ``` \n [/INST] \n"""

    inputs = tokenizer(messages, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids.to(model.device)
    sequences = model.generate(
        **inputs,
        max_new_tokens=500,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    sequences_list.append(tokenizer.decode(sequences[0][len(input_ids[0]) :]))
    idx += 50

# Save the summaries

clean_summaries = [remove_parentheses_and_tags(sent.strip()) for sent in sequences_list]
random_summaries = pd.DataFrame(clean_summaries, columns=["summary"])
random_summaries.to_csv(
    "/nfs/turbo/isr-fconrad1/projects/corpus-variability/data/randoms_summaries.csv",
    index=False,
)

# Now, let's do the same for the citizens
idx = 0
sequences_list = []
citizens["Message"] = citizens["Message"].apply(p.clean)
for i in tqdm(range(100)):
    edx = idx + 50
    documents = list(citizens.loc[idx:edx, "Message"])
    tweets = ""
    i = 1
    for text in documents:
        tweets += str(i) + "-[" + text + "] "
        i += 1
    messages = f"""<s>[INST] <</SYS>>\nI would like you to help me by summarizing a group of tweets, delimited by triple backticks, and each tweet is labeled by a number in a given format: number-[tweet]. Give me a comprehensive summary in a concise paragraph and as you generate each sentence, provide the identifying number of tweets on which that sentence is based:\n<</SYS>>\n\n
                      ``` {tweets} ``` \n [/INST] \n"""
    inputs = tokenizer(messages, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids.to(model.device)
    sequences = model.generate(
        **inputs,
        max_new_tokens=500,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    sequences_list.append(tokenizer.decode(sequences[0][len(input_ids[0]) :]))
    idx += 50


# Save the summaries

clean_summaries = [remove_parentheses_and_tags(sent.strip()) for sent in sequences_list]

citizen_summaries = pd.DataFrame(clean_summaries, columns=["summary"])
citizen_summaries.to_csv(
    "/nfs/turbo/isr-fconrad1/projects/corpus-variability/data/citizens_summaries.csv",
    index=False,
)
