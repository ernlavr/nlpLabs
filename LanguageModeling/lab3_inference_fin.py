import torch
torch.cuda.empty_cache()

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split


import numpy as np
from torch import nn
from transformers import PreTrainedModel, DistilBertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

device = torch.device("cpu")
if torch.cuda.is_available():    
    device = torch.device("cuda")

model_name = 'Finnish-NLP/gpt2-finnish' #'distilgpt2'

tokenizer = GPT2Tokenizer.from_pretrained( model_name, do_lower_case=True, use_fast=True, pad_token='<pad>', eos_token='<endoftext>', bos_token='<startoftext>')
SPECIAL_TOKENS_DICT = {
    'pad_token' : '<pad>',
    'additional_special_tokens' : ['<question>', '<context>']
}

tokenizer.add_special_tokens( SPECIAL_TOKENS_DICT ) 

things_to_ask = [
    f"<question>Mikä on paras uskonto??<context>",
    f"<question>Onko parasta uskontoa?<context>",
    f"<question>Kuinka vanha Marry on?<context>",
    f"<question>Kuinka vanha on Elon Musk?<context>",
    f"<question>jääkarhu?<context>",
    f"<question>Mikä on jääkarhun elinympäristö?<context>",
    f"<question>Missä jääkarhu asuu?<context>",
    f"<question>Missä tomaattikarhu asuu?<context>",
    f"<question>Tomaattikarhu?<context>",
    f"<question>Kuinka tomarokarhu?<context>",
    f"<question>Sillä mitä Japani on kuuluisa?<context>",
    f"<question>Kun Titanic upposi?<context>",
    f"<question>Kun Titanik upposi?<context>",
    f"<question>Mikä on markkinointi?<context>",
    f"<question>Missä Spiderman asuu?<context>"
]

model_name = "best_fin_model_GPT2"

model = GPT2LMHeadModel.from_pretrained( model_name, output_hidden_states=True ).to(device)
model.resize_token_embeddings( len(tokenizer) )


model.eval()

torch.manual_seed(0)

for i in things_to_ask:
    input_ids = tokenizer.encode( i, add_special_tokens=True, return_tensors='pt')
    input_ids = input_ids.to(device)

    greedy_output = model.generate(input_ids, top_p=0.95, do_sample=True, max_length=100, pad_token_id=tokenizer.eos_token_id, num_return_sequences=10 )

    for i, output in enumerate(greedy_output):
        print( "{}) {}".format( i, tokenizer.decode(output, skip_special_tokens=True).strip() ) )

    print( "=" * 100 )

print( "Finished training.")

