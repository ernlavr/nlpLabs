import torch
torch.cuda.empty_cache()

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModel, AdamW, T5Tokenizer, AutoModelForCausalLM

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

model_name = 'rinna/japanese-gpt2-medium'

tokenizer = T5Tokenizer.from_pretrained( model_name)

SPECIAL_TOKENS_DICT = {
    'pad_token' : '<pad>',
    'additional_special_tokens' : ['<question>', '<context>']
}

tokenizer.add_special_tokens( SPECIAL_TOKENS_DICT ) 
    
things_to_ask = [ 
    f"<question>最高の宗教は??<context>",
    f"<question>最高の宗教はありますか?<context>",
    f"<question>結婚は何歳ですか?<context>",
    f"<question>Elon Musk は何歳ですか?<context>",
    f"<question>シロクマ?<context>",
    f"<question>ホッキョクグマの生息地は?<context>",
    f"<question>ホッキョクグマはどこに住んでいますか?<context>",
    f"<question>クマのトマトはどこに住んでいますか?<context>",
    f"<question>トマトくま?<context>",
    f"<question>トマロカはどうやって生えるの?<context>",
    f"<question>日本で有名なのは?<context>",
    f"<question>タイタニック号が沈んだのはいつですか?<context>",
    f"<question>タイタニック号が沈んだのはいつですか?<context>",
    f"<question>マーケティングとは?<context>",
    f"<question>スパイダーマンはどこに住んでいますか?<context>"
]

model_name = 'rinna/japanese-gpt2-medium'

model = AutoModelForCausalLM.from_pretrained( model_name, output_hidden_states=True ).to(device)
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

