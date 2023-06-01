import torch
torch.cuda.empty_cache()

from datasets import load_dataset, Dataset

from transformers import AdamW, GPT2Tokenizer, T5Tokenizer, AutoModelForCausalLM

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split

dataset = load_dataset("tydiqa", 'primary_task')
dataset = dataset.filter( lambda df: df["language"].startswith("japanese") ) #or df["language"].startswith("finnish") or df["language"].startswith("japanese") )

train_set = dataset["train"]
validation_set = dataset["validation"]

# preprocessing our data
train_set = train_set.to_pandas() #.append(train_set2.to_pandas(), ignore_index=True) 
train_set = train_set.drop(['document_title', 'document_url'], axis=1)
    
validation_set = validation_set.to_pandas()
validation_set = validation_set.drop(['document_title', 'document_url'], axis=1)

# sort to later form batch with less possible padding
train_set = train_set.sort_values( by='document_plaintext', key=lambda col: col.str.len() )
validation_set = validation_set.sort_values( by='document_plaintext', key=lambda col: col.str.len() )

batch_size = 8

class MyDataset(Dataset):
    def __init__(self, data ):        
        self.df = data
    
    def __len__(self):
        return len( self.df )
    
    def __getitem__( self, idx ):
        item = self.df.iloc[idx]
        return item['question_text'], item['document_plaintext'].replace('\n', ' ' ), item['annotations']

train_set = MyDataset(train_set)
validation_set = MyDataset(validation_set)
 
model_name = 'rinna/japanese-gpt2-medium'

tokenizer = T5Tokenizer.from_pretrained( model_name)

SPECIAL_TOKENS_DICT = {
    'pad_token' : '<pad>',
    'additional_special_tokens' : ['<question>', '<context>']
}

tokenizer.add_special_tokens( SPECIAL_TOKENS_DICT ) 

def my_tokenizer( tokenizer, input, max_len ):

    return tokenizer.encode_plus(
                    input,        # Sentence to encode.
                    add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                    max_length = max_len,      # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
            )

def collate_fn( input ):

    input_ids = []
    token_type_ids = []
    attention_masks = []
    labels_all = []

    question_tkn = tokenizer.additional_special_tokens_ids[0]
    context_tkn = tokenizer.additional_special_tokens_ids[1]

    pad_tkn = tokenizer.pad_token_id
    eos_tkn = tokenizer.eos_token_id

    max_len = 0

    for inp in input:
        question = inp[0]
        answer = inp[1]

        qtoks = tokenizer.tokenize( question )
        atoks = tokenizer.tokenize( answer )

        if len(qtoks) + len(atoks) > max_len:
            max_len = len(qtoks) + len(atoks)

    dic_inp = {}
    dic_tok = {}
    dic_lab = {}
    dic_att = {}

    def insert( here, key, value ):
        if key in here:
            here[key].append( value )
        else:
            here[key] = [value]

    for inp in input:
        question = inp[0]
        answer = inp[1]
        context = inp[2]

        ans_start = -1
        ans_ends  = -1

        qc_pair = f'<question>{question}<context>{answer}'

        encoded_dict = my_tokenizer( tokenizer, qc_pair, max_len )
        
        ctx_pos = np.where( encoded_dict['input_ids'].detach().clone().cpu().numpy()[0] == context_tkn )[0][0]

        num_of_toks = torch.sum( encoded_dict['attention_mask'] ).item()
        
        act_num_of_toks = len( encoded_dict['input_ids'][0] )
        
        token_ids = [question_tkn] * ctx_pos + [context_tkn] * (num_of_toks-ctx_pos) + [pad_tkn] * (act_num_of_toks-num_of_toks) 

        ctx = encoded_dict['input_ids'][0][ctx_pos+1:num_of_toks].detach().clone().cpu().numpy()

        label = [-100] * (ctx_pos+1) + ctx.tolist() + [-100] * (act_num_of_toks-num_of_toks) 

        for i in range( 0, num_of_toks, 1024 ):

            diff = 1024 if num_of_toks - i > 1024 else num_of_toks - i

            inpt = torch.unsqueeze( torch.tensor(label[i:i+diff]), dim=0 )

            if len(inpt[0]) == 1:
                continue

            insert( dic_lab, diff, inpt )
            insert( dic_inp, diff, torch.unsqueeze( encoded_dict['input_ids'][0][i:i+diff], dim=0 ) )
            insert( dic_tok, diff, torch.unsqueeze( torch.tensor(token_ids[i:i+diff]), dim=0 ) )
            insert( dic_att, diff, torch.unsqueeze( encoded_dict['attention_mask'][0][i:i+diff], dim=0 ) )

    ndic_lab = {}
    ndic_tok = {}
    ndic_att = {}
    ndic_inp = {}

    keys = dic_lab.keys()    
    for k in keys:
        ndic_lab[k] = torch.cat(dic_lab[k], dim=0 )
        ndic_tok[k] = torch.cat(dic_tok[k], dim=0 )
        ndic_att[k] = torch.cat(dic_att[k], dim=0 )
        ndic_inp[k] = torch.cat(dic_inp[k], dim=0 )

    td = ( ndic_inp, ndic_tok, ndic_att, ndic_lab, keys ) 
    
    return td

train_dataloader = DataLoader( train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False )
val_dataloader = DataLoader( validation_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False )

import numpy as np
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_squared_error

from sklearn.metrics import f1_score

def evaluate( val_data, model ):
    model.eval()

    nlls = []
    end_loc = 0
    
    for step, batch in enumerate(val_data):

        keys = batch[-1]
        for k in keys:
        
            numbatches = batch[0][k].size()[0]

            for b in range(0, numbatches, batch_size ):
            
                b_input_ids = batch[0][k][b:b+batch_size].to(device)
                b_tt_ids = batch[1][k][b:b+batch_size].to(device)
                b_am = batch[2][k][b:b+batch_size].to(device)
                b_lbl = batch[3][k][b:b+batch_size].to(device)

                seq_len = b_input_ids.size()[1]
                end_loc += seq_len

                with torch.no_grad():
                    outputs = model( b_input_ids, token_type_ids=b_tt_ids, attention_mask=b_am, labels=b_lbl )

                    loss, logits = outputs[:2]

                    neg_log_likelihood = loss * seq_len
                    
                    nlls.append( torch.tensor(neg_log_likelihood.item()) ) 

    nlls = torch.tensor(nlls)

    print( f'end_loc={end_loc}' )
    ppl = torch.exp( torch.sum(nlls) / end_loc ) 

    return ppl


device = torch.device("cpu")
if torch.cuda.is_available():    
    device = torch.device("cuda")

from transformers import TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained( model_name, output_hidden_states=True ).to(device)
model.resize_token_embeddings( len(tokenizer) )

optimizer = torch.optim.AdamW( model.parameters(), lr=1e-5, eps=1e-8)

epochs = 100
print( "Started training...")
best_acc = 0.0

things_to_ask = [ f"<question>ロバとは<context>",
    f"<question>ゴールドタウンが破壊されたとき？<context>",
    f"<question>マリーは何歳ですか？<context>",
    f"<question>キングってなんであんなにバカなの？<context>", ]

torch.autograd.set_detect_anomaly(True)

for epoch in range(epochs):
    
    model.train()
    losses = []
    
    for step, batch in enumerate(train_dataloader):

        keys = batch[-1]
    
        for k in keys:
        
            numbatches = batch[0][k].size()[0]

            for b in range(0, numbatches, batch_size ):

                b_input_ids = batch[0][k][b:b+batch_size].to(device)
                b_tt_ids = batch[1][k][b:b+batch_size].to(device)
                b_am = batch[2][k][b:b+batch_size].to(device)
                b_lbl = batch[3][k][b:b+batch_size].to(device)

                optimizer.zero_grad()
                outputs = model( b_input_ids, token_type_ids=b_tt_ids, attention_mask=b_am, labels=b_lbl )
                
                loss, logits = outputs[:2]

                loss.backward()

                optimizer.step()

                losses.append( loss.item() ) #torch.mean( torch.tensor( [loss1.item(), loss2.item(), loss3.item()] ) ) )

    ppl = evaluate( val_dataloader, model )
    
    print(f'Validation accuracy at {epoch}: ppl={ppl} train loss: {sum(losses) / len(losses)}')

    print( "=" * 20 )

    model.eval()

    for i in things_to_ask:
        input_ids = tokenizer.encode( i, add_special_tokens=True, return_tensors='pt')
        input_ids = input_ids.to(device)

        # generate text until the output length (which includes the context length) reaches 50
        greedy_output = model.generate(input_ids, top_p=0.95, do_sample=True, max_length=50, pad_token_id=tokenizer.eos_token_id )

        print( tokenizer.decode(greedy_output[0], skip_special_tokens=True))

    # Keep track of the best model based on the accuracy
    model.save_pretrained("best_jap_model_GPT2_window_data_all_data")

print( "Finished training.")

