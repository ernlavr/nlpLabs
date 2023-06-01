import torch
torch.cuda.empty_cache()

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split


import numpy as np
from torch import nn
from transformers import PreTrainedModel, DistilBertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, accuracy_score

import torch
torch.cuda.empty_cache()

from datasets import load_dataset, Dataset

from transformers import AdamW, GPT2Tokenizer, GPT2LMHeadModel

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split

dataset = load_dataset("copenlu/answerable_tydiqa")
dataset = dataset.filter( lambda df: df["language"].startswith("finnish") ) #or df["language"].startswith("finnish") or df["language"].startswith("japanese") )

train_set = dataset["train"]
validation_set = dataset["validation"]

# preprocessing our data
train_set = train_set.to_pandas()
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
 
model_name = 'best_fin_model_GPT2'

tokenizer = GPT2Tokenizer.from_pretrained( 'Finnish-NLP/gpt2-finnish', do_lower_case=True)

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
                    truncation = True,
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

    if max_len > 1024:
        max_len = 1024

    for inp in input:
        question = inp[0]
        answer = inp[1]
        context = inp[2]
        answerable = 0 if context['answer_start'][0] == -1 else 1

        input = [ f'<question>{question}<context>{answer}' ]

        for inp in input:
            encoded_dict = my_tokenizer( tokenizer, inp, max_len )
            
            ctx_pos = np.where( encoded_dict['input_ids'].detach().clone().cpu().numpy()[0] == context_tkn )[0][0]

            num_of_toks = torch.sum( encoded_dict['attention_mask'] ).item()

            
            act_num_of_toks = len( encoded_dict['input_ids'][0] )
            
            token_ids = [question_tkn] * ctx_pos + [context_tkn] * (num_of_toks-ctx_pos) + [pad_tkn] * (act_num_of_toks-num_of_toks) 
            
            token_type_ids.append( torch.unsqueeze( torch.tensor(token_ids), dim=0 ) )
            ctx = encoded_dict['input_ids'][0][ctx_pos+1:num_of_toks].detach().clone().cpu().numpy()
        
            labels_all.append( torch.unsqueeze( torch.tensor(answerable), dim=0 ) )

            input_ids.append( encoded_dict['input_ids'] )
            attention_masks.append( encoded_dict['attention_mask'] )

    input_ids = torch.cat(input_ids, dim=0 )
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    labels_all = torch.cat( labels_all, dim=0 )

    td = TensorDataset( input_ids, token_type_ids, attention_masks, labels_all ) 
    
    return td

train_dataloader = DataLoader( train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False )
val_dataloader = DataLoader( validation_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False )

device = torch.device("cpu")
if torch.cuda.is_available():    
    device = torch.device("cuda")

# GPT2Model
model = GPT2LMHeadModel.from_pretrained( model_name, output_hidden_states=True ).to(device)
model.resize_token_embeddings( len(tokenizer) )


model.eval()

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='saga', max_iter=1000)

Xtrain = []
ytrain = []

for step, batch in enumerate(train_dataloader):

    b_input_ids = batch[:][0].to(device)
    b_tt_ids = batch[:][1].to(device)
    b_am = batch[:][2].to(device)
    b_lbl = batch[:][3].to(device)

    output = model( b_input_ids, attention_mask=b_am, token_type_ids=b_tt_ids )
    last_hid_lay = torch.stack( list(output['hidden_states']), dim=0 )[-1][:,:,:]
    
    Xtrain.extend( torch.sum(last_hid_lay, dim=1 ).detach().clone().cpu().numpy().tolist() )
    ytrain.extend( b_lbl.detach().clone().cpu().numpy().tolist() ) 


Xtrain = np.squeeze( np.asarray( Xtrain ) )
ytrain = np.squeeze( np.asarray( ytrain ) )
print( Xtrain.shape, ytrain.shape )

Xval = []
yval = []

torch.cuda.empty_cache()

for step, batch in enumerate(val_dataloader):

    b_input_ids = batch[:][0].to(device)
    b_tt_ids = batch[:][1].to(device)
    b_am = batch[:][2].to(device)
    b_lbl = batch[:][3].to(device)

    output = model( b_input_ids, attention_mask=b_am, token_type_ids=b_tt_ids )
    last_hid_lay = torch.stack( list(output['hidden_states']), dim=0 )[-1][:,:,:]
    
    Xval.extend( torch.sum(last_hid_lay, dim=1 ).detach().clone().cpu().numpy().tolist() )
    yval.extend( b_lbl.detach().clone().cpu().numpy().tolist() ) 

torch.cuda.empty_cache()

Xval = np.squeeze( np.asarray( Xval ) )
yval = np.squeeze( np.asarray( yval ) )

print( Xval.shape, yval.shape )


clf.fit( Xtrain, ytrain ) 
pred = clf.predict( Xval )

print( f"FIN: f1_score={ f1_score( yval, pred )} accuracy={ accuracy_score( yval, pred )}" )


##########################
# SAVE-LOAD using pickle #
##########################
import pickle

# save
with open(f'{model_name}-answrbl.pkl','wb') as f:
    pickle.dump(clf,f)

# load
#with open('model.pkl', 'rb') as f:
#    clf2 = pickle.load(f)


print( "Finished training.")

