## Features: Word Embeddings
## ## Models: SVM

import sys, os

import json, re
import pandas as pd
import numpy as np
import time

from nltk.corpus import stopwords

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch

start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

my_loc = os.path.dirname(__file__)

df_stopwords = set(stopwords.words('english'))




train_data = json.load(open('my_code/proc_data/train.json', 'r'))
val_data = json.load(open('my_code/proc_data/val.json', 'r'))
test_data = json.load(open('my_code/proc_data/test.json', 'r'))
claim_data = json.load(open('my_code/proc_data/claim_dict.json', 'r'))


data_type = {'train': train_data, 'val': val_data, 'test': test_data, 'claims': claim_data}

## Bert Sentence embeddings from a sentence transformer
#Primary run: bert-base-nli-cls-token'
# 2nd contrastive submission: bert-large-nli-max-tokens'
# 1st contrastive submission: distilbert multilingual

model_types = ['bert-base-nli-cls-token']

for md in model_types:
    model = SentenceTransformer(md).to(device)

    embed_dict = {'train':{'id': [], 'embs': [] }, 
                    'val': {'id': [], 'embs': []},
                    'test': {'id': [], 'embs': []},
                    'claims': {'id': [], 'embs': [], 'embs2': []}}

    for phase in data_type:
        corpus = []
        corpus2 = []
        data = data_type[phase]
        for id in data:
            if phase == 'claims':
                claim = data[id]['claim']
                title = data[id]['title']

                corpus.append(claim)
                corpus2.append(title)

            else:
                text = data[id]['text']
                corpus.append(text)
            
            embed_dict[phase]['id'].append(id)
        
        if phase == 'claims':
            embeddings1 = model.encode(corpus, batch_size=32)
            embeddings2 = model.encode(corpus2, batch_size=32)
            for emb1, emb2 in zip(embeddings1, embeddings2):
                embed_dict[phase]['embs'].append(emb1.tolist())
                embed_dict[phase]['embs2'].append(((emb1+emb2)/2).tolist())
        else:
            embeddings = model.encode(corpus, batch_size=32)
            for emb in embeddings:
                embed_dict[phase]['embs'].append(emb.tolist())
            

    json.dump(embed_dict, open('bert_embs/%s_raw_text.json'%(md), 'w'))

print("--- %s seconds ---" % (time.time() - start_time))
for md in model_types:
    model = SentenceTransformer(md).to(device)

    embed_dict = {'train':{'id': [], 'embs': [] }, 
                    'val': {'id': [], 'embs': []},
                    'test': {'id': [], 'embs': []},
                    'claims': {'id': [], 'embs': [], 'embs2': []}}

    for phase in data_type:
        corpus = []
        corpus2 = []
        data = data_type[phase]
        for id in data:
            if phase == 'claims':
                claim = data[id]['claim_proc']
                title = data[id]['title_proc']

                claim = [word for word in claim if not re.search(r'<(/?)[a-z]+>', word)]
                title = [word for word in title if not re.search(r'<(/?)[a-z]+>', word)]

                claim_text = ""
                for word in claim:
                    claim_text += word if word in [',', '.'] else " "+word

                title_text = ""
                for word in title:
                    title_text += word if word in [',', '.'] else " "+word

                corpus.append(claim_text)
                corpus2.append(title_text)

            else:
                proc_text = data[id]['wiki_proc']
                proc_text = [word for word in proc_text if not re.search(r'<(/?)[a-z]+>', word)]

                text = ""
                for word in proc_text:
                    text += word if word in [',', '.'] else " "+word

                corpus.append(text)
            
            embed_dict[phase]['id'].append(id)
            
        if phase == 'claims':
            embeddings1 = model.encode(corpus, batch_size=32)
            embeddings2 = model.encode(corpus2, batch_size=32)
            for emb1, emb2 in zip(embeddings1, embeddings2):
                embed_dict[phase]['embs'].append(emb1.tolist())
                embed_dict[phase]['embs2'].append(((emb1+emb2)/2).tolist())
        else:
            embeddings = model.encode(corpus, batch_size=32)
            for emb in embeddings:
                embed_dict[phase]['embs'].append(emb.tolist())

    json.dump(embed_dict, open('bert_embs/%s_proc_text.json'%(md), 'w'))

    
