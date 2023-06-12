
## Features: Bert Word Embeddings
## ## Models: KD-Tree Search

import sys, os, io, time

from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition, ensemble, tree

from nltk.corpus import stopwords

import json, random, math, re
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader

import argparse

start_time_reading = time.time()
parser = argparse.ArgumentParser(description='Finetune with Bert Sent Embeddings')
parser.add_argument('--proc', type=int, default=0, help="0,1")
parser.add_argument('--bert_type', type=int, default=0, help="Bert Type, Base, Large, Roberta, etc...")

args = parser.parse_args()


my_loc = os.path.dirname(__file__)


files = ['_raw_text', '_proc_text']
fname = files[args.proc]
#Primary run: bert-base-nli-cls-tokens'
# 2nd contrastive submission: bert-large-nli-max-tokens'
# 1st contrastive submission: distilbert multilingual
emb_list = ['bert-base-nli-cls-token']


emb_type = emb_list[args.bert_type]


model_path = 'models/finetune_%s_%s'%(fname,emb_type)
model = SentenceTransformer(model_path).eval()


val_data = json.load(open(my_loc+'my_code/proc_data/test.json', 'r'))
claim_data = json.load(open(my_loc+'my_code/proc_data/claim_dict.json', 'r'))

data_type = {'test': val_data, 'claims': claim_data}

reading_time = time.time() - start_time_reading
print("Reading Files Time")
print(reading_time)

if 'raw' in fname:
    embed_dict = { 'val': {'id': [], 'embs': []},
                        'test': {'id': [], 'embs': []},
                        'claims': {'id': [], 'embs': [], 'embs2': [], 'embs3':[]}}
    for phase in data_type:
        corpus = []
        corpus2 = []
        corpus3 = []
        data = data_type[phase]

        if phase == 'claims':
            for idx in data:
                claim = data[idx]['claim']
                title = data[idx]['title']

                corpus.append(claim.strip())
                corpus2.append(title.strip())
                corpus3.append(title.strip()+" "+claim.strip())

                embed_dict[phase]['id'].append(idx)

        else:
            for idx in data:
                text = data[idx]['text']
                corpus.append(text.strip())

                embed_dict[phase]['id'].append(idx)


        if phase == 'claims':
            embeddings1 = model.encode(corpus, batch_size=64)
            embeddings2 = model.encode(corpus2, batch_size=64)
            embeddings3 = model.encode(corpus3, batch_size=64)

            for emb1, emb2 in zip(embeddings1, embeddings2):
                embed_dict[phase]['embs'].append(emb1.tolist())
                embed_dict[phase]['embs2'].append(((emb1+emb2)/2).tolist())
            for emb in embeddings3:
                embed_dict[phase]['embs3'].append(emb.tolist())
        else:
            embeddings = model.encode(corpus, batch_size=64)
            for emb in embeddings:
                embed_dict[phase]['embs'].append(emb.tolist())

    json.dump(embed_dict, open(my_loc+'bert_embs/test_%s_%s.json'%(fname,emb_type), 'w', encoding='utf-8'))

else:
    embed_dict = { 'val': {'id': [], 'embs': []},
                        'test': {'id': [], 'embs': []},
                        'claims': {'id': [], 'embs': [], 'embs2': [], 'embs3':[]}}
    for phase in data_type:
        corpus = []
        corpus2 = []
        corpus3 = []
        data = data_type[phase]

        if phase == 'claims':
            start_time_claim_dict = time.time()
            for idx in data:
                claim = data[idx]['claim_proc']
                title = data[idx]['title_proc']

                claim = [word for word in claim if not re.search(r'<(/?)[a-z]+>', word)]
                title = [word for word in title if not re.search(r'<(/?)[a-z]+>', word)]

                claim_text = ""
                for word in claim:
                    claim_text += word if word in [',', '.'] else " "+word

                title_text = ""
                for word in title:
                    title_text += word if word in [',', '.'] else " "+word

                corpus.append(claim_text.strip())
                corpus2.append(title_text.strip())
                corpus3.append(title_text.strip()+" "+claim_text.strip())

                embed_dict[phase]['id'].append(idx)
            claim_dict_time = time.time() - start_time_claim_dict
            print("Claim Dict Time")
            print(claim_dict_time)

        else:
            start_time_0 = time.time()
            for idx in data:
                proc_text = data[idx]['wiki_proc']
                proc_text = [word for word in proc_text if not re.search(r'<(/?)[a-z]+>', word)]

                text = ""
                for word in proc_text:
                    text += word if word in [',', '.'] else " "+word

                corpus.append(text.strip())

                embed_dict[phase]['id'].append(idx)
            text_dict_time = time.time() - start_time_0
            print("Text Dict Time")
            print(text_dict_time)


        if phase == 'claims':
            start_time_claim = time.time()
            embeddings1 = model.encode(corpus, batch_size=64)
            embeddings2 = model.encode(corpus2, batch_size=64)
            embeddings3 = model.encode(corpus3, batch_size=64)

            for emb1, emb2 in zip(embeddings1, embeddings2):
                embed_dict[phase]['embs'].append(emb1.tolist())
                embed_dict[phase]['embs2'].append(((emb1+emb2)/2).tolist())
            for emb in embeddings3:
                embed_dict[phase]['embs3'].append(emb.tolist())
            claim_emb_encoding_dict_time = time.time() - start_time_claim
            print("Claim Embedding Encoding Time")
            print(claim_emb_encoding_dict_time)
        else:
            start_time_1 = time.time()
            embeddings = model.encode(corpus, batch_size=64)
            for emb in embeddings:
                embed_dict[phase]['embs'].append(emb.tolist())
            txt_emb_encoding_dict_time = time.time() - start_time_1
            print("Text Embedding Encoding Time")
            print(txt_emb_encoding_dict_time)
    start_time_2 = time.time()
    json.dump(embed_dict, open(my_loc+'bert_embs/test_%s_%s.json'%(fname,emb_type), 'w', encoding='utf-8'))



data = json.load(open(my_loc+'bert_embs/test_%s_%s.json'%(fname,emb_type), 'r', encoding='utf-8'))
claim_data = data['claims']
val_data = data['test']
val_ids = val_data['id']
calim_ids = claim_data['id']

for clm_type in ['embs', 'embs2', 'embs3']:
    print('----------------------------------------------------------------------------------------\n')
    print(clm_type)
    ft_claim = np.array(claim_data[clm_type])

    ft_val = np.array(val_data['embs'])

    kdtree = KDTree(ft_claim)

    with open('my_code/file_results/bertsent_finetune_res_%s_%s.tsv'%(fname,emb_type), 'w') as f:
        dists, inds = kdtree.query(ft_val, k=1000)

        for i in range(ft_val.shape[0]):
            cos_sc = cosine_similarity(np.expand_dims(ft_val[i, :], 0), ft_claim[inds[i, :]]).flatten()

            for j in range(inds.shape[1]):
                f.write("%d\tQ0\t%d\t1\t%f\t%s\n" % (int(val_data['id'][i]), inds[i, j], cos_sc[j], 'bert_word'))

        #for i in range(ft_val.shape[0]):
        #    i_dist = dists[i]
        #    i_dist = 1 - i_dist/max(i_dist)

            #for j in range(inds.shape[1]):
            #    f.write("%s\tQ0\t%s\t1\t%f\t%s\n"%(str(val_data['id'][i]),calim_ids[inds[i,j]],i_dist[j],'bert_word'))

    #os.system('python evaluate.py --scores my_code/file_results/bertsent_finetune_res_%s_%s.tsv --gold-labels data/test/tweet-vclaim-pairs.qrels'%(fname,emb_type))
    #print('-------------------------------------------------------------------------------------------\n')
pred_time = time.time() - start_time_2
print("--- %s pred seconds ---" % (pred_time))

print("Final Time without claim embedding time")
print(text_dict_time + txt_emb_encoding_dict_time + pred_time + reading_time)