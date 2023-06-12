## Features: Bert Word Embeddings
## ## Models: KD-Tree Search

import sys, os, io, time

from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition, ensemble, tree

from nltk.corpus import stopwords
from elasticsearch import Elasticsearch

import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import spacy
import gensim
import gensim.downloader as api
from gensim.models.fasttext import FastText

import argparse

parser = argparse.ArgumentParser(description='KD-Tree search with Bert Word Embeddings')
parser.add_argument('--proc', type=int, default=0, help="0,1")

args = parser.parse_args()

my_loc = os.path.dirname(__file__)


files = ['_raw_text', '_proc_text']
fname = files[args.proc]


emb_list = ['bert-base-nli-mean-tokens']

for emb_type in emb_list:
    since = time.time()
    data = json.load(open(my_loc+'/bert_embs/'+emb_type+fname+'.json', 'r')) 
    claim_data = data['claims']
    val_data = data['test']
    val_ids = val_data['id']


    print(fname+'-------'+emb_type+'---------------------------------------------------------------------\n')
    ft_claim = np.array(claim_data['embs2'])

    ft_val = np.array(val_data['embs'])

    kdtree = KDTree(ft_claim)

    with open('my_code/file_results/bertsent_res_%s_%s.tsv'%(fname, emb_type), 'w') as f:
        dists, inds = kdtree.query(ft_val, k=1000)

        for i in range(ft_val.shape[0]):
            cos_sc = cosine_similarity(np.expand_dims(ft_val[i,:],0), ft_claim[inds[i,:]]).flatten()

            for j in range(inds.shape[1]):
                f.write("%d\tQ0\t%d\t1\t%f\t%s\n"%(int(val_data['id'][i]),inds[i,j],cos_sc[j],'bert_word'))
    
    os.system('python evaluate.py --scores my_code/file_results/bertsent_res_%s_%s.tsv --gold-labels data/test/tweet-vclaim-pairs.qrels'%(fname, emb_type))
    print('-------------------------------------------------------------------------------------------\n')