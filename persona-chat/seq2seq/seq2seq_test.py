#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

import math
import pickle
import os

from IPython.core.debugger import set_trace
from torch.nn.utils.rnn import pad_sequence
from IPython.core.debugger import set_trace

import numpy as np
import logging

from log import init_log
from dataset import DictDataset
from coll import *
from utils import test_multi_generate, test_multi_ppl, generate_samples
from model import VanillaSeq2Seq

from transformers import GPT2Tokenizer
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import matplotlib.pyplot as plt  
import tqdm
import json

from macro import *

# In[6]:

# 参数传递问题
def test_ppl(args):
    
    model = args['model']
    dataloader = args['test_dataloader']
    device = args['device'] if args['device'] is not None else torch.device('cpu')
    n_print = args['n_print']
    writer = args['writer']
    base_name = args['base_name']
    tag = args['test_tag']


    test_step = 0
    n_all = len(dataloader)
    
    n_loss = []
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for que, que_lens, src, src_lens, tgt in tqdm.tqdm(dataloader):

            que = que.to(device)
            src = src.to(device) 
            tgt = tgt.to(device)

            y = model(que, que_lens, src, src_lens)# 输出变化
            lm = y[0]
            lm = lm.transpose(-1, -2)
            l1 = F.cross_entropy(lm, tgt, ignore_index=-1)

            n_loss.append(l1.item())

            test_step += 1

            if test_step % n_print == 0:
                ppl = math.exp(l1.item())
                logging.info(f'{test_step}/{n_all} step: cur loss: {l1.item()}, cur_ppl: {ppl}')

    avg_loss = sum(n_loss) / len(n_loss)
    avg_ppl = math.exp(avg_loss)
    logging.info(f'n_epoch {model.n_epoch} -- avg_ppl: {avg_ppl}')   

    if writer != None:
        writer.add_hparams({'model' : base_name + '_' + str(model.n_epoch), 'tag':tag}, {'ppl' : avg_ppl})
     
    return math.exp(avg_loss)

# In[8]:

def greedy_generate(model, device, context, query):
    """
    有两个部分，Persona的生成
    """

    model.to(device)
    model.eval()

    c1 = []
    c2 = []

    if context != []:
        for c in context[0]:
            c1 += tok.encode(c) + [tok.eos_token_id]
        
        for c in context[1]:
            c2 += tok.encode(c) + [tok.eos_token_id]
            
    que = tok.encode(query)
    src = c1 + c2 + que

    src = torch.tensor([src]).to(device)
    src_lens = [src.size(-1)]
       
    n_max = MAX_GEN_LEN
    n_gen = 0

    with torch.no_grad():
        eos_id = tok.eos_token_id
        tgt = torch.LongTensor([[tok.eos_token_id]]).to(device)
        pre_len = tgt.size(-1)
        tgt_lens = [tgt.size(-1)]
        for i in range(n_max):        

            y = model(src, src_lens, tgt, tgt_lens) # 子回归生成    
            lm = y[0]      
            o = torch.argmax(lm[0, -1, :])
            
            if o.item() == eos_id:
                break
            else:
                n_gen += 1
                tgt = torch.cat([tgt, torch.tensor([[o]], device=device)], dim=-1)
                tgt_lens[0] += 1
            
    gen = tgt[0][pre_len:pre_len + n_gen].tolist()
    
    return tok.decode(gen)

def main():
    with open(TOK_PATH, 'rb') as f:
        tok = pickle.load(f)

    test_dataset = DictDataset.load_dataset(TEST_PC_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    base_name = 'seq2seq'

    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    # PersonaChat-Dataset
    ppl_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=coll_pc_201,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    generate_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=coll_pc_211,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    summary_dir = os.path.join(EVAL_DIR, base_name, 'eval')
    writer = SummaryWriter(summary_dir)

    model_suffix = list(range(6, 10+1))

    device = gpu_device
    
    # model_paths = [os.path.join(MODEL_DIR, base_name + '.pt_' + str(x)) for x in model_suffix]

    # 完整测试 
    # Seq2Seq = 8
    model_paths = [os.path.join(FINAL_MODEL_DIR, base_name + '.pt_8')]
    
    # n_ppl = test_multi_ppl(model_paths, device, ppl_dataloader, test_ppl, base_name=base_name, writer=writer)
    test_multi_generate(model_paths, device, generate_dataloader, generate_samples, greedy_generate, full_test=True)

if __name__ == '__main__':
    main()