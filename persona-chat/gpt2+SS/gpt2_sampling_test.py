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
import numpy as np
import logging
import tqdm
import json

from log import init_log
from dataset import DictDataset
from coll import *
from utils import test_multi_generate, test_multi_ppl, generate_samples, get_test_args
from model import GPT2Sampling
from transformer_util import get_padding_mask
import matplotlib.pyplot as plt  
from macro import *

# In[6]:

def test_ppl(args):

    model = args['model']
    dataloader = args['test_dataloader']
    device = args['device'] if args['device'] is not None else torch.device('cpu')
    n_print = args['n_print']
    writer = args['writer']
    base_name = args['base_name']
    test_tag = args['test_tag']

    # n_step = model.n_step
    n_step = 0

    n_all = len(dataloader)
    n_loss = []
    
    ########################
    model.train_flag = False
    ########################

    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for que, q_lens, tgt, _ in tqdm.tqdm(dataloader):
        # for que, q_lens, tgt in tqdm.tqdm(dataloader):

            que = que.to(device)
            tgt = tgt.to(device) 

            y = model(que, q_lens, None)# 不是train model不需要这个mask
            lm = y[0]
            lm = lm.transpose(-1, -2)
            l1 = F.cross_entropy(lm, tgt, ignore_index=-1)
            
            n_loss.append(l1.item())

            n_step += 1
            if n_step % n_print == 1:
                logging.info('%d/%d step: loss: %f ppl: %f, avg_loss: %f' % (n_step, n_all, l1.item(), math.exp(l1.item()), sum(n_loss) / len(n_loss))) # 增加 总体进行程度说明。

        avg_loss = sum(n_loss) / len(n_loss)
        avg_ppl = math.exp(avg_loss)
        
        logging.info('eval over: avg_loss: %f avg_ppl: %f' % (avg_loss, avg_ppl))
        if writer != None:
            writer.add_hparams({'model' : base_name + '_' + str(model.n_epoch), 'tag':test_tag}, {'ppl' : avg_ppl})
        
    return avg_ppl

# In[8]:

def greedy_generate(model, device, context, query):
    """
    Args:
        model ([type]): [description]
        device ([type]): [description]
        context ([type]): [description]
        query ([type]): [description]
    """
    model.to(device)
    model.eval()

    ####################################
    model.train_flag = False
    ####################################
    
    c1 = []
    for c in context[0]:
        c1 += tok.encode(c) + [tok.eos_token_id]
    c2 = []
    for c in context[1]:
        c2 += tok.encode(c) + [tok.eos_token_id]
            
    que = tok.encode(query)
    src = c1 + c2 + que + [tok.eos_token_id]

    src_lens = [len(src)]
    pre_len = len(src)

    src = torch.LongTensor([src]).to(device)
    
    n_max = MAX_GEN_LEN
    n_gen = 0
#     set_trace()
    with torch.no_grad():
        eos_id = tok.eos_token_id
        for i in range(n_max):        

            y = model(src, src_lens, None) # 子回归生成

            lm = y[0]
            # 还是argmax
            o = torch.argmax(lm[0, -1, :])
            # 

            if o.item() == eos_id:
                break
            else:
                n_gen += 1
                src = torch.cat([src, torch.tensor([[o]], device=device)], dim=-1)
                src_lens[0] += 1
#     set_trace()
    gen = src[0][pre_len:pre_len + n_gen].tolist()
    
    return tok.decode(gen)  


def main():
    with open(TOK_PATH, 'rb') as f:
        tok = pickle.load(f)

    test_dataset = DictDataset.load_dataset(TEST_PC_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    base_name = 'gpt2_sampling'

    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    ppl_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=coll_pc_204,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )
    generate_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=coll_pc_211,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    # model_suffix = [1]
    model_suffix = list(range(1, 7+1))
    device = gpu_device

    model_paths = [os.path.join(MODEL_DIR, base_name + '.pt_' + str(x)) for x in model_suffix]

    summary_dir = os.path.join(EVAL_DIR, base_name, 'eval')
    writer = SummaryWriter(summary_dir)

    # for gpt2_sampling == 3
    model_paths = [os.path.join(FINAL_MODEL_DIR, base_name + '.pt_3')]
    
    # n_ppl = test_multi_ppl(model_paths, device, ppl_dataloader, test_ppl, base_name=base_name, writer=writer)
    test_multi_generate(model_paths, device, generate_dataloader, generate_samples, greedy_generate, full_test=True)

    # n_ppl = test_multi_ppl(model_paths, device, ppl_dataloader, test_ppl, base_name=base_name, writer=writer) 
    # print(n_ppl)
    # test_multi_generate(model_paths, device, generate_dataloader, generate_samples, greedy_generate, generate_tag='generate')

    # model = torch.load(os.join(MODEL_DIR, base_name + '.pt_10'))
    # generate_samples(model, device, special_generate_dataloader, greedy_generate, os.path.join(GEN_DIR, 'special_gpt2_generate'))

if __name__ == '__main__':
    main()
