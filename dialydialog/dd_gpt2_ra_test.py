#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt  
import tqdm
import math
import pickle
import json
import os
import numpy as np
import logging

from log import init_log
from dataset import DictDataset
from coll import *
from utils import test_multi_generate, test_multi_ppl, test_generate, generate_samples
from transformer_util import get_padding_mask
from model import DDGPT2Pred
from macro import *

# In[6]:

def test_ppl(args):

    model = args['model']
    dataloader = args['test_dataloader']
    device = args['device'] if args['device'] is not None else torch.device('cpu')
    n_print = args['n_print']
    writer = args['writer']
    base_name = args['base_name']
    tag = args['test_tag']

    n_step = 0
    n_all = len(dataloader)

    n_loss = []
    
    model.eval()
    model.stat = EVAL_STAT
    model.to(device)
    
    with torch.no_grad():
        for que, src, q_lens, s_lens, tgt in tqdm.tqdm(iter(dataloader)):
            

            que = que.to(device)
            src = src.to(device)
            tgt = tgt.to(device) 


            y = model(que, src, q_lens, s_lens)# 输出变化
            lm = y[0]
            lm = lm.transpose(-1, -2)

            l1 = F.cross_entropy(lm, tgt, ignore_index=-1)
            
            n_loss.append(l1.item())

            n_step += 1
            if n_step % n_print == 1:
                logging.info('%d/%d step: loss: %f ppl: %f, avg_loss: %f' % (n_step, n_all, l1.item(), math.exp(l1.item()), sum(n_loss) / len(n_loss))) # 增加 总体进行程度说明。
            
            # if n_step == 10:
            #     break

        avg_loss = sum(n_loss) / len(n_loss)
        avg_ppl = math.exp(avg_loss)
        
        logging.info('eval over: avg_loss: %f avg_ppl: %f' % (avg_loss, avg_ppl))

    if writer != None:
        writer.add_hparams({'model' : base_name + '_' + str(model.n_epoch), 'tag':tag}, {'ppl' : avg_ppl})

    return avg_ppl

# In[8]:

# tok dependance
def greedy_generate(model, device, context, query):

    model.to(device)
    model.eval()
    model.stat = EVAL_STAT
            
    que = tok.encode(query)
    # 会不会太激进了？
    # src = c1 + c2 + que + [tok.eos_token_id]

    src = [tok.eos_token_id]

    pre_len = len(src)
    q_lens = [len(que)]
    s_lens = [len(src)]


    que = torch.LongTensor([que]).to(device)
    src = torch.LongTensor([src]).to(device)

    n_max = MAX_GEN_LEN
    n_gen = 0
#     set_trace()
    with torch.no_grad():
        eos_id = tok.eos_token_id
        for i in range(n_max):        

            y = model(que, src, q_lens, s_lens) # 子回归生成
            ### 增加idx 0
            lm = y[0]
            o = torch.argmax(lm[0, -1, :])
            if o.item() == eos_id:
                break
            else:
                n_gen += 1
                src = torch.cat([src, torch.tensor([[o]], device=device)], dim=-1)
                s_lens[0] += 1
#     set_trace()
    gen = src[0][pre_len:pre_len + n_gen].tolist()
    
    return tok.decode(gen)   


# In[10]:

def main():
    
    test_dataset = DictDataset.load_dataset(TEST_DD_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    ppl_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=coll_dd_101,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )
    generate_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=coll_dd_111, # 生成Dataloader实际上是一样的
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    base_name = 'dd_model_pred'
    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    device = gpu_device

    summary_dir = os.path.join(EVAL_DIR, base_name, 'eval')
    writer = SummaryWriter(summary_dir)

    model_suffix = list(range(1, 6+1))
    model_paths = [os.path.join(MODEL_DIR, base_name + '.pt_' + str(x)) for x in model_suffix]

    # for dd_gpt2_pred == 6
    model_paths = [os.path.join(FINAL_MODEL_DIR, base_name + '.pt_6')]
    
    # n_ppl = test_multi_ppl(model_paths, device, ppl_dataloader, test_ppl, base_name=base_name, writer=writer)
    test_multi_generate(model_paths, device, generate_dataloader, generate_samples, greedy_generate, full_test=True)


    # n_ppl = test_multi_ppl(model_paths, device, ppl_dataloader, test_ppl, writer=writer, base_name=base_name)
    # print(n_ppl)
    # test_multi_generate(model_paths, device, generate_dataloader, generate_samples, greedy_generate, generate_tag='generate')

if __name__ == '__main__':
    main()
