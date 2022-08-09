#!/usr/bin/env python
# coding: utf-8
# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import os 
import signal
import json
import pickle
import math
import numpy as np
import tqdm

from transformer_util import *
from model import DDFFModel
from log import init_log, logging
from dataset import DictDataset
from coll import *
from dd_final_test import test_ppl
from utils import get_train_args, get_test_args

from macro import *


# In[4]:

def train(args):
    
    model = args['model']
    lr = args['lr']
    train_dataloader = args['train_dataloader']
    eval_dataloader = args['eval_dataloader']
    n_epoch = args['n_epoch']
    device = args['device']
    save_path = args['save_path']
    base_name = args['base_name']
    n_print = args['n_print']
    writer = args['writer']

    model.to(device)
    model.train()

    #####################
    model.stat = TRAIN_STAT
    mu = 7
    ####################

    n_step = model.n_step

    if model.n_epoch == n_epoch:
        logging.info("model trained for %d epoch, stop" % (model.n_epoch))
        return 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in range(model.n_epoch + 1, n_epoch + 1):

        ################################
        model.set_select_rate(model.n_epoch, mu=mu)
        logging.info(f'select rate: {model.select_rate}')
        #########################################
    
        n_all = len(train_dataloader)
        n_loss = []

        for que, src, q_lens, s_lens, tgt in tqdm.tqdm(iter(train_dataloader)):
            
            # ctx = ctx.to(device)
            que = que.to(device)
            src = src.to(device)
            tgt = tgt.to(device) 
            # c2 = c2.to(device)

            optimizer.zero_grad()

            y = model(que, src, q_lens, s_lens)# 输出变化
            lm = y[0]
            lm = lm.transpose(-1, -2)

            l1 = F.cross_entropy(lm, tgt, ignore_index=-1)
            l2 = y[1]
            
            n_loss.append(l1.item())

            l = l1 + l2
            l.backward()
            # l1.backward()
            # l2.backward()
        
            optimizer.step()

            n_step += 1
            model.n_step += 1

            if n_step % n_print == 1:
                ppl = math.exp(l1.item())
                avg_ppl = math.exp(sum(n_loss) / len(n_loss))
                logging.info(f'n_epoch {i} -- {n_step}/{n_all} step: cur loss: {l1.item()}, cur_ppl: {ppl}')
                logging.info(f'n_epoch {i} -- {n_step}/{n_all} step: cur kl: {l2.item()}')

                if writer != None and n_step != 1:
                    writer.add_scalar('LMLoss/train', avg_ppl, n_step)
                    writer.add_scalar('CUR_LMLoss/train', ppl, n_step)
                    writer.add_scalar('CUR_KLLoss/train', l2.item(), n_step)
            # break
            
        avg_ppl = math.exp(sum(n_loss) / len(n_loss))
        logging.info(f"n_epoch {i} -- avg_ppl: {avg_ppl}")

        model.n_epoch = i
        model.cpu()
        torch.save(model, save_path + '_' + str(i)) # name ...
        model.to(device)

    return 


# In[13]:

def main():

    train_dataset = DictDataset.load_dataset(TRAIN_DD_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")


    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=coll_dd_103,
        batch_size= DEFAULT_TINY_BATCH_SIZE , 
        shuffle=False,
    )

    base_name = 'dd_final'
    # base_name = 'dd_ffmodel_mu4'
    device = gpu_device

    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    save_path = os.path.join(MODEL_DIR, base_name + '.pt')

    summary_dir = os.path.join(EVAL_DIR, base_name, 'train')
    writer = SummaryWriter(summary_dir)

    model = DDFFModel()
    # model = torch.load(os.path.join(MODEL_DIR, base_name + '.pt_3'))
    model.stat = TRAIN_STAT

    train_args = get_train_args(
        model, 
        train_dataloader, 
        save_path=save_path, 
        base_name=base_name, 
        eval_dataloader=None, 
        device=device, 
        writer=writer, 
        lr=DEFAULT_LR, 
        n_epoch=6
    )
    train(train_args)

if __name__ == '__main__':
    main()



# %%
