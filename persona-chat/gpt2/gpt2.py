# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
import pickle
import os
import numpy as np
import logging
import json
import matplotlib.pyplot as plt  

from transformer_util import PositionalEncoding, get_pretrained_mask
from dataset import DictDataset
from log import init_log
from coll import *
from utils import get_train_args, get_test_args
from gpt2_test import test_ppl
from model import GPT2
from macro import *
import tqdm


# In[3]:

def train(args):
    
    model = args['model']
    lr = args['lr']
    train_dataloader = args['train_dataloader']
    eval_dataloader = args['eval_dataloader']
    n_epoch = args['n_epoch']
    device = args['device']
    save_path = args['save_path']
    writer = args['writer']
    n_print = args['n_print']
    base_name = args['base_name']

    model.to(device)
    model.train()

    if model.n_epoch == n_epoch:
        logging.info("model trained for %d epoch, stop" % (model.n_epoch))
        return 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in range(model.n_epoch + 1, n_epoch + 1):
        
        n_step = 0
        n_all = len(train_dataloader)
        n_loss = []

        for que, q_lens, tgt in tqdm.tqdm(train_dataloader):

            que = que.to(device)
            tgt = tgt.to(device) 
            optimizer.zero_grad()    

            y = model(que, q_lens)# 输出变化
            lm = y[0]    #？

            lm = lm.transpose(-1, -2)
            l1 = F.cross_entropy(lm, tgt, ignore_index=-1)
            # 为了符合交叉熵计算的api要求
            n_loss.append(l1.item())

            l1.backward()
            
            optimizer.step()

            n_step += 1
            model.n_step += 1

            if n_step % n_print == 1: 
                ppl = math.exp(l1.item())
                logging.info(f'n_epoch {i} -- {n_step}/{n_all} step: cur loss: {l1.item()}, cur_ppl: {ppl}')
                             
        logging.info("n_epoch %d -- avg_ppl: %f" % (model.n_epoch , math.exp(sum(n_loss) / len(n_loss))))

        model.n_epoch = i
        model.cpu()
        torch.save(model, save_path + '_' + str(i)) 
        model.to(device)

        avg_ppl = math.exp(sum(n_loss) / len(n_loss))
        logging.info(f"n_epoch {model.n_epoch} -- avg_ppl: {avg_ppl}")


        if eval_dataloader != None:
            logging.info(f'n_epoch {model.n_epoch} -- eval --')
            eval_args = get_test_args(model, eval_dataloader, base_name=base_name, device=device, writer=writer, test_tag='eval')
            test_ppl(eval_args)
    return 


# In[15]:

def main():

    train_dataset = DictDataset.load_dataset(TRAIN_PC_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    # for twitter/dd dataset
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=coll_pc_202,
        batch_size=DEFAULT_SMALL_BATCH_SIZE, 
        shuffle=False,
    )

    base_name = 'gpt2'
    device = gpu_device
    
    # model = GPT2()
    model = torch.load(os.path.join(MODEL_DIR, base_name + '.pt_5'))

    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    save_path = os.path.join(MODEL_DIR, base_name + '.pt')

    summary_dir = os.path.join(EVAL_DIR, base_name, 'train')
    writer = SummaryWriter(summary_dir)

    train_args = get_train_args(
        model, 
        train_dataloader, 
        save_path=save_path, 
        base_name=base_name, 
        eval_dataloader=None, 
        device=device, 
        writer=writer, 
        lr=DEFAULT_LR, 
        n_epoch=DEFAULT_EPOCHS + 3
    )

    train(train_args)

if __name__ == '__main__':
    main()
