import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import pickle
import os
import json
import tqdm

import logging
from log import init_log
from coll import *
from dataset import DictDataset

import numpy as np

from model import Seq2SeqPred
from macro import *
from utils import get_test_args, get_train_args
from seq2seq_test import test_ppl


# In[2]:

# all_default params with GPT2 Em

        
# In[15]:

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

    if model.n_epoch == n_epoch:
        logging.info("model trained for %d epoch, stop" % (model.n_epoch))
        return 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in range(model.n_epoch + 1, n_epoch + 1):

        # ....
        model.to(device)
        model.train()

        n_step = model.n_step
        n_all = len(train_dataloader)
        n_loss = []
        
        for enc, enc_len, src, src_len, dst in tqdm.tqdm(train_dataloader):
            

            enc = enc.to(device)
            enc_len = enc_len.to(device)
    
            src = src.to(device)
            src_len = src_len.to(device)
            dst = dst.to(device) 
#
            optimizer.zero_grad()
 
            y = model(enc, enc_len, src, src_len)# 输出变化
            lm = y[0].transpose(-1, -2)

            l1 = F.cross_entropy(lm, dst, ignore_index=-1)
            l2 = y[1]

            l = l1 + l2
            l.backward()

            n_loss.append(l1.item())
            
            optimizer.step()

            n_step += 1
            model.n_step += 1

            if n_step % n_print == 1:
                ppl = math.exp(l1.item())
                logging.info(f'n_epoch {i} -- {n_step}/{n_all} step: cur loss: {l1.item()}, cur_ppl: {ppl}')
                logging.info(f'n_epoch {i} -- {n_step}/{n_all} step: cur mse: {l2.item()}')
                avg_ppl = math.exp(sum(n_loss) / len(n_loss))
                if writer != None and n_step != 1:
                    writer.add_scalar('LMLoss/train', avg_ppl, n_step)
                    writer.add_scalar('CUR_LMLoss/train', ppl, n_step)
        
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

# In[18]:

def main():

    train_dataset = DictDataset.load_dataset(TRAIN_PC_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    # for double persona dataset
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=coll_pc_201,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    base_name = 'seq2seq_pred'
    device = gpu_device

    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    model = Seq2SeqPred()
    
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
        lr=DEFAULT_SEQ_LR, 
        n_epoch=10
    )
    train(train_args)

    return 

if __name__ == '__main__':
    main()

# In[ ]:


## io case 

# enc = torch.arange(0, 9).reshape(3, 3)
# enc_mask = torch.LongTensor([1, 2, 3])
# src = torch.arange(50, 62).reshape(3, 4)
# src_mask = torch.LongTensor([2, 3, 1, 4])

# # y = model(enc, enc_mask, src, src_mask) # error ! propagation

# model.trans.generate_square_subsequent_mask(5) # miss mask

