import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformer_util import get_padding_mask, get_bidirection_mask, get_pretrained_mask, generate_square_subsequent_mask
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
import numpy as np
import random
import logging

from macro import *

class PositionalEncoding(nn.Module):
    """默认的PostionEmbedding，固定参数，不可变

    Args:
        nn ([type]): [description]
    """
    def __init__(self, d_model=768,dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LanguageModel(nn.Module):
    
    def __init__(self):
        
        super(LanguageModel, self).__init__()

        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.n_step = 0
        self.d_model = DEFAULT_D_MODEL

        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False

        self.rnn = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )

        self.output = nn.Linear(self.d_model, self.vocab_size)
 
    def forward(self, query, q_lens):

        src = self.wte(query)
        
        packed = pack_padded_sequence(src, q_lens, batch_first=True, enforce_sorted=False)
        src_output, _ = self.rnn(packed)
        src_output = pad_packed_sequence(src_output, batch_first=True)[0] # 最后只需要这个输出的第一部分

        y = self.output(src_output)
        
        return y

class VanillaTransformer(nn.Module):
    
    def __init__(self):
        
        super(VanillaTransformer, self).__init__()
        
        
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.ff_size = 1024
        self.drop_rate = 0.1
        self.d_model = DEFAULT_D_MODEL
        self.n_enc = 4
        self.n_dec = 4
        self.n_head = DEFAULT_N_HEAD
        self.n_step = 0
        self.n_epoch = 0
        
        self.trans = nn.Transformer(
            d_model=self.d_model, 
            nhead=self.n_head, 
            num_encoder_layers=self.n_enc, 
            num_decoder_layers=self.n_dec, 
            dim_feedforward=self.ff_size, 
            dropout=self.drop_rate
        ) # default
        
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False

        self.pte = PositionalEncoding()
        
        self.output = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, src, src_mask, dst, dst_mask):
        
        src = self.wte(src)
        src = src + self.pte(src)
        dst = self.wte(dst) 
        dst = dst + self.pte(dst)
        
        src = src.transpose(0, 1)
        dst = dst.transpose(0, 1)
    
        n_src_mask = self.trans.generate_square_subsequent_mask(src.size(0))
        n_dst_mask = self.trans.generate_square_subsequent_mask(dst.size(0))
        
        n_src_mask = n_src_mask.to(src.device)
        n_dst_mask = n_dst_mask.to(dst.device)

        y = self.trans(src, dst, src_mask=n_src_mask, tgt_mask=n_dst_mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=dst_mask)
        y = y.transpose(0, 1)
        y = self.output(y)
        return y

class VanillaSeq2Seq(nn.Module):
    
    def __init__(self):

        super(VanillaSeq2Seq, self).__init__()
        
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.drop_rate = 0.1
        self.d_model = DEFAULT_D_MODEL
        self.d_emb = DEFAULT_D_MODEL
        
        self.n_step = 0
        self.n_epoch = 0
        
        self.enc = nn.GRU( 
            input_size=self.d_emb,
            hidden_size=self.d_model,
            batch_first=True
        )
        self.dec = nn.GRU( 
            input_size=self.d_emb,
            hidden_size=self.d_model,
            batch_first=True
        )
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        self.wte = nn.Embedding(self.vocab_size, self.d_model)
        self.wte.weight.data.copy_(wte_weight)
        self.wte.weight.requires_grad = False
        
        # self.wte = nn.Embedding(self.vocab_size, self.d_emb)
        
        self.output = nn.Linear(self.d_model, self.vocab_size)
        
    # ignore mem mask
    def forward(self, src, src_len, dst, dst_len):

        src = self.wte(src)  //que
        dst = self.wte(dst)  //src
        
        packed = pack_padded_sequence(src, src_len, batch_first=True, enforce_sorted=False)
        _, final_state = self.enc(packed)
        # final state是B*1*维度的张量
        # src_out, _ = pad_packed_sequence(src_out, batch_first=True)
        packed = pack_padded_sequence(dst, dst_len, batch_first=True, enforce_sorted=False)
        # 返回packedsequence，这里没有排序，所以sorted项为false
        # set_trace()
        dst_out, _ = self.dec(packed, final_state)
        dst_out, _ = pad_packed_sequence(dst_out, batch_first=True)
        # hn = hn.transpose(0, 1)
        y = self.output(dst_out)
        
        return y,

class Seq2SeqPred(nn.Module):
    
    def __init__(self):

        super(Seq2SeqPred, self).__init__()
        
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.drop_rate = 0.1
        self.d_model = DEFAULT_D_MODEL
        
        self.n_step = 0
        self.n_epoch = 0

        self.param1 = 0
        self.param2 = 1
        self.stat = TRAIN_STAT
        
        self.enc = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )
        self.dec = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.output = nn.Linear(self.d_model, self.vocab_size)

        self.cattn = DecayContextAttn()
        self.ll = nn.Linear(self.d_model, self.d_model)
        
    # ignore mem mask
    def forward(self, src, src_len, dst, dst_len):

        src = self.wte(src)
        dst = self.wte(dst)

        ### 单纯省事
        c_pri_output = src
        c_pri_lens = src_len

        c_output = self.cattn(src, dst, src_len, dst_len)

        c_aft_output = self.ll(c_pri_output)

        if self.stat == TRAIN_STAT:
            c_multi_output = self.param1 * c_aft_output + self.param2 * c_output
        else:
            c_multi_output = c_aft_output

        if self.stat == TRAIN_STAT:
            if self.n_step <= 16429:
                self.param2 = 1 - self.n_step / 16429
                self.param1 = 1 - self.param2
            if self.n_step % 100 == 0:
                print(f'param2: {self.param2}')
            
        
        packed = pack_padded_sequence(c_multi_output, src_len, batch_first=True, enforce_sorted=False)
        _, final_state = self.enc(packed)
        # src_out, _ = pad_packed_sequence(src_out, batch_first=True)
        packed = pack_padded_sequence(dst, dst_len, batch_first=True, enforce_sorted=False)
        dst_out, _ = self.dec(packed, final_state)
        dst_out, _ = pad_packed_sequence(dst_out, batch_first=True)
        # hn = hn.transpose(0, 1)
        y = self.output(dst_out)

        latent_loss = F.mse_loss(c_aft_output, c_output, reduction='sum')
        # l2 = F.mse_loss(c_aft_output, c_pri_output, reduction='sum')
        # latent_loss = torch.tensor([0])

        return y, latent_loss

class Seq2SeqSelf(nn.Module):
    """
    这个是非Teacher-Forcing的版本Seq2Seq，是试验方案的seq2seq版本的一部分

    Args:
        nn ([type]): [description]
    """
    
    def __init__(self):

        super(Seq2SeqSelf, self).__init__()
        
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.drop_rate = 0.1
        self.d_model = DEFAULT_D_MODEL
        
        self.n_step = 0
        self.n_epoch = 0
        self.train_flag = True
        self.tau = 0.5
        self.select_rate = 0
        
        self.enc = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )
        self.dec = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.output = nn.Linear(self.d_model, self.vocab_size)
        
    def set_select_rate(self, e, mu=12):
        p = mu / (mu + math.exp(e / mu))
        self.select_rate = p
        return 
    # ignore mem mask
    def forward(self, src, src_len, dst, dst_len):

        src = self.wte(src)
        dst = self.wte(dst)
        
        packed = pack_padded_sequence(src, src_len, batch_first=True, enforce_sorted=False)
        _, final_state = self.enc(packed)
        # final_state = final_state.transpose(0, 1)
        
        # set_trace()
        final_output = []
        h_0 = self.one_step(dst[:,[0],:], final_state)
        final_output.append(h_0)
        
        for i in range(dst.size(1) - 1):
            h_input = self.Sampling(h_0, dst[:,[i+1],:])
            h_0 = self.one_step(h_input, h_0)
            final_output.append(h_0)
    
        final_output = pad_sequence(final_output, batch_first=True)
        final_output = final_output.squeeze(1)
        final_output = final_output.transpose(0, 1)
            
        y = self.output(final_output)
        return y,
    
    def one_step(self, cur_input, h_0):

        _, h_n = self.dec(cur_input, h_0)
        return h_n

    def Sampling(self, h_input, h_init):

        if self.train_flag == True:

            oracle_words = torch.argmax(F.gumbel_softmax(h_input.transpose(0, 1), tau=self.tau), dim=-1)

            if random.random() > self.select_rate:
                return self.wte(oracle_words)
            else:
                return h_init
        else:
            return h_init

class FinalSeq2Seq(nn.Module):
    """
    融合两种方案的seq2seq

    Args:
        nn ([type]): [description]
    """
    
    def __init__(self):

        super(FinalSeq2Seq, self).__init__()
        
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.drop_rate = 0.1
        self.d_model = DEFAULT_D_MODEL
        
        self.n_step = 0
        self.n_epoch = 0

        self.param1 = 0
        self.param2 = 1
        self.select_rate = 0
        self.tau = 0.5
        self.stat = TRAIN_STAT

        self.cattn = DecayContextAttn()
        
        self.enc = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )
        self.dec = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.output = nn.Linear(self.d_model, self.vocab_size)
        self.ll = nn.Linear(self.d_model, self.d_model)
        
    def set_select_rate(self, e, mu=12):
        # 计算当前epoch应该用多大的概率选择oracle-word

        p = mu / (mu + math.exp(e / mu))
        self.select_rate = p
        return 
    # ignore mem mask
    def forward(self, src, src_len, dst, dst_len):

        src = self.wte(src)
        dst = self.wte(dst)

        if self.stat == TRAIN_STAT:  
            packed = pack_padded_sequence(src, src_len, batch_first=True, enforce_sorted=False)
            _, final_state = self.enc(packed)

            # set_trace()
            final_output = []
            h_0 = self.one_step(dst[:,[0],:], final_state)
            final_output.append(h_0)
            
            for i in range(dst.size(1) - 1):
                h_input = self.Sampling(h_0, dst[:,[i+1],:])
                h_0 = self.one_step(h_input, h_0)
                final_output.append(h_0)
        
            final_output = pad_sequence(final_output, batch_first=True)
            final_output = final_output.squeeze(1) # B * 1 * L * D
            final_output = final_output.transpose(0, 1)

            # 修改后的dst变量为cand
            cand = self.output(final_output)
            cand = torch.argmax(cand, dim=-1)
            cand = self.wte(cand)

            c_output = self.cattn(src, cand, src_len, dst_len)
        else:
            # 不进行计划采样
            # 空动作
            pass

        c_aft_output = self.ll(src)

        if self.stat == TRAIN_STAT:
            c_multi_output = self.param1 * c_aft_output + self.param2 * c_output
        else:
            c_multi_output = c_aft_output

        if self.stat == TRAIN_STAT:
            if self.n_step <= 9507:
                self.param2 = 1 - self.n_step / 9507
                self.param1 = 1 - self.param2
            if self.n_step % 100 == 0:
                logging.info(f'param2: {self.param2}')
            
        packed = pack_padded_sequence(c_multi_output, src_len, batch_first=True, enforce_sorted=False)
        _, final_state = self.enc(packed)
        # src_out, _ = pad_packed_sequence(src_out, batch_first=True)
        packed = pack_padded_sequence(dst, dst_len, batch_first=True, enforce_sorted=False)
        dst_out, _ = self.dec(packed, final_state)
        dst_out, _ = pad_packed_sequence(dst_out, batch_first=True)
        # hn = hn.transpose(0, 1)
        y = self.output(dst_out)

        if self.stat == TRAIN_STAT:
            latent_loss = F.mse_loss(c_aft_output, c_output, reduction='sum')
            return y, latent_loss
        else:
            return y, 
    
    def one_step(self, cur_input, h_0):
        _, h_n = self.dec(cur_input, h_0)
        return h_n

    def Sampling(self, h_input, h_init):
        """从预测分布中以及原始词嵌入中决定是否替换
        Args:
            h_input ([type]): 预测词概率分布，使用gumbel-softmax
            h_init ([type]): 原始输入单词的嵌入

        Returns:
            [type]: rnn下一次迭代的输入嵌入
        """

        if self.stat == TRAIN_STAT:

            oracle_words = torch.argmax(F.gumbel_softmax(h_input.transpose(0, 1), tau=self.tau), dim=-1)    
            if random.random() > self.select_rate:
                return self.wte(oracle_words)
            else:
                return h_init
        else:
            return h_init

def dot_attn(query, key, value, lens_mask=None):
    """普通点积注意力，计算了加上value之后的最终结果, key必须和value一样
        lens_mask是value的mask，某些value是padding的位置，所以应该给出value的长度

    Args:
        query ([type]): [description]
        key ([type]): [description]
        value ([type]): [description]

    Returns:
        [type]: [description]
    """

    weight = torch.matmul(query, key.transpose(-1, -2))

    if lens_mask != None:
        for i in range(len(lens_mask)): 
            # 似乎是这样
            weight[i, :, lens_mask[i]:] = float('-inf')

    score = torch.softmax(weight, dim=-1)
    attn = torch.matmul(score, value)
    query = query + attn
    return query

class AttnSeq2Seq(nn.Module):
    
    def __init__(self):

        super(AttnSeq2Seq, self).__init__()
        
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.drop_rate = 0.1
        self.d_model = DEFAULT_D_MODEL
        
        self.n_step = 0
        self.n_epoch = 0
        
        self.enc = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )
        self.dec = nn.GRU( 
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True
        )
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.output = nn.Linear(self.d_model, self.vocab_size)
        

    # ignore mem mask
    def forward(self, src, src_len, dst, dst_len):

        src = self.wte(src)
        dst = self.wte(dst)
        
        packed = pack_padded_sequence(src, src_len, batch_first=True, enforce_sorted=False)
        enc_state, final_state = self.enc(packed)
        # src_out, _ = pad_packed_sequence(src_out, batch_first=True)
        packed = pack_padded_sequence(dst, dst_len, batch_first=True, enforce_sorted=False)
        # set_trace()
        dst_out, _ = self.dec(packed, final_state)
        dst_out, _ = pad_packed_sequence(dst_out, batch_first=True)
        # hn = hn.transpose(0, 1)

        enc_state, enc_lens = pad_packed_sequence(enc_state, batch_first=True)
        # 应该等于src_lens
        
        dst_out = dot_attn(dst_out, enc_state, enc_state, enc_lens)
        y = self.output(dst_out)
        
        return y

class GPT2(nn.Module):
    
    def __init__(self):
        
        super(GPT2, self).__init__()
        
        self.enc = torch.load(PRETRAINED_PATH)
        self.enc.transformer.wte.weight.requires_grad = False
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.n_step = 0
        self.d_model = DEFAULT_D_MODEL
 
    def forward(self, query, q_lens):
        
        query_mask = torch.tensor(get_pretrained_mask(q_lens, padding_len=query.size(1))).to(query.device)
        output = self.enc(query, attention_mask=query_mask)[0] 
        return output, 

class GPT2Revised(nn.Module):
    
    def __init__(self):
        # 去掉Grad固定
        
        super(GPT2Revised, self).__init__()
        
        self.enc = torch.load(PRETRAINED_PATH)
        # self.enc.transformer.wte.weight.requires_grad = False
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.n_step = 0
        self.d_model = DEFAULT_D_MODEL
 
    def forward(self, query, q_lens):
        
        query_mask = torch.tensor(get_pretrained_mask(q_lens, padding_len=query.size(1))).to(query.device)
        output = self.enc(query, attention_mask=query_mask)[0] 
        return output

class NLIDecoder(nn.Module):
    def __init__(self):
        """构建NLIDecoder

        Args:
        """
    
        super(NLIDecoder, self).__init__()
        
        self.drop_rate = 0.1
        self.d_model = DEFAULT_D_MODEL
        self.n_head = DEFAULT_N_HEAD
        
        self.pretrained = torch.load(PRETRAINED_PATH)
        self.attn = nn.MultiheadAttention(self.d_model, self.n_head)
    
    def forward(self, tgt, memory, tgt_lens, memory_lens):
        """输出部分

        Args:
            tgt (Tensor): 解码器的输入部分，ctx + eos + response
            memory (Tensor): 注意力、记忆部分
            tgt_lens (List): 解码器输出长度
            memory_lens

        Returns:
            [type]: [description]
        """
        
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        memory_padding_mask = torch.BoolTensor(get_padding_mask(memory_lens)).to(tgt.device)
        tgt += self.attn(tgt, memory, memory, key_padding_mask=memory_padding_mask)[0]        
        tgt = tgt.transpose(0, 1)
        
        tgt_padding_mask = torch.BoolTensor(get_padding_mask(tgt_lens)).to(tgt.device)
        tgt_padding_mask = tgt_padding_mask.bitwise_not().int()
        
        y = self.pretrained(attention_mask=tgt_padding_mask, inputs_embeds=tgt)[0]
        return y

class YetAnotherDecoder(nn.Module):
    def __init__(self):
        """构建我就是想用拼接形式的，消融实验
        Args:
        """
        super(YetAnotherDecoder, self).__init__()
        
        self.pretrained = torch.load(PRETRAINED_PATH)
    
    def forward(self, tgt, memory, tgt_lens, memory_lens=None):
        """输出部分

        Args:
            tgt (Tensor): 解码器的输入部分，ctx + eos + response
            memory (Tensor): 注意力、记忆部分
            tgt_lens (List): 解码器输出长度
            memory_lens (长度，但是，这是一个拼接值，不好计算长度, optional): 干脆不用把. Defaults to None.

        Returns:
            [type]: [description]
        """
        
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        # 没有长度标记
        tgt += self.attn(tgt, memory, memory)[0]        
        tgt = tgt.transpose(0, 1)

        tgt = torch.cat([memory])
        
        tgt_padding_mask = torch.BoolTensor(get_padding_mask(tgt_lens)).to(tgt.device)
        tgt_padding_mask = tgt_padding_mask.bitwise_not().int()
        
        y = self.pretrained(attention_mask=tgt_padding_mask, inputs_embeds=tgt)[0]
        
        return y

class PersonaNLI(nn.Module):
    
    def __init__(self):
        # no use of ndim
        super(PersonaNLI, self).__init__()
        
        self.drop_rate = 0.1
        self.d_model = DEFAULT_D_MODEL
        self.n_head = DEFAULT_N_HEAD

        self.attn = nn.MultiheadAttention(self.d_model, self.n_head)
        self.anti_attn = nn.MultiheadAttention(self.d_model, self.n_head)
        
    def forward(self, tgt, memory, tgt_lens, memory_lens):
        

        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        # 未使用
        # tgt_mask = torch.FloatTensor(get_bidirection_mask(tgt.size(0), tgt.size(0))).to(tgt.device)
        # memory_mask = torch.FloatTensor(get_bidirection_mask(tgt.size(0), memory.size(0))).to(tgt.device)
        
        tgt_padding_mask = torch.BoolTensor(get_padding_mask(tgt_lens)).to(tgt.device)
        memory_padding_mask = torch.BoolTensor(get_padding_mask(memory_lens)).to(tgt.device)
        
        
        y = tgt + self.attn(tgt, memory, memory, key_padding_mask=memory_padding_mask)[0]
        y = y.transpose(0, 1)
        
        z = memory + self.anti_attn(memory, tgt, tgt, key_padding_mask=tgt_padding_mask)[0]
        z = z.transpose(0, 1)

        return y, z

class ContextAttn(nn.Module):
    
    def __init__(self):
        # no use of ndim
        super(ContextAttn, self).__init__()
        
        self.d_model = DEFAULT_D_MODEL
        self.n_head = DEFAULT_N_HEAD

        self.attn = nn.MultiheadAttention(self.d_model, self.n_head)
        
    def forward(self, tgt, memory, tgt_lens, memory_lens):
        
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        # tgt_padding_mask = torch.BoolTensor(get_padding_mask(tgt_lens)).to(tgt.device)
        memory_padding_mask = torch.BoolTensor(get_padding_mask(memory_lens)).to(tgt.device)
        y = tgt + self.attn(tgt, memory, memory, key_padding_mask=memory_padding_mask)[0]
        y = y.transpose(0, 1)

        return y

class DecayContextAttn(nn.Module):
    
    def __init__(self, rate=1):
        # no use of ndim
        super(DecayContextAttn, self).__init__()
        
        self.d_model = DEFAULT_D_MODEL
        self.n_head = DEFAULT_N_HEAD
        self.rate = rate
        self.attn = nn.MultiheadAttention(self.d_model, self.n_head)
        
    def forward(self, tgt, memory, tgt_lens, memory_lens):
        
        # shreshold
        if self.rate > 0.25:
            tgt = tgt.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_padding_mask = torch.BoolTensor(get_padding_mask(memory_lens)).to(tgt.device)
            y = tgt + self.rate * self.attn(tgt, memory, memory, key_padding_mask=memory_padding_mask)[0]
            y = y.transpose(0, 1)
            return y
        else:
            return tgt

class NormalGPT2(nn.Module):
    def __init__(self):
        """构建NLIDecoder

        Args:
        """
    
        super(NormalGPT2, self).__init__()

        self.pretrained = torch.load(PRETRAINED_PATH)
    
    def forward(self, tgt, tgt_lens):
              
        tgt_padding_mask = torch.BoolTensor(get_padding_mask(tgt_lens)).to(tgt.device)
        tgt_padding_mask = tgt_padding_mask.bitwise_not().int()
        
        y = self.pretrained(attention_mask=tgt_padding_mask, inputs_embeds=tgt)[0]
        return y

class GPT2Pred(nn.Module):
    """这个模型单纯的使用线性预测
    不使用计划采样的方案
    直接融合Response

    """
    
    def __init__(self):
        
        super(GPT2Pred, self).__init__()          
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.d_model = DEFAULT_D_MODEL
        self.n_step = 0
        self.stat = TRAIN_STAT
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.ndec = NormalGPT2()

        self.cattn = ContextAttn()
        self.c2attn = ContextAttn()
        self.c3attn = ContextAttn()

        self.param1 = 0
        self.param2 = 1

        self.ll = nn.Linear(self.d_model, self.d_model)
    
    # TODO: TEST ENV
    def forward(self, ctx, query, response, c2, c_lens, q_lens, r_lens, c2_lens):
        """[summary]
        Args:
            ctx ([type]): 实际是Persona
            query ([type]): 就是Post/Query
            response ([type]): 实际上不是response，而是拼接版本
            c_lens ([type]): [description]
            q_lens ([type]): [description]
            r_lens ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        que_emb = self.wte(query)
        ctx_emb = self.wte(ctx)
        res_emb = self.wte(response)
        c2_emb = self.wte(c2)

        # c1 = self.cattn(ctx_emb, que_emb, c_lens, q_lens)
        # c2 = self.c2attn(c2_emb, que_emb, c2_lens, q_lens)
        # c3 = self.c3attn(que_emb, que_emb, q_lens, q_lens)

        c1 = self.cattn(ctx_emb, res_emb, c_lens, r_lens)
        c2 = self.c2attn(c2_emb, res_emb, c2_lens, r_lens)
        c3 = self.c3attn(que_emb, res_emb, q_lens, r_lens)

        tmp = []
        for i in range(len(c_lens)):
            tmp.append(torch.cat([ c1[ i, : c_lens[i],], c2[ i, : c2_lens[i],], c3[ i, : q_lens[i],] ], dim=0))
        c_output_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens)
        c_output_lens = c_output_lens.tolist()
        c_output = pad_sequence(tmp, batch_first=True)


        tmp = []
        for i in range(len(c_lens)):
            tmp.append(torch.cat([ ctx_emb[ i, : c_lens[i],], c2_emb[ i, : c2_lens[i],], que_emb[ i, : q_lens[i],] ], dim=0))
        c_pri_output = pad_sequence(tmp, batch_first=True)
        c_pri_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens)
        c_pri_lens = c_pri_lens.tolist()
        
        # c_pri_output = c_pri_output.detach()
        
        # Linear Trans 变换之后差异巨大
        c_aft_output = self.ll(c_pri_output)

        # 获得元素数量
        ###
        if self.stat == TRAIN_STAT:
            c_multi_output = self.param1 * c_aft_output + self.param2 * c_output
        else:
            c_multi_output = c_aft_output

        if self.stat == TRAIN_STAT:
            if self.n_step <= 16429:
                self.param2 = 1 - self.n_step / 16429
                self.param1 = 1 - self.param2
            if self.n_step % 100 == 0:
                print(f'param2: {self.param2}')

        ###

        tmp = []
        for i in range(len(c_lens)):
            tmp.append(torch.cat([ c_multi_output[ i, : c_output_lens[i],], res_emb[ i, : r_lens[i],] ], dim=0))
        final_output = pad_sequence(tmp, batch_first=True)
        final_output_lens = np.array(c_output_lens) + np.array(r_lens)

        dec_output = self.ndec(final_output, final_output_lens)
        

        latent_loss = F.mse_loss(c_aft_output, c_output, reduction='sum')
        # l2 = F.mse_loss(c_aft_output, c_pri_output, reduction='sum')
        # latent_loss = torch.tensor([0])

        return dec_output, latent_loss

class DDGPT2Pred(nn.Module):
    """这个模型单纯的使用线性预测
    不使用计划采样的方案
    直接融合Response

    """
    
    def __init__(self):
        
        super(DDGPT2Pred, self).__init__()          
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.d_model = DEFAULT_D_MODEL
        self.n_step = 0
        self.stat = TRAIN_STAT
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.ndec = NormalGPT2()

        self.cattn = ContextAttn()
        self.c2attn = ContextAttn()
        self.c3attn = ContextAttn()

        self.param1 = 0
        self.param2 = 1

        self.ll = nn.Linear(self.d_model, self.d_model)
    def set_decay_rate(self):
        if self.n_step <= 16429:
            self.param2 = max(1 - self.n_step / 16429, 0.2) # p2是真实值占比从1-0
            self.param1 = 1 - self.param2 # p1代表了预测值占比，从0-1
        else:
            self.param1 = 0.8
            self.param2 = 0.2
        return 

    # TODO: TEST ENV
    def forward(self, query, response, q_lens, r_lens):
        """[summary]
        Args:
            ctx ([type]): 实际是Persona
            query ([type]): 就是Post/Query
            response ([type]): 实际上不是response，而是拼接版本
            c_lens ([type]): [description]
            q_lens ([type]): [description]
            r_lens ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        que_emb = self.wte(query)
        res_emb = self.wte(response)


        # c1 = self.cattn(ctx_emb, que_emb, c_lens, q_lens)
        # c2 = self.c2attn(c2_emb, que_emb, c2_lens, q_lens)
        # c3 = self.c3attn(que_emb, que_emb, q_lens, q_lens)

        c3 = self.c3attn(que_emb, res_emb, q_lens, r_lens)

        tmp = []
        for i in range(len(q_lens)):
            tmp.append(torch.cat([c3[ i, : q_lens[i],] ], dim=0))
        c_output_lens = np.array(q_lens)
        c_output_lens = c_output_lens.tolist()
        c_output = pad_sequence(tmp, batch_first=True)


        tmp = []
        for i in range(len(q_lens)):
            tmp.append(torch.cat([ que_emb[ i, : q_lens[i],] ], dim=0))
        c_pri_output = pad_sequence(tmp, batch_first=True)
        c_pri_lens =  np.array(q_lens)
        c_pri_lens = c_pri_lens.tolist()
        
        # c_pri_output = c_pri_output.detach()
        
        # Linear Trans 变换之后差异巨大
        c_aft_output = self.ll(c_pri_output)

        # 获得元素数量
        ###
        if self.stat == TRAIN_STAT:
            c_multi_output = self.param1 * c_aft_output + self.param2 * c_output
        else:
            c_multi_output = c_aft_output

        if self.stat == TRAIN_STAT:
            self.set_decay_rate()

        ###

        tmp = []
        for i in range(len(q_lens)):
            tmp.append(torch.cat([ c_multi_output[ i, : c_output_lens[i],], res_emb[ i, : r_lens[i],] ], dim=0))
        final_output = pad_sequence(tmp, batch_first=True)
        final_output_lens = np.array(c_output_lens) + np.array(r_lens)

        dec_output = self.ndec(final_output, final_output_lens)
        

        latent_loss = F.mse_loss(c_aft_output, c_output, reduction='sum')
        # l2 = F.mse_loss(c_aft_output, c_pri_output, reduction='sum')
        # latent_loss = torch.tensor([0])

        return dec_output, latent_loss

###
# 历史遗留问题
###
class DDModel22(nn.Module):
    """这个模型单纯的使用线性预测
    不使用计划采样的方案
    直接融合Response

    """
    
    def __init__(self):
        
        super(DDModel22, self).__init__()          
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.d_model = DEFAULT_D_MODEL
        self.n_step = 0
        self.stat = TRAIN_STAT
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.ndec = NormalGPT2()

        self.cattn = ContextAttn()
        self.c2attn = ContextAttn()
        self.c3attn = ContextAttn()

        self.param1 = 0
        self.param2 = 1

        self.ll = nn.Linear(self.d_model, self.d_model)
    def set_decay_rate(self):
        if self.n_step <= 16429:
            self.param2 = max(1 - self.n_step / 16429, 0.2) # p2是真实值占比从1-0
            self.param1 = 1 - self.param2 # p1代表了预测值占比，从0-1
        else:
            self.param1 = 0.8
            self.param2 = 0.2
        return 

    # TODO: TEST ENV
    def forward(self, query, response, q_lens, r_lens):
        """[summary]
        Args:
            ctx ([type]): 实际是Persona
            query ([type]): 就是Post/Query
            response ([type]): 实际上不是response，而是拼接版本
            c_lens ([type]): [description]
            q_lens ([type]): [description]
            r_lens ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        que_emb = self.wte(query)
        res_emb = self.wte(response)


        # c1 = self.cattn(ctx_emb, que_emb, c_lens, q_lens)
        # c2 = self.c2attn(c2_emb, que_emb, c2_lens, q_lens)
        # c3 = self.c3attn(que_emb, que_emb, q_lens, q_lens)

        c3 = self.c3attn(que_emb, res_emb, q_lens, r_lens)

        tmp = []
        for i in range(len(q_lens)):
            tmp.append(torch.cat([c3[ i, : q_lens[i],] ], dim=0))
        c_output_lens = np.array(q_lens)
        c_output_lens = c_output_lens.tolist()
        c_output = pad_sequence(tmp, batch_first=True)


        tmp = []
        for i in range(len(q_lens)):
            tmp.append(torch.cat([ que_emb[ i, : q_lens[i],] ], dim=0))
        c_pri_output = pad_sequence(tmp, batch_first=True)
        c_pri_lens =  np.array(q_lens)
        c_pri_lens = c_pri_lens.tolist()
        
        # c_pri_output = c_pri_output.detach()
        
        # Linear Trans 变换之后差异巨大
        c_aft_output = self.ll(c_pri_output)

        # 获得元素数量
        ###
        if self.stat == TRAIN_STAT:
            c_multi_output = self.param1 * c_aft_output + self.param2 * c_output
        else:
            c_multi_output = c_aft_output

        if self.stat == TRAIN_STAT:
            self.set_decay_rate()

        ###

        tmp = []
        for i in range(len(q_lens)):
            tmp.append(torch.cat([ c_multi_output[ i, : c_output_lens[i],], res_emb[ i, : r_lens[i],] ], dim=0))
        final_output = pad_sequence(tmp, batch_first=True)
        final_output_lens = np.array(c_output_lens) + np.array(r_lens)

        dec_output = self.ndec(final_output, final_output_lens)
        

        latent_loss = F.mse_loss(c_aft_output, c_output, reduction='sum')
        # l2 = F.mse_loss(c_aft_output, c_pri_output, reduction='sum')
        # latent_loss = torch.tensor([0])

        return dec_output, latent_loss


class FFReModel(nn.Module):
    """这个是最终的模型的基础上，去掉了参数Decay
        去掉了候选替换单词的emb平均操作
        总之就是简化操作

    Args:
        nn ([type]): [description]
    """
    
    def __init__(self):
        
        super(FFReModel, self).__init__()          
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.d_model = DEFAULT_D_MODEL
        self.n_step = 0
        self.stat = TRAIN_STAT
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.ndec = NormalGPT2()

        self.cattn = DecayContextAttn()
        self.c2attn = DecayContextAttn()
        self.c3attn = DecayContextAttn()

        self.select_rate = 0
        self.tau = 0.5

        self.ll = nn.Linear(self.d_model, self.d_model)
    
    def set_select_rate(self, e, mu=12):
        # 增大ORC的概率
        # 计算当前epoch应该用多大的概率选择oracle-word
        p = mu / (mu + math.exp(e / mu))
        self.select_rate = p
        return 
    
    def get_oracle_emb(self, prob, i, j):
        """output is output logits，i是batch，j是序列中的位置

        Args:
            output ([type]): [description]
            i ([type]): [description]
            j ([type]): [description]
        """
        # 禁用topk平均的做法

        oracle_words = torch.argmax(prob)
        # oracle_words_idx = torch.topk(prob[j], k=5)[-1] # topk自带取得索引的功能
        oracle_emb = self.wte(oracle_words_idx)
        oracle_emb = torch.mean(oracle_emb, dim=0)
        return oracle_emb

    def forward(self, ctx, query, response, c2, c_lens, q_lens, r_lens, c2_lens):
        """[summary]
        Args:
            TODO
        Returns:
            [type]: [description]
        """
        
        que_emb = self.wte(query)
        ctx_emb = self.wte(ctx)
        c2_emb = self.wte(c2)
        res_emb = self.wte(response)

        ###########################################
        self.cattn.rate = self.select_rate
        self.c2attn.rate = self.select_rate
        self.c3attn.rate = self.select_rate
        ###########################################

        tmp = []
        for i in range(len(c_lens)):
            tmp.append(torch.cat([ ctx_emb[ i, : c_lens[i],], c2_emb[ i, : c2_lens[i],], que_emb[ i, : q_lens[i],], res_emb[i, : r_lens[i]] ], dim=0))
        c_pri_output = pad_sequence(tmp, batch_first=True)
        c_pri_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + np.array(r_lens)

        if self.stat == TRAIN_STAT:

            # 用一定的概率进行选择，这个概率由epoch决定，select-rate是选择Truth的概率，如果随机出来的超过这个之，代表需要进行替换
            ##############
            # mask_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) 
            mask_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + 1
            # 需要+1，否则SOS可能被替换
            ##############
            select = (np.random.rand(c_pri_output.size(0), c_pri_output.size(1)) - self.select_rate) > 0

            for i in range(len(mask_lens)):
                select[i, :mask_lens[i] ] = False

            if random.random() > 0.95:
                all_size = np.sum(select.shape[-1] - np.array(mask_lens))
                sums = np.sum(select) # 这个select里面还包括了pad部分的。。。
                print(f"replace rate: {sums / all_size}")   

            output = self.ndec(c_pri_output, c_pri_lens)   
            prob = F.gumbel_softmax(output[i], tau=self.tau)          

            for i in range(len(select)):
                for j in range(len(select[i])):
                    if select[i][j] == True:
                        # all_query[i][j] = oracle_words[i][j]
                        c_pri_output[i][j] = self.get_oracle_emb(prob, i, j)
            
            new_res_emb = []
            for i in range(len(c_pri_output)):
                ## 可能会有问题
                new_res_emb.append(c_pri_output[i, mask_lens[i]-1:mask_lens[i]-1+r_lens[i]])
            new_res_emb = pad_sequence(new_res_emb, batch_first=True)
            
            c1 = self.cattn(ctx_emb, new_res_emb, c_lens, r_lens)
            c2 = self.c2attn(c2_emb, new_res_emb, c2_lens, r_lens)
            c3 = self.c3attn(que_emb, new_res_emb, q_lens, r_lens)

            tmp = []
            for i in range(len(c_lens)):
                tmp.append(torch.cat([ c1[ i, : c_lens[i],], c2[ i, : c2_lens[i],], c3[ i, : q_lens[i]], new_res_emb[i, : r_lens[i]] ], dim=0))
            c_output_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + np.array(r_lens)
            c_output_lens = c_output_lens.tolist()
            c_output = pad_sequence(tmp, batch_first=True)

        c_aft_output = self.ll(c_pri_output)

        # 获得元素数量
        ###
        if self.stat == TRAIN_STAT:
            c_multi_output = c_output
        else:
            c_multi_output = c_aft_output

        final_output = self.ndec(c_multi_output, c_pri_lens)

        if self.stat == TRAIN_STAT:
            latent_loss = F.mse_loss(c_aft_output, c_output, reduction='sum')
            return final_output, latent_loss
        else:
            return final_output, 

class FFModel(nn.Module):
    """这个是最终的模型

    Args:
        nn ([type]): [description]
    """
    
    def __init__(self):
        
        super(FFModel, self).__init__()          
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.d_model = DEFAULT_D_MODEL
        self.n_step = 0
        self.stat = TRAIN_STAT
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.ndec = NormalGPT2()

        self.cattn = DecayContextAttn()
        self.c2attn = DecayContextAttn()
        self.c3attn = DecayContextAttn()

        self.param1 = 0
        self.param2 = 1
        self.select_rate = 0
        self.tau = 0.5

        self.ll = nn.Linear(self.d_model, self.d_model)
    
    def set_select_rate(self, e, mu=12):
        # 增大ORC的概率
        # 计算当前epoch应该用多大的概率选择oracle-word
        p = mu / (mu + math.exp(e / mu))
        self.select_rate = p
        return 

    def set_decay_rate(self):
        if self.n_step <= 16429:
            self.param2 = max(1 - self.n_step / 16429, 0.2) # p2是真实值占比从1-0
            self.param1 = 1 - self.param2 # p1代表了预测值占比，从0-1
        else:
            self.param1 = 0.8
            self.param2 = 0.2
        return 
    
    def get_oracle_emb(self, prob, i, j):
        """output is output logits，i是batch，j是序列中的位置

        Args:
            output ([type]): [description]
            i ([type]): [description]
            j ([type]): [description]
        """
        # oracle_words = torch.argmax(prob)
        oracle_words_idx = torch.topk(prob[j], k=5)[-1] # topk自带取得索引的功能
        oracle_emb = self.wte(oracle_words_idx)
        oracle_emb = torch.mean(oracle_emb, dim=0)
        return oracle_emb

    def forward(self, ctx, query, response, c2, c_lens, q_lens, r_lens, c2_lens):
        """[summary]
        Args:
            TODO
        Returns:
            [type]: [description]
        """

        # tmp = []
        # for i in range(len(c_lens)):
        #     tmp.append(torch.cat([ ctx[ i, : c_lens[i]], c2[ i, : c2_lens[i],], query[ i, : q_lens[i],], response[i, : r_lens[i]] ], dim=0))
        # all_query = pad_sequence(tmp, batch_first=True)
        # all_query_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + np.array(r_lens)

        que_emb = self.wte(query)
        ctx_emb = self.wte(ctx)
        c2_emb = self.wte(c2)
        res_emb = self.wte(response)

        ###########################################
        self.cattn.rate = self.select_rate
        self.c2attn.rate = self.select_rate
        self.c3attn.rate = self.select_rate
        ###########################################

        tmp = []
        for i in range(len(c_lens)):
            tmp.append(torch.cat([ ctx_emb[ i, : c_lens[i],], c2_emb[ i, : c2_lens[i],], que_emb[ i, : q_lens[i],], res_emb[i, : r_lens[i]] ], dim=0))
        c_pri_output = pad_sequence(tmp, batch_first=True)
        c_pri_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + np.array(r_lens)

        if self.stat == TRAIN_STAT:

            # 用一定的概率进行选择，这个概率由epoch决定，select-rate是选择Truth的概率，如果随机出来的超过这个之，代表需要进行替换
            ##############
            # mask_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) 
            mask_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + 1
            # 需要+1，否则SOS可能被替换
            ##############
            select = (np.random.rand(c_pri_output.size(0), c_pri_output.size(1)) - self.select_rate) > 0

            for i in range(len(mask_lens)):
                select[i, :mask_lens[i] ] = False

            if random.random() > 0.95:
                all_size = np.sum(select.shape[-1] - np.array(mask_lens))
                sums = np.sum(select) # 这个select里面还包括了pad部分的。。。
                print(f"replace rate: {sums / all_size}")   

            output = self.ndec(c_pri_output, c_pri_lens)   
            prob = F.gumbel_softmax(output[i], tau=self.tau)          

            for i in range(len(select)):
                for j in range(len(select[i])):
                    if select[i][j] == True:
                        # all_query[i][j] = oracle_words[i][j]
                        c_pri_output[i][j] = self.get_oracle_emb(prob, i, j)
  
            # tmp = []
            # for i in range(len(c_lens)):
            #     tmp.append(torch.cat([ ctx_emb[ i, : c_lens[i],], c2_emb[ i, : c2_lens[i],], que_emb[ i, : q_lens[i],], res_emb[i, : r_lens[i]] ], dim=0))
            # c_pri_output = pad_sequence(tmp, batch_first=True)
            # c_pri_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + np.array(r_lens)
            
            new_res_emb = []
            for i in range(len(c_pri_output)):
                ## 可能会有问题
                new_res_emb.append(c_pri_output[i, mask_lens[i]-1:mask_lens[i]-1+r_lens[i]])
            new_res_emb = pad_sequence(new_res_emb, batch_first=True)
            
            c1 = self.cattn(ctx_emb, new_res_emb, c_lens, r_lens)
            c2 = self.c2attn(c2_emb, new_res_emb, c2_lens, r_lens)
            c3 = self.c3attn(que_emb, new_res_emb, q_lens, r_lens)

            # c1 c2 c3的attn不同存在差异性。
            #
            #

            tmp = []
            for i in range(len(c_lens)):
                tmp.append(torch.cat([ c1[ i, : c_lens[i],], c2[ i, : c2_lens[i],], c3[ i, : q_lens[i]], new_res_emb[i, : r_lens[i]] ], dim=0))
            c_output_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + np.array(r_lens)
            c_output_lens = c_output_lens.tolist()
            c_output = pad_sequence(tmp, batch_first=True)

        c_aft_output = self.ll(c_pri_output)

        # 这里是区别所在
        # c_aft_output = self.ll(c_pri_output)

        # 获得元素数量
        ###
        if self.stat == TRAIN_STAT:
            c_multi_output = self.param1 * c_aft_output + self.param2 * c_output
        else:
            c_multi_output = c_aft_output

        if self.stat == TRAIN_STAT:
            self.set_decay_rate()
            if self.n_step % 100 == 0:
                print(f'param2: {self.param2}')

        final_output = self.ndec(c_multi_output, c_pri_lens)

        if self.stat == TRAIN_STAT:
            latent_loss = F.mse_loss(c_aft_output, c_output, reduction='sum')
            return final_output, latent_loss
        else:
            return final_output, 

class DDFFModel(nn.Module):
    
    def __init__(self):
        
        super(DDFFModel, self).__init__()          
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.d_model = DEFAULT_D_MODEL
        self.n_step = 0
        self.stat = TRAIN_STAT
        
        wte_weight = torch.load(PRETRAINED_PATH).state_dict()['transformer.wte.weight']
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.wte.weight.data.copy_(wte_weight)
        
        self.wte.weight.requires_grad = False
        
        self.ndec = NormalGPT2()

        self.c3attn = DecayContextAttn()

        self.param1 = 0
        self.param2 = 1
        self.select_rate = 0
        self.tau = 0.5

        self.ll = nn.Linear(self.d_model, self.d_model)
    
    def set_select_rate(self, e, mu=12):
        # 增大ORC的概率
        # 计算当前epoch应该用多大的概率选择oracle-word
        p = mu / (mu + math.exp(e / mu))
        self.select_rate = p
        return 

    def set_decay_rate(self):
        if self.n_step <= 16429:
            self.param2 = max(1 - self.n_step / 16429, 0.2) # p2是真实值占比从1-0
            self.param1 = 1 - self.param2 # p1代表了预测值占比，从0-1
        else:
            self.param1 = 0.8
            self.param2 = 0.2
        return 
    
    def get_oracle_emb(self, prob, i, j):
        """output is output logits，i是batch，j是序列中的位置

        Args:
            output ([type]): [description]
            i ([type]): [description]
            j ([type]): [description]
        """
        # oracle_words = torch.argmax(prob)
        oracle_words_idx = torch.topk(prob[j], k=5)[-1] # topk自带取得索引的功能
        oracle_emb = self.wte(oracle_words_idx)
        oracle_emb = torch.mean(oracle_emb, dim=0)
        return oracle_emb

    def forward(self, query, response, q_lens, r_lens):
        """[summary]
        Args:
            TODO
        Returns:
            [type]: [description]
        """

        # tmp = []
        # for i in range(len(c_lens)):
        #     tmp.append(torch.cat([ ctx[ i, : c_lens[i]], c2[ i, : c2_lens[i],], query[ i, : q_lens[i],], response[i, : r_lens[i]] ], dim=0))
        # all_query = pad_sequence(tmp, batch_first=True)
        # all_query_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + np.array(r_lens)

        que_emb = self.wte(query)
        res_emb = self.wte(response)

        ###########################################
        self.c3attn.rate = self.select_rate
        ###########################################

        tmp = []
        for i in range(len(q_lens)):
            tmp.append(torch.cat([que_emb[ i, : q_lens[i],], res_emb[i, : r_lens[i]] ], dim=0))
        c_pri_output = pad_sequence(tmp, batch_first=True)
        c_pri_lens = np.array(q_lens) + np.array(r_lens)

        if self.stat == TRAIN_STAT:

            # 用一定的概率进行选择，这个概率由epoch决定，select-rate是选择Truth的概率，如果随机出来的超过这个之，代表需要进行替换
            ##############
            # mask_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) 
            mask_lens = np.array(q_lens) + 1
            # 需要+1，否则SOS可能被替换
            ##############
            select = (np.random.rand(c_pri_output.size(0), c_pri_output.size(1)) - self.select_rate) > 0

            for i in range(len(mask_lens)):
                select[i, :mask_lens[i] ] = False

            if random.random() > 0.95:
                all_size = np.sum(select.shape[-1] - np.array(mask_lens))
                sums = np.sum(select) # 这个select里面还包括了pad部分的。。。
                print(f"replace rate: {sums / all_size}")   

            output = self.ndec(c_pri_output, c_pri_lens)   
            prob = F.gumbel_softmax(output[i], tau=self.tau)          

            for i in range(len(select)):
                for j in range(len(select[i])):
                    if select[i][j] == True:
                        # all_query[i][j] = oracle_words[i][j]
                        c_pri_output[i][j] = self.get_oracle_emb(prob, i, j)
  
            # tmp = []
            # for i in range(len(c_lens)):
            #     tmp.append(torch.cat([ ctx_emb[ i, : c_lens[i],], c2_emb[ i, : c2_lens[i],], que_emb[ i, : q_lens[i],], res_emb[i, : r_lens[i]] ], dim=0))
            # c_pri_output = pad_sequence(tmp, batch_first=True)
            # c_pri_lens = np.array(c_lens) + np.array(c2_lens) + np.array(q_lens) + np.array(r_lens)
            
            new_res_emb = []
            for i in range(len(c_pri_output)):
                ## 可能会有问题
                new_res_emb.append(c_pri_output[i, mask_lens[i]-1:mask_lens[i]-1+r_lens[i]])
            new_res_emb = pad_sequence(new_res_emb, batch_first=True)
            
            c3 = self.c3attn(que_emb, new_res_emb, q_lens, r_lens)

            # c1 c2 c3的attn不同存在差异性。
            #
            #

            tmp = []
            for i in range(len(q_lens)):
                tmp.append(torch.cat([ c3[ i, : q_lens[i]], new_res_emb[i, : r_lens[i]] ], dim=0))
            c_output_lens =  np.array(q_lens) + np.array(r_lens)
            c_output_lens = c_output_lens.tolist()
            c_output = pad_sequence(tmp, batch_first=True)

        c_aft_output = self.ll(c_pri_output)

        # 这里是区别所在
        # c_aft_output = self.ll(c_pri_output)

        # 获得元素数量
        ###
        if self.stat == TRAIN_STAT:
            c_multi_output = self.param1 * c_aft_output + self.param2 * c_output
        else:
            c_multi_output = c_aft_output

        if self.stat == TRAIN_STAT:
            self.set_decay_rate()
            if self.n_step % 100 == 0:
                print(f'param2: {self.param2}')

        final_output = self.ndec(c_multi_output, c_pri_lens)

        if self.stat == TRAIN_STAT:
            latent_loss = F.mse_loss(c_aft_output, c_output, reduction='sum')
            return final_output, latent_loss
        else:
            return final_output, 

class GPT2Sampling(nn.Module):
    
    def __init__(self):
        
        super(GPT2Sampling, self).__init__()
        
        self.enc = torch.load(PRETRAINED_PATH)
        self.enc.transformer.wte.weight.requires_grad = False
        self.vocab_size = PRETRAINED_VOCAB_SIZE
        self.n_epoch = 0
        self.n_step = 0
        self.d_model = DEFAULT_D_MODEL
        self.select_rate = 0
        self.train_flag = True
        self.tau = 0.5

    def forward(self, query, q_lens, mask_lens):
        """Forward函数的构成方式

        Args:
            query (B * L): [description]
            q_lens (B): [description]
            mask_lens (B): 这部分参数需要，因为Input部分不应该被替换

        Returns:
            [type]: [description]
        """
        
        query_mask = torch.tensor(get_pretrained_mask(q_lens, padding_len=query.size(1))).to(query.device)
        output = self.enc(query, attention_mask=query_mask)[0]

        # 就是单纯的对Input进行修改
        if self.train_flag == True:

            # 得到可供替换（生成出来的，还是用了一个gumbel-softmax）的单词
            oracle_words = torch.argmax(F.gumbel_softmax(output, tau=self.tau), dim=-1)

            # 用一定的概率进行选择，这个概率由epoch决定，select-rate是选择Truth的概率，如果随机出来的超过这个之，代表需要进行替换
            select = (np.random.rand(oracle_words.size(0), oracle_words.size(1)) - self.select_rate) > 0

            # 前面部分不能替换
            for i in range(len(mask_lens)):
                select[i, :mask_lens[i] ] = False

            if random.random() > 0.95:
                all_size = np.sum(select.shape[-1] - np.array(mask_lens))
                sums = np.sum(select)
                print(f"oracle rate: {sums / all_size}")                

            for i in range(len(select)):
                for j in range(len(select[i])):
                    if select[i][j] == True:
                        query[i][j] = oracle_words[i][j]
            output = self.enc(query, attention_mask=query_mask)[0]

        return output,
    
    def set_select_rate(self, e, mu=12):
        # 计算当前epoch应该用多大的概率选择oracle-word

        p = mu / (mu + math.exp(e / mu))
        self.select_rate = p
        return 
        
