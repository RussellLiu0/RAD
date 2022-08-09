import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from transformer_util import get_padding_mask
from macro import *

with open(TOK_PATH, 'rb') as f:
    tok = pickle.load(f)

def coll_pc_201(batch):
    """5参数版本，适用于Encoder-Decoder结构（seq2seq），带长度信息
    src包含了两个persona的信息

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """

    enc = []
    src = []
    dst = []    

    for data in batch:  
        query = data['query']
        response = data['response']

        c1 = data['persona_1']
        c2 = data['persona_2']
    
        que = []
        for c in c1:
            que += c + [tok.eos_token_id]
        for c in c2:
            que += c + [tok.eos_token_id]

        enc.append(que + query) 
        src.append([tok.eos_token_id] + response)
        dst.append(response + [tok.eos_token_id])
    
    enc_len = torch.tensor([len(x) for x in enc])
    src_len = torch.tensor([len(x) for x in src])
    
    enc = pad_sequence([torch.tensor(x) for x in enc], batch_first=True)
    src = pad_sequence([torch.tensor(x) for x in src], batch_first=True)
    dst = pad_sequence([torch.tensor(x) for x in dst], batch_first=True, padding_value=-1) # ... 

    return enc, enc_len, src, src_len, dst


def coll_pc_202(batch):
    """三参数版本，包含两个persona，用于拼接预训练模型

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """
    src = []
    tgt = []
    
    for data in batch:
        
        query = data['query']
        response = data['response']
        c1 = data['persona_1']
        c2 = data['persona_2']
    
        que = []
        for c in c1:
            que += c + [tok.eos_token_id]
        for c in c2:
            que += c + [tok.eos_token_id]

        que += query
        
        src.append(que + [tok.eos_token_id] + response)
        tgt.append(len(que) * [-1] + response + [tok.eos_token_id])

    src_lens = [len(x) for x in src] 
    src = pad_sequence([torch.LongTensor(x) for x in src], batch_first=True)
    tgt = pad_sequence([torch.LongTensor(x) for x in tgt], batch_first=True, padding_value=-1) # ... 
    
    return src, src_lens, tgt

def coll_pc_203(batch):
    """七参数版本，分离所有的context，包括query
    但是tgt拼接所有输入输出，用于final

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """

    c1 = []
    c2 = []
    que = []
    src = []
    tgt = []

    for data in batch:
        
        query = data['query']
        response = data['response']
        
        p1 = []
        for c in data['persona_1']:
            p1 += c + [tok.eos_token_id]
        c1.append(p1)

        p2 = []
        for c in data['persona_2']:
            p2 += c + [tok.eos_token_id]
        c2.append(p2)
        que.append(query)
        src.append([tok.eos_token_id] + response)
        tgt.append([-1] * (len(p1) + len(p2) + len(query)) + response + [tok.eos_token_id])

    c1_lens = [len(x) for x in c1]
    src_lens = [len(x) for x in src]
    que_lens = [len(x) for x in que]
    c2_lens = [len(x) for x in c2]

    c1 = pad_sequence([torch.LongTensor(x) for x in c1], batch_first=True)
    c2 = pad_sequence([torch.LongTensor(x) for x in c2], batch_first=True)
    que = pad_sequence([torch.LongTensor(x) for x in que], batch_first=True)
    src = pad_sequence([torch.LongTensor(x) for x in src], batch_first=True)
    tgt = pad_sequence([torch.LongTensor(x) for x in tgt], batch_first=True, padding_value=-1) 

    return c1, c2, que, src, c1_lens, c2_lens, que_lens, src_lens, tgt

def coll_pc_204(batch):
    """用于PC数据集，四参数版本，计划采样，
    在拼接版本的基础上，增加了一个提示不应该产生oracle词的mask
    

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """
    src = []
    tgt = []

    mask_lens = []
    
    for data in batch:
        
        query = data['query']
        response = data['response']
        c1 = data['persona_1']
        c2 = data['persona_2']
    
        que = []
        for c in c1:
            que += c + [tok.eos_token_id]
        for c in c2:
            que += c + [tok.eos_token_id]

        que += query
        
        src.append(que + [tok.eos_token_id] + response)
        tgt.append(len(que) * [-1] + response + [tok.eos_token_id])

        mask_lens.append(len(que) + 1) # 长度为que以及一个开始提示符

    src_lens = [len(x) for x in src] 
    src = pad_sequence([torch.LongTensor(x) for x in src], batch_first=True)
    tgt = pad_sequence([torch.LongTensor(x) for x in tgt], batch_first=True, padding_value=-1) # ... 
    
    return src, src_lens, tgt, mask_lens

def coll_pc_211(batch):
    
    """用于GEN，三参数版本，包含Persona1，Persona2

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """
    new = []
    decode = tok.decode
    
    for d in batch: 

        c1 = [decode(persona_list) for persona_list in d['persona_1']]
        c2 = [decode(persona_list) for persona_list in d['persona_2']]
        query = decode(d['query'])
        response = decode(d['response'])
        context = [c1, c2]
        data = [context, query, response] 
        new.append(data)
        
    return new

def coll_dd_101(batch):
    """不包含Persona，Seq2seq范式
    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """

    que = []
    src = []
    tgt = []
    for data in batch:
        query = data['query']
        response = data['response']
        que.append(query)
        src.append([tok.eos_token_id] + response)
        tgt.append(response + [tok.eos_token_id])

    src_lens =  torch.tensor([len(x) for x in src])  # 张量，记录每个句子的长度
    que_lens =  torch.tensor([len(x) for x in que])
    src = pad_sequence([torch.LongTensor(x) for x in src], batch_first=True)
    # 张量，返回L*B*维度 ，B是batchsize，L是最长句子的长度，在dd数据集里是33,32+EOS标识符
    que = pad_sequence([torch.LongTensor(x) for x in que], batch_first=True)
    tgt = pad_sequence([torch.LongTensor(x) for x in tgt], batch_first=True, padding_value=-1)
    
    return que, que_lens, src,  src_lens, tgt

    # que src 是tensor
    
def coll_dd_102(batch):
    """三参数版本，用于拼接预训练模型，消融实验，不包含Persona
    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """
    """ 
    gpt返回三个参数，seq2seq返回五个
    这里的src是输入编码器的，tgt是ground truth，用来计算loss
    torch.LongTensor(x)把tensor的列表转化为L*B的矩阵
    """
    src = []
    tgt = []
    for data in batch:
        que = data['query']
        response = data['response']
        src.append(que + [tok.eos_token_id] + response)
        tgt.append(len(que) * [-1] + response + [tok.eos_token_id])

    src_lens = [len(x) for x in src]
    src = pad_sequence([torch.LongTensor(x) for x in src], batch_first=True)
    tgt = pad_sequence([torch.LongTensor(x) for x in tgt], batch_first=True, padding_value=-1)
    
    return src, src_lens, tgt

def coll_dd_103(batch):
    """七参数版本，分离所有的context，包括query
    但是tgt拼接所有输入输出，用于dd-final
    由于DD数据集不含context，所以等同于coll-dd-101，
    这是一个编程上的失误。

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """

    que = []
    src = []
    tgt = []

    for data in batch:
        
        query = data['query']
        response = data['response']

        que.append(query)
        src.append([tok.eos_token_id] + response)
        tgt.append([-1] * (len(query)) + response + [tok.eos_token_id])

    src_lens = [len(x) for x in src]
    que_lens = [len(x) for x in que]

    que = pad_sequence([torch.LongTensor(x) for x in que], batch_first=True)
    src = pad_sequence([torch.LongTensor(x) for x in src], batch_first=True)
    tgt = pad_sequence([torch.LongTensor(x) for x in tgt], batch_first=True, padding_value=-1) 

    return que, src, que_lens, src_lens, tgt

def coll_dd_104(batch):
    """四参数版本，计划采样
    增加了一个mask，指示不产生替换的长度
    这个适用于，DD数据集，不包括persona信息

    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """
    src = []
    tgt = []

    mask_lens = []
    
    for data in batch:
        
        query = data['query']
        response = data['response']

        src.append(query + [tok.eos_token_id] + response)
        tgt.append(len(query) * [-1] + response + [tok.eos_token_id])

        mask_lens.append(len(query) + 1) # 长度为que以及一个开始提示符

    src_lens = [len(x) for x in src] 
    src = pad_sequence([torch.LongTensor(x) for x in src], batch_first=True)
    tgt = pad_sequence([torch.LongTensor(x) for x in tgt], batch_first=True, padding_value=-1) # ... 
    
    return src, src_lens, tgt, mask_lens

def coll_dd_111(batch):
    """用于generate，不含Persona
    Args:
        batch ([type]): [description]

    Returns:
        [type]: [description]
    """

    new = []
    decode = tok.decode
    
    for d in batch: 

        context = []
        query = decode(d['query'])
        response = decode(d['response'])
        data = [context, query, response] 
        new.append(data)

    return new