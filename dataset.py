import tqdm
import pickle
import os
import random
from torch.utils.data import Dataset
import copy
from macro import *

def show_dataset(dataset, datalen=5):
    for i in range(datalen):
        print(dataset[i])
    print('-------------------------------')
    return 


class DictDataset(Dataset):
    def __init__(self, path):

        self.datas = []
        converter = tok.encode
        #tokenizer 的一个方法
        with open(path, 'r') as f:
            lines = f.read().split('\n')[:-1]
            print(len(lines))
            
        for i in list(range(len(lines)))[::2]:
            data = {
                'query' : converter(lines[i]), 
                'response' : converter(lines[i+1]),
            }
            self.datas.append(data)
        return 
                
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return len(self.datas)

    @staticmethod
    def load_dataset(path : str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            print(f"load {path}, Length: {len(obj)}")
        return obj

    @staticmethod
    def save_dataset(dataset, path : str):
        if os.path.exists(path):
            info = input('dataset existed, do you want to overwrite ? [y/N]')
            if info != 'y' and info != 'Y':
                print('operation cancelled by user')
                return 
            else:
                print('overwrite existed file')
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return 


class DictDataset(Dataset):
    def __init__(self, pds):

        self.datas = []
        converter = tok.encode
        for pd in pds:
            persona_1 = [converter(' '.join(p)) for p in pd.p1] # 只使用Response的persona信息
            persona_2 = [converter(' '.join(p)) for p in pd.p2]
            
            context = []
            for i in range(len(pd.conv)):
                query = converter(' '.join(pd.conv[i][0]))
                response = converter(' '.join(pd.conv[i][1]))
                
                context_copy = context.copy()
                
                data = {
                    'persona_1' : persona_1,
                    'persona_2' : persona_2,
                    'context' : context_copy, 
                    'query' : query, 
                    'response' : response
                }
                
                self.datas.append(data)
                context.append([query, response])
        return 
                
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return len(self.datas)

    @staticmethod
    def load_dataset(path : str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            print(f"load {path}, Length: {len(obj)}")
        return obj

    @staticmethod
    def save_dataset(dataset, path : str):
        if os.path.exists(path):
            info = input('dataset existed, do you want to overwrite ? [y/N]')
            if info != 'y' and info != 'Y':
                print('operation cancelled by user')
                return 
            else:
                print('overwrite existed file')
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return 

class PersonaNSPDataset(Dataset):

    def __init__(self, dataset : DictDataset):
        self.datas = []

        # 搞不懂，这个data copy的问题
        all_len = len(dataset)
        for i in range(len(dataset)):
            data = copy.deepcopy(dataset[i])
            data2 = copy.deepcopy(dataset[i])
            data.pop('context') # 禁用context
            data2.pop('context')
            data['nsp_tag'] = 1
            self.datas.append(data)
            
            # 两次选中同一个认为是小概率事件
            rnd = random.randrange(0, all_len)
            if rnd == i:
                rnd = random.randrange(0, all_len)

            data2['nsp_tag'] = 0
            data2['response'] = dataset[rnd]['response']
            self.datas.append(data2)

        return
    
                
    def __getitem__(self, idx):
        return self.datas[idx]
    
    def __len__(self):
        return len(self.datas)

    @staticmethod
    def load_dataset(path : str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            print(f"load {path}, Length: {len(obj)}")
        return obj

    @staticmethod
    def save_dataset(dataset, path : str):
        if os.path.exists(path):
            info = input('dataset existed, do you want to overwrite ? [y/N]')
            if info != 'y' and info != 'Y':
                print('operation cancelled by user')
                return 
            else:
                print('overwrite existed file')
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return 

def main():
    dataset = DictDataset.load_dataset(TRAIN_PC_DATASET_PATH)
    show_dataset(dataset)

if __name__ == '__main__':
    main()