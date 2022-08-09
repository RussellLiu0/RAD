from eval_1 import get_eval_1, print_eval_result
from eval_2 import get_eval_2
from eval_3 import get_eval_3
from eval_util import convert_generate_file
import os
import time
import datetime
import pickle
from macro import *

# from torch.utils.tensorboard import SummaryWriter

import pandas

def get_generate_files(path, base_name=None):
    """从给定的目录中过滤生成文件
    basename： list

    Args:
        path ([type]): [description]
        base_name ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    generate_files_path = [x for x in os.listdir(path) if x.endswith('generate')]
    output_file_path = generate_files_path
    if base_name != None:
        if isinstance(base_name, str):
            output_file_path = [x for x in generate_files_path if os.path.splitext(x)[0] == base_name]
        elif isinstance(base_name, list):
            for name in base_name:
                output_file_path += [x for x in generate_files_path if os.path.splitext(x)[0] == name]
        else:
            raise TypeError("list or string for generate basename")

    output_file_path = [os.path.join(path , x) for x in output_file_path]
    output_file_path = sorted(output_file_path) # 排序方便阅读
    return output_file_path


def get_single_file(path, eval_func):
    save_path = convert_generate_file(path)
    if save_path != False:
        results = eval_func(save_path)
        print_eval_result(results)
    return 


def main():
    ts = time.strftime("%m-%d-%H-%M", time.localtime()) # 月日时分

    eval_name = 'eval3'
    eval_func = get_eval_3
    # base_name = 'baseline_transformer'
    # base_name = 'finalmodel'
    # base_name = 'baseline_gpt2'
    # base_name = 'modelacl'
    # base_name = 'ffmodel_mu4'
    # base_name = 'finalmodel_mu4'
    # base_name = 'model24'
    # base_name = 'baseline_gpt2_medium'
    # base_name = 'baseline_seq2seq_attn'
    # base_name = 'baseline_seq2seq'
    # base_name = 'baseline_gpt2revised'
    # base_name = 'baseline_seq2seq_lowdim'
    # base_name = 'dd_baseline_seq2seq'
    # base_name = 'dd_baseline_gpt2'
    base_name = 'final_seq2seq'
    # base_name = ['seq2seq_sampling', 'final_seq2seq', 'seq2seq', 'seq2seq_pred', 'pc_baseline_seq2seq_pred']
    # base_name = 'dd_seq2seq'

    with open(TOK_PATH, 'rb') as f:
        tok = pickle.load(f)


    generate_files_path = get_generate_files(GEN_DIR, base_name=base_name)

    # generate_files_path = get_generate_files(FINAL_GEN_DIR)

    df = pandas.DataFrame()
    
    for p in generate_files_path:
        print(f'cur eval: {p}')
        save_path = convert_generate_file(p)
        if save_path != False:
            results = eval_func(save_path, tok=tok, lines_lim=None)
            print_eval_result(results)
            ps = pandas.Series(results, name=os.path.basename(p))
            df = df.append(ps)
    df.to_csv('/home/kananos/kanaexper_smb/' + datetime.datetime.now().strftime('%m-%d_%H-%M') + '.csv')       



def f2():
    eval_func = get_eval_3
    # path = os.path.join(GEN_DIR, 'model16_full.pt_1_generate')
    # path = os.path.join(GEN_DIR, 'model16_full.pt_2_generate')
    # path = os.path.join(GEN_DIR, 'baseline_gpt2.pt_1_special_generate')
    # path = os.path.join(GEN_DIR, 'baseline_seq2seq_no_context.pt_3_generate')
    # path = os.path.join(GEN_DIR, 'baseline_seq2seq_attn.pt_5_generate')
    # get_single_file(path, eval_func)
    return 

if __name__ == "__main__":
    main()
    # f2()
