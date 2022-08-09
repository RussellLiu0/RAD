
import json
import os 
import tempfile

def convert_generate_file(path : str, save_path=None):

    # generate file必须以generate结尾
    if not path.endswith('generate'):
        return False
    try:
        with open(path, 'rb') as f:
            datas = json.load(f)
    except Exception:
        print(f'{path} file has error')
        return False

    if save_path == None:
        # 
        save_path = tempfile.NamedTemporaryFile().name
    
    with open(save_path, 'w') as f:
        for d in datas:
            pair = d['pred'] + '\t' + d['response'] + '\n'
            f.write(pair)
    return save_path