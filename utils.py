import os
import torch
import logging
import tqdm
import json
from macro import *

def check_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'total parameters: {total_params}')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'training parameters: {total_trainable_params}')
    return 

def check_files_exists(file_list):
    for file_path in file_list:
        if os.path.exists(file_path) == False:
            print("file %s: not exists" % file_path)
            return False
    return True

def test_multi_ppl(model_paths, device, dataloader, test_ppl_fn, base_name='model', writer=None):
    n_results = []
    n_model = len(model_paths)
    
    if not check_files_exists(model_paths):
        return 
    
    for i in range(n_model):
        model = torch.load(model_paths[i])
        logging.info('ppl_test for: %s' % model_paths[i])

        test_args = get_test_args(model, dataloader, base_name=base_name, device=device, writer=writer)
        result = test_ppl_fn(test_args)
        n_results.append(result)
    logging.info(str(n_results))
    return n_results

def test_multi_generate(model_paths, device, dataloader, generate_samples_fn, generate_fn, generate_tag='generate', full_test=False):
    
    n_model = len(model_paths)

    if full_test == True:
        n_generate = None
    else:
        n_generate = 50
    
    # TODO: 检查Path是否正确
    n_save_path = [os.path.join(GEN_DIR, os.path.basename(p) + '_' + generate_tag) for p in model_paths]
    
    if not check_files_exists(model_paths):
        return 
    
    for i in range(n_model):
        model = torch.load(model_paths[i])
        logging.info('generate_test for: %s' % model_paths[i])
        generate_samples_fn(model, device, dataloader, generate_fn, n_save_path[i], force_save=True, n_generate=n_generate)
    return 


def test_generate(model, device, generate_fn, generate_dataloader):

    batch = next(iter(generate_dataloader))
    for ctx, que, tgt in batch:

        pred = generate_fn(model, device, ctx, que)          
    
        logging.info('ctx: %s' % str(ctx))
        logging.info('query: %s' % str(que))
        logging.info('cand: %s' % str(tgt))
        logging.info('pred: %s' % str(pred))
        logging.info("####################")
    return 


def generate_samples(model, device, generate_dataloader, generate_fn, save_path, force_save=False, n_generate=50):

    # 实际上就是强制覆盖
    gen_data = []
    n_count = 0
    
    try:
        for batch in tqdm.tqdm(iter(generate_dataloader)):
            for ctx, que, response in batch:

                pred = generate_fn(model, device, ctx, que)              
                pred_data = {'ctx': ctx, 'query': que, 'response': response, 'pred': pred}
                gen_data.append(pred_data)
                
            n_count += 1
            if n_generate != None and n_count >= n_generate:
                break
        f = open(save_path, 'w') # 到了最后才openfile
    finally:
        json.dump(gen_data, f, indent=4)
        print("generated saved")
        f.close()
    return 

def get_train_args(model, train_dataloader, save_path='model.pt', base_name='model', eval_dataloader=None, n_epoch=10, device=None, writer=None, lr=1e-5, n_print=50):

    return {
        'model' : model,
        'train_dataloader' : train_dataloader,
        'save_path' : save_path,
        'eval_dataloader' : eval_dataloader,
        'n_epoch' : n_epoch,
        'device' : device,
        'writer' : writer,
        'lr' : lr,
        'n_print' : n_print,
        'base_name' : base_name
    }

def get_test_args(model, test_dataloader, base_name='model', device=None, writer=None, n_print=50, test_tag='test'):
    return {
        'model' : model,
        'test_dataloader' : test_dataloader,
        'device' : device,
        'writer' : writer,
        'n_print' : n_print,
        'base_name' : base_name,
        'test_tag' : test_tag
    }