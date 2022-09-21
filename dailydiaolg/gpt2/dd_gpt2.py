from gpt2 import *

# In[3]:

def main():

    train_dataset = DictDataset.load_dataset(TRAIN_DD_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    # for twitter/dd dataset
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=coll_dd_102,
        batch_size=DEFAULT_SMALL_BATCH_SIZE, 
        shuffle=False,
    )

    base_name = 'dd_gpt2'
    device = gpu_device
    
    model = GPT2()
    # model = torch.load(os.path.join(MODEL_DIR, base_name + '.pt_3'))

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
        n_epoch=DEFAULT_EPOCHS
    )

    train(train_args)

if __name__ == '__main__':
    main()
