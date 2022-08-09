from final_seq2seq import *

def main():

    train_dataset = DictDataset.load_dataset(TRAIN_DD_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")


    # for twitter/dd dataset
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=coll_dd_101,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    # base_name = 'dd_baseline_seq2seq_pred'
    base_name = 'dd_final_seq2seq'
    device = gpu_device

    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    model = FinalSeq2Seq()
    # model = torch.load(os.path.join(MODEL_DIR, base_name + '.pt_5'))

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
        n_epoch=DEFAULT_SEQ_EPOCHS + 10
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

