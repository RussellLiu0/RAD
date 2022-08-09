from seq2seq import *

def main():

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    train_dataset = DictDataset.load_dataset(TRAIN_DD_DATASET_PATH)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=coll_dd_101,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    base_name = 'dd_seq2seq'
    device = gpu_device

    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    # model = VanillaSeq2Seq()
    model = torch.load(os.path.join(MODEL_DIR, base_name + '.pt_10'))

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
        n_epoch=20
    )
    train(train_args)

    return 

if __name__ == '__main__':
    main()
