from final_seq2seq_test import *


def main():
    with open(TOK_PATH, 'rb') as f:
        tok = pickle.load(f)

    test_dataset = DictDataset.load_dataset(TEST_DD_DATASET_PATH)

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    base_name = 'dd_final_seq2seq'


    log_path = os.path.join(LOG_DIR, base_name + '.log')
    init_log(log_path)

    ppl_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=coll_dd_101,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    generate_dataloader = DataLoader(
        dataset=test_dataset,
        collate_fn=coll_dd_111,
        batch_size=DEFAULT_BATCH_SIZE, 
        shuffle=False,
    )

    summary_dir = os.path.join(EVAL_DIR, base_name, 'eval')
    writer = SummaryWriter(summary_dir)

    model_suffix = list(range(11, 20+1))
    device = gpu_device
    model_paths = [os.path.join(MODEL_DIR, base_name + '.pt_' + str(x)) for x in model_suffix]

    # for dd_final_seq2seq == 16
    model_paths = [os.path.join(FINAL_MODEL_DIR, base_name + '.pt_16')]
    
    # n_ppl = test_multi_ppl(model_paths, device, ppl_dataloader, test_ppl, base_name=base_name, writer=writer)
    test_multi_generate(model_paths, device, generate_dataloader, generate_samples, greedy_generate, full_test=True)

    
    # n_ppl = test_multi_ppl(model_paths, device, ppl_dataloader, test_ppl, base_name=base_name, writer=writer)
    # test_multi_generate(model_paths, device, generate_dataloader, generate_samples, greedy_generate)


if __name__ == '__main__':
    main()