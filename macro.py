import os

TRAIN_STAT = 0
EVAL_STAT = 1

WORK_DIR = '/home/kananos/dialog'

GEN_DIR = os.path.join(WORK_DIR, 'generate')
EVAL_DIR = os.path.join(WORK_DIR, 'eval')
# DATA_DIR = os.path.join(WORK_DIR, 'data')
LOG_DIR = os.path.join(WORK_DIR, 'log')

FINAL_GEN_FULL_DIR = os.path.join(WORK_DIR, 'final-generate-full')
FINAL_GEN_DIR = os.path.join(WORK_DIR, 'final-generate')

MODEL_DIR = os.path.join(WORK_DIR, 'model')
FINAL_MODEL_DIR = os.path.join(WORK_DIR, 'final-model')
DATASET_DIR = os.path.join(WORK_DIR, 'dataset')

TWITTER_DATASET_DIR = os.path.join(DATASET_DIR, 'twitter')
DD_DATASET_DIR = os.path.join(DATASET_DIR, 'dailydialog')
PC_DATASET_DIR = os.path.join(DATASET_DIR, 'personachat')

TOK_PATH = '/home/kananos/PRE/gpt2tok.bin'
LEGACY_TOK_PATH = '/home/kananos/PRE/gpt2tok_legacy.bin'
PRETRAINED_PATH = '/home/kananos/PRE/gpt2lm_additional.pt'
NORMAL_PRETRAINED_PATH = '/home/kananos/PRE/gpt2_additional.pt'
PRETRAINED_VOCAB_SIZE = 50260
LEGACY_VOCAB_SIZE = 50260 - 3

TEST_PC_DATASET_PATH = os.path.join(PC_DATASET_DIR, 'test.dataset')
TRAIN_PC_DATASET_PATH = os.path.join(PC_DATASET_DIR, 'train.dataset')

TRAIN_TWITTER_DATASET_PATH = os.path.join(TWITTER_DATASET_DIR, 'train.dataset')
TEST_TWITTER_DATASET_PATH = os.path.join(TWITTER_DATASET_DIR, 'test.dataset')

TRAIN_DD_DATASET_PATH = os.path.join(DD_DATASET_DIR, 'train.dataset')
TEST_DD_DATASET_PATH = os.path.join(DD_DATASET_DIR, 'test.dataset')

# NO USED
# TEST_NSP_DATASET_NAME = 'test_nsp.dataset'
# TRAIN_NSP_DATASET_NAME = 'train_nsp.dataset'
# TEST_NSP_DATASET_PATH = os.path.join(DATASET_DIR, TEST_NSP_DATASET_NAME)
# TRAIN_NSP_DATASET_PATH = os.path.join(DATASET_DIR, TRAIN_NSP_DATASET_NAME)

DEFAULT_BATCH_SIZE = 16
DEFAULT_SMALL_BATCH_SIZE = 8
DEFAULT_TINY_BATCH_SIZE = 4
DEFAULT_LR = 1e-5
DEFAULT_DD_LR = 1e-5 * 5
DEFAULT_SEQ_LR = 1e-5 * 10
DEFAULT_EPOCHS = 5
DEFAULT_SEQ_EPOCHS = 10
DEFAULT_D_MODEL = 768
DEFAULT_N_HEAD = 12

DEFAULT_MU = 4

MAX_GEN_LEN = 40


FINA_MANUAL_PRESET_FILES = [
    'gpt2.pt_6_generate', 
    'final.pt_5_generate', 
    'final_seq2seq.pt_2_generate', 
    'seq2seq.pt_8_generate',
    'gpt2_pred.pt_6_generate',
    'gpt2_sampling.pt_3_generate',
    'seq2seq_pred.pt_8_generate',
    'seq2seq_sampling.pt_10_generate'
]
