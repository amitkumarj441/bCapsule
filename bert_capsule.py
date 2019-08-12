#Load modules
import sys, gc, os, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numba import cuda


package_dir = "bCapsule/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.append(package_dir)

import torch.utils.data
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig

warnings.filterwarnings(action='once')
device = torch.device('cuda')

# Parameters/Path Settings
MAX_SEQUENCE_LENGTH = 220
SEED = 42
BATCH_SIZE = 32
BERT_MODEL_PATH = 'bCapsule/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
LARGE_BERT_MODEL_PATH = 'bCapsule/bert-pretrained-models/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/'
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


# Pretrained BERT models - Google's pretrained BERT model
BERT_SMALL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
BERT_LARGE_PATH = '../input/bert-pretrained-models/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/'

# FastText fine-tuned BERT models
FT_BERT_SMALL_MODEL_PATH = 'fasttext_pytorch.bin'
FT_BERT_LARGE_MODEL_PATH = 'pytorch_model.bin' # 1 epoch for testing - or 5/10 epoch
FT_BERT_SMALL_JSON_PATH = 'bert_config.json'
FT_BERT_LARGE_JSON_PATH = 'config.json' # 1 epoch for testing - or 5/10 epoch
NUM_BERT_MODELS = 1 #2
INFER_BATCH_SIZE = 64

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

test_preds = np.zeros((test_df.shape[0],NUM_BERT_MODELS))
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



print("Predicting BERT large model......")

# Prepare data
bert_config = BertConfig(FT_BERT_LARGE_JSON_PATH)
tokenizer = BertTokenizer.from_pretrained(BERT_LARGE_PATH, cache_dir=None,do_lower_case=True)
X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))      

# Load fine-tuned BERT model
gc.collect()
model = BertForSequenceClassification(bert_config, num_labels=1)
model.load_state_dict(torch.load(FT_BERT_LARGE_MODEL_PATH))
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

# Predicting
model_preds = np.zeros((len(X_test)))
test_loader = torch.utils.data.DataLoader(test, batch_size=INFER_BATCH_SIZE, shuffle=False)
tk0 = tqdm(test_loader)
for i, (x_batch,) in enumerate(tk0):
        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        model_preds[i * INFER_BATCH_SIZE:(i + 1) * INFER_BATCH_SIZE] = pred[:, 0].detach().cpu().squeeze().numpy()

test_preds[:,0] = torch.sigmoid(torch.tensor(model_preds)).numpy().ravel()
del model
gc.collect()

cuda.select_device(0)
cuda.close()
