from locale import normalize
import os
import sys
import ast
import math
import random
import re
import argparse
import copy
from copy import deepcopy as cp
from collections import OrderedDict
import dotenv
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.autograd.function import InplaceFunction
from torch.autograd import Variable

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, BertTokenizer, AutoModelForSequenceClassification, AutoModel

import wandb

from tqdm import tqdm

from data.data_utils import remove_emojis, remove_html, remove_email, clean_tweets
from data.custom_tokenizers import custom_wp_tokenizer
from data.datasets import ClassificationDataset
from models.models import IsoFormerForClassificationPL

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == '__main__':
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

    parser.add_argument('--train_data_file', type=str, default='../data/imdb_data_train.csv', required=False,
                        help='train data')
    parser.add_argument('--test_data_file', type=str, default='../data/imdb_data_test.csv', required=False,
                        help='test data')

    parser.add_argument('--use_eigen', action='store_true',
                        help='use eigenvalue decomposition')
    parser.add_argument('--use_random_features', type=str, default='true', required=False, 
                        help = 'Whether to use random features')

    parser.add_argument('--classification_type', type=str, default='binary', required=False,
                        help='Type of classification: binary, multiclass')
    parser.add_argument('--pooling', type=str, default='mean', required=False,
                        help='Type of pooling; max or mean')

    parser.add_argument('--max_text_len', type=int, default=512, required=False,
                        help='maximum length of text')
    parser.add_argument('--n_layers', type=int, default=6, required=False,
                        help='maximum length of text')
    parser.add_argument('--d_model', type=int, default=512, required=False,
                        help='hidden size of the model')
    parser.add_argument('--n_experts', type=int, default=4, required=False,
                        help='number of attention heads')
    parser.add_argument('--lambda_', type=float, default=0.001, required=False,
                        help='L2 regularization weight')
    parser.add_argument('--data_sample', type=float, default=1.0, required=False,
                        help='Training data sample')

    parser.add_argument('--epochs', type=int, default=30, required=False,
                        help='number of epochs')
    parser.add_argument('--max_grad_norm', type=int, default=5, required=False,
                        help='max grad norm')
    parser.add_argument('--lr', type=float, default=0.0005, required=False,
                        help='learning rate')
    parser.add_argument('--max_lr', type=float, default=0.5, required=False,
                        help='maximum learning rate')
    parser.add_argument('--n_warmup', type=int, default=4000, required=False,
                        help='warmup steps')
    parser.add_argument('--early_stopping_rounds', type=int, default=4, required=False,
                        help='number of epochs for early stopping')

    parser.add_argument('--train_batch_size', type=int, default=16, required=False,
                        help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, required=False,
                        help='eval batch size')

    parser.add_argument('--model_save_path', type=str, default='../models/imdb/', required=False,
                        help='model save path')

    parser.add_argument('--wandb_logging', action='store_true',
                        help='wandb logging needed')
    parser.add_argument('--wandb_project_name', type=str, default='IMDb Classification', required=False,
                        help='wandb project name')

    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='seed')

    args = parser.parse_args()
    print (args)

    try:
        assert args.classification_type in ['binary', 'multiclass']
    except:
        raise ValueError("classification_type should be in ['binary','multiclass']")

    train_df = pd.read_csv(args.train_data_file).sample(frac=1).sample(frac=args.data_sample).reset_index(drop=True)
    test_df = pd.read_csv(args.test_data_file).sample(frac=1).reset_index(drop=True)

    print (train_df.sentiment.value_counts(normalize=True), test_df.sentiment.value_counts(normalize=True))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
        
    ll = LabelEncoder()
    train_df['sentiment_id'] = ll.fit_transform(train_df.sentiment.values.reshape(-1,1))
    test_df['sentiment_id'] = ll.transform(test_df.sentiment.values.reshape(-1,1))

    print (ll.classes_)

    train_dataset = ClassificationDataset(texts=train_df.text.values.tolist(), \
            out=train_df.sentiment_id.values[:,np.newaxis], src_tokenizer=tokenizer, \
                MAX_LEN=args.max_text_len)
    val_dataset = ClassificationDataset(texts=test_df.text.values.tolist(), \
            out=test_df.sentiment_id.values[:,np.newaxis], src_tokenizer=tokenizer, \
                MAX_LEN=args.max_text_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = IsoFormerForClassificationPL(args, vocab_size, args.d_model, args.max_text_len, args.n_layers, args.n_experts, 1, args.use_eigen, \
        random_features=ast.literal_eval(args.use_random_features.capitalize()), pooling=args.pooling)

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    print ("Total number of parameters={}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    now = int(datetime.now().timestamp())

    model_checkpoint_path = os.path.join(args.model_save_path, str(now))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",dirpath=model_checkpoint_path)
    early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.005, patience=args.early_stopping_rounds)
    
    try:
        os.makedirs(model_checkpoint_path)
    except:
        pass

    if args.wandb_logging == True:
        config = vars(args)
        config['model_name'] = model.__class__.__name__
        config['model_checkpoint'] = model_checkpoint_path
        config['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.login()
        wandb.init(project=args.wandb_project_name,config=config)
        artifact = wandb.Artifact('Model', type='model')
        wandb.watch(model, log_freq=100)

        trainer = Trainer(logger=WandbLogger(), accelerator=accelerator, \
            callbacks=[checkpoint_callback,early_stopping], max_epochs=args.epochs, gradient_clip_val=args.max_grad_norm)
    else:
        trainer = Trainer(accelerator=accelerator, \
            callbacks=[checkpoint_callback,early_stopping], max_epochs=args.epochs, gradient_clip_val=args.max_grad_norm)

    trainer.fit(model, train_loader, val_loader)
    