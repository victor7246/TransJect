import os
import shutil

import tensorflow as tf
import tokenizers
import pandas as pd

class CustomTokenizer:
    def __init__(self, lower=False, split=' ', char_level=False):
        super().__init__() 
        self.lower = lower
        self.split = split
        self.char_level = char_level
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=lower, char_level=char_level, split=split, filters='')
        
    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        
    def encode(self, text, padding=None, max_length=512):
        if padding is None:
            seq = self.tokenizer.texts_to_sequences([text])
        else:
            seq = tf.keras.utils.pad_sequences(sequences=self.tokenizer.texts_to_sequences([text]), \
                                           maxlen=max_length, padding='post')
        return seq[0]

def custom_wp_tokenizer(corpus,text_filepath,tokenizer_save_path,vocab_size=10000,min_frequency=3):
    if type(corpus[0]) == list:
        corpus = [" ".join(i) for i in corpus]

    try:
        os.makedirs(text_filepath)
    except OSError:
        pass

    tokenizer = tokenizers.BertWordPieceTokenizer(
            unk_token='[UNK]',
            sep_token='[SEP]',
            cls_token='[CLS]',
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True,
            lowercase=True,
            wordpieces_prefix='##'
        )#SentencePieceBPETokenizer()

    df = pd.DataFrame()
    df['text'] = corpus
    df.to_csv(os.path.join(text_filepath,'file.txt'),header=False,index=False)

    try:
        os.makedirs(tokenizer_save_path)
    except OSError:
        pass

    tokenizer.train(files=os.path.join(text_filepath,'file.txt'), vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]'])

    #tokenizer.save(directory=tokenizer_save_path,name='wpe')
    tokenizer.save_model(tokenizer_save_path, 'wpe')
    
    shutil.move(os.path.join(tokenizer_save_path,'wpe-vocab.txt'), os.path.join(tokenizer_save_path,'vocab.txt'))

    os.remove(os.path.join(text_filepath,'file.txt'))

def custom_bpe_tokenizer(corpus,text_filepath,tokenizer_save_path,vocab_size=10000,min_frequency=3):
    if type(corpus[0]) == list:
        corpus = [" ".join(i) for i in corpus]

    try:
        os.makedirs(text_filepath)
    except OSError:
        pass

    tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=None,
            merges_file=None,
        )#SentencePieceBPETokenizer()

    df = pd.DataFrame()
    df['text'] = corpus
    df.to_csv(os.path.join(text_filepath,'file.txt'),header=False,index=False)

    try:
        os.makedirs(tokenizer_save_path)
    except OSError:
        pass

    tokenizer.train(files=os.path.join(text_filepath,'file.txt'), vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]'])

    tokenizer.save(directory=tokenizer_save_path,name='bpe')

    shutil.move(os.path.join(tokenizer_save_path,'bpe-vocab.json'), os.path.join(tokenizer_save_path,'vocab.json'))
    shutil.move(os.path.join(tokenizer_save_path,'bpe-merges.txt'), os.path.join(tokenizer_save_path,'merges.txt'))

    os.remove(os.path.join(text_filepath,'file.txt'))