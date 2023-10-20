import torch
import numpy as np
from pytorch_lightning import LightningDataModule
import pickle

class AANDataset:
    def __init__(self, file):
        self.data = pickle.load(open(file, 'rb'))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        src = torch.LongTensor(self.data[item]['input_ids_0'].tolist() +self.data[item]['input_ids_1'].tolist())
        out_ = torch.FloatTensor([self.data[item]['label']])

        return {"input_ids": src, "out": out_}
    
class CustomDataset:
    def __init__(self, inp, out):
        
        self.inp = inp
        self.out = out
        
    def __len__(self):
        return len(self.inp)

    def __getitem__(self, item):
        src = torch.LongTensor(self.inp[item])
        
        if len(np.unique(np.asarray(self.out))) in [1,2]:
            out_ = torch.FloatTensor([self.out[item]])
        else:
            out_ = torch.LongTensor([self.out[item]])
        
        return {"input_ids": src, "out": out_}

class CustomDataset:
    def __init__(self, inp, out):
        
        self.inp = inp
        self.out = out
        
    def __len__(self):
        return len(self.inp)

    def __getitem__(self, item):
        src = torch.LongTensor(self.inp[item])
        
        if len(np.unique(np.asarray(self.out))) in [1,2]:
            out_ = torch.FloatTensor([self.out[item]])
        else:
            out_ = torch.LongTensor([self.out[item]])
        
        return {"input_ids": src, "out": out_}

class ClassificationDataset:
    def __init__(self, texts, out, src_tokenizer, MAX_LEN):
        
        self.texts = texts
        self.out = out
        self.src_tokenizer = src_tokenizer
        self.MAX_LEN = MAX_LEN
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        
        src = torch.LongTensor(self.src_tokenizer.encode(text, padding='max_length', max_length=self.MAX_LEN)[:self.MAX_LEN])
        if len(np.unique(np.asarray(self.out))) in [1,2]:
            out_ = torch.FloatTensor(self.out[item])
        else:
            out_ = torch.LongTensor(self.out[item])
        
        return {"input_ids": src, "out": out_}

class SimilarityDataset:
    def __init__(self, data, src_tokenizer, MAX_LEN):
        
        self.texts1 = data.text1
        self.texts2 = data.text2
        self.out = data.label.values[:,np.newaxis]
        self.src_tokenizer = src_tokenizer
        self.MAX_LEN = MAX_LEN
        
    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, item):
        text1 = self.texts1[item]
        text2 = self.texts2[item]

        src1 = torch.LongTensor(self.src_tokenizer.encode(text1, padding='max_length', max_length=self.MAX_LEN)[:self.MAX_LEN])
        src2 = torch.LongTensor(self.src_tokenizer.encode(text2, padding='max_length', max_length=self.MAX_LEN)[:self.MAX_LEN])

        if len(np.unique(np.asarray(self.out))) in [1,2]:
            out_ = torch.FloatTensor(self.out[item])
        else:
            out_ = torch.LongTensor(self.out[item])
        
        return {"input_ids1": src1, "input_ids2" : src2, "out": out_}

class GenerationDataset:
    def __init__(self, encoder_texts, decoder_texts, src_tokenizer, trg_tokenizer, encoder_MAX_LEN, decoder_MAX_LEN):
        
        self.encoder_texts = encoder_texts
        self.decoder_texts = decoder_texts
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.encoder_MAX_LEN = encoder_MAX_LEN
        self.decoder_MAX_LEN = decoder_MAX_LEN
        
    def __len__(self):
        return len(self.encoder_texts)

    def __getitem__(self, item):
        text1 = self.encoder_texts[item]
        text2 = self.decoder_texts[item]
        
        src = torch.LongTensor(self.src_tokenizer.encode(text1, padding='max_length', max_length=self.encoder_MAX_LEN)[:self.encoder_MAX_LEN])
        trg = torch.LongTensor(self.trg_tokenizer.encode(text2, padding='max_length', max_length=self.decoder_MAX_LEN)[:self.decoder_MAX_LEN])
        
        return {"input_ids": src, "output_ids": trg}                

class GLUEDataset:
    def __init__(self, texts1, texts2, out, tokenizer, MAX_LEN, label_dtype):
        
        self.texts1 = texts1
        self.texts2 = texts2
        self.out = out
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN
        self.data_type = label_dtype

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, item):
        text1 = self.texts1[item]
        text2 = self.texts2[item]
        
        text = text1 + "[SEP]" + text2

        src = torch.LongTensor(self.tokenizer.encode(text, padding='max_length', max_length=self.MAX_LEN)[:self.MAX_LEN])

        if self.data_type == 'float':
            out_ = torch.FloatTensor(self.out[item])
        else:
            out_ = torch.LongTensor(self.out[item])

        return {"input_ids": src, "out": out_}   