import numpy as np
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
import argparse
import torchtext
from torchtext.data.utils import get_tokenizer
import copy
from typing import Optional, Any, Union, Callable
from torch import Tensor

from models.layers import IsoEncoderLayer

import dotenv

from datetime import datetime
import wandb
import os
#torch.manual_seed(0)

def batchify(data, bsz):
    
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class IsoEncoder(Module):
    def __init__(self, d_model, n_experts, dim_feedforward=2048, dropout=0.1, activation = "elu", 
                 use_LayerNorm = False, init_resweight = 0, resweight_trainable = True, use_ortho=True):
        super(IsoEncoder, self).__init__()
        self.self_attn = IsoEncoderLayer(d_model, n_experts,random_features=False, batch_first=False, use_ortho=use_ortho)
        #self.self_attn = MultiheadAttention(d_model, n_experts, dropout=dropout)

        # Define the Resisdual Weight for ReZero
        self.resweight = torch.nn.Parameter(torch.Tensor([init_resweight]), requires_grad = resweight_trainable)

        self.all_resweights = []
        # Implementation of Feedforward model
        self.use_LayerNorm = use_LayerNorm

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.elu #F.relu
        super(IsoEncoder, self).__setstate__(state)

    def forward(self, src, eig, src_mask=None, src_key_padding_mask=None):
        out = self.self_attn(src, self.resweight, eig)

        return out

class TransformerEncoder2(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(
            self,
            src: Tensor,
            eig,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product attention.
                Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        '''
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        '''

        #print (src.shape)

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        for mod in self.layers:
            output = mod(output, eig)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class TransformerModel(torch.nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, 
                 encoder_version = 'ReZero', use_eigen=False, use_ortho=True):
        super(TransformerModel, self).__init__()
        #from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.use_eigen = use_eigen
        self.src_mask = None
        self.pos_encoder = ConcatenatedPositionalEncoder(ninp//2, 4000) #PositionalEncoding(ninp, dropout)

        self.U = torch.nn.utils.parametrizations.orthogonal(nn.Linear(ninp, ninp))

        encoder_layers = IsoEncoder(ninp, nhead, nhid, dropout, 
            use_LayerNorm = False, init_resweight = 0, 
            resweight_trainable = True, activation='elu', use_ortho=use_ortho)
        self.transformer_encoder = TransformerEncoder2(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken, ninp//2)
        self.ninp = ninp
        self.decoder = torch.nn.Linear(ninp, ntoken)
        self._reset_parameters()
        self.init_weights()
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1) #.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) #* math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        #src = src.transpose(0,1)
        #print (src.shape)

        recon_loss = 0

        if self.use_eigen == True:
            #M = torch.einsum('mbd,mbe->bde', src,src)
            mask = (torch.triu(torch.ones(src.shape[0], src.shape[0])) == 1).transpose(0, 1).float().to(src.device)
            src1_ = torch.einsum('mbd,mn->nbd',src,mask)
            #print (src1_)
            #src1_ = torch.nan_to_num(src1_ / src1_.norm(dim=-1).unsqueeze(-1))
            #print (src1_)
            M = torch.einsum('mbd,nbe->bde', src,src1_) / self.ninp
            #M = torch.einsum('bmnde,mn->bde',M,mask)
            #M = torch.nan_to_num(M / M.norm(dim=-1).unsqueeze(-1))
            U = self.U.weight
            bias = self.U.bias
            M2 = torch.einsum('bmn,mn->bmn',M,U.transpose(0,1)) + bias 
            M2 = torch.einsum('mn,bmn->bmn',U,M2) + bias
            eigs2 = torch.tanh(torch.diagonal(M2, dim1 = -2, dim2 = -1))
            eigs2 = eigs2/eigs2.max(1).values.unsqueeze(1)

            eigs = []
            for bs in range(src.size(1)):
                eig_ = torch.diag(eigs2[bs])
                eigs.append(eig_.unsqueeze(1))
            
            eig = torch.cat(eigs, 1)

            M_ = torch.einsum('mbn,mn->bmn',eig, U) + bias
            M_ = torch.einsum('mn,bmn->bmn',U.transpose(0,1),M_) + bias
            recon_loss += nn.MSELoss(reduction='mean')(M, M_)

        else:
            #eig = F.softmax(torch.einsum('bmd,bme->bde', src, src), -1)
            eig = []
            for bs in range(src.size(1)):
                eig_ = nn.Parameter(torch.rand(self.ninp), requires_grad=True).to(src.device)
                eig_ = torch.tanh(torch.diag(eig_))
                eig_ = eig_/torch.abs(eig_).max()
                eig.append(eig_.unsqueeze(1))
            
            eig = torch.cat(eig, 1)
        #print (src.shape, eig.shape)

        #print (src.shape)
        output = self.transformer_encoder(src, eig, self.src_mask)
        output = self.decoder(output)
        #output = output.transpose(0,1)
        return output, recon_loss
    

######################################################################
# Positional Encoding

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print (x.shape, self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ConcatenatedPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, self.d_model)
        for pos in range(max_seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len].repeat(x.size(0),1,1),requires_grad=False)        
        x = torch.cat([x,pe],-1)
        return x
    
def setup_and_train(epochs, lr, emsize, nhid, nlayers, nhead, dropout, encoder_version, use_eigen=False, plt_jacobian = True, use_ortho=True):
    
    ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
    
    ######################################################################
    # Model setup

    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, 
                             dropout, encoder_version = encoder_version, use_eigen=use_eigen, use_ortho=use_ortho).to(device)
    model.to(device)
    print ("Total number of parameters={}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    ######################################################################
    # Define criterion and optimizer

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr = lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9)

    ######################################################################
    # Define the training

    def train():
        model.train() # Turn on the train mode
        total_loss = 0.
        total_recon_loss = 0.
        start_time = time.time()
        ntokens = len(TEXT.vocab.stoi)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            output, recon_loss = model(data) #.transpose(0,1)
            loss = criterion(output.view(-1, ntokens), targets)
            loss_main = loss + args.lambda_ * recon_loss
            loss_main.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            #total_recon_loss += recon_loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                cur_recon_loss = total_recon_loss/log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.0f} | '
                      'loss {:5.2f} | recon loss {:5.2f} | ppl {:6.0f}'.format(
                        epoch, batch, len(train_data) // bptt, lr,
                        elapsed * 1000 / log_interval,cur_loss, cur_recon_loss, math.exp(cur_loss)))
                if args.wandb_logging == True:
                    wandb.log({'training loss': cur_loss})
                    wandb.log({'training recon loss': cur_recon_loss})

                total_loss = 0
                total_recon_loss = 0
                start_time = time.time()

    ######################################################################
    # Define the evaluation

    def evaluate(eval_model, data_source):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(TEXT.vocab.stoi)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                output, _ = eval_model(data) #.transpose(0,1)
                output_flat = output.view(-1, ntokens)
                total_loss += data.shape[0] * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    ######################################################################
    # Train the model
    now = int(datetime.now().timestamp())
    model_checkpoint_path = os.path.join(args.model_save_path, str(now))

    try:
        os.makedirs(model_checkpoint_path)
    except:
        pass
    
    if args.wandb_logging == True:
        config = vars(args)
        config['model_name'] = 'TransJect'
        config['model_checkpoint'] = model_checkpoint_path
        config['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.login()
        wandb.init(project=args.wandb_project_name,config=config)
        artifact = wandb.Artifact('Model', type='model')
        wandb.watch(model, log_freq=100)

    best_model_params_path = os.path.join(model_checkpoint_path, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print('-' * 88)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 88)
        #scheduler.step()

        torch.save(model.state_dict(), best_model_params_path)
        
        if args.wandb_logging == True:
            wandb.log({'validation loss': val_loss})
            wandb.log({'validation perplexity': math.exp(val_loss)})
            
    epoch_start_time = time.time()
    test_loss = evaluate(model, test_data)
    print('-' * 88)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
            'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        test_loss, math.exp(test_loss)))
    if args.wandb_logging == True:
            wandb.log({'Test loss': test_loss})
            wandb.log({'Test perplexity': math.exp(test_loss)})
    print('-' * 88)

if __name__ == '__main__':
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

    parser.add_argument('--dataset', type=str, required=False, default='wikitext',
                        help='Dataset - wikitext or ptb')
    parser.add_argument('--use_rezero', action='store_true',
                        help='use rezero')
    parser.add_argument('--use_ortho', action='store_true',
                        help='use orthogonal parametrization')
    parser.add_argument('--use_eigen', action='store_true',
                        help='use eigen approximation')
    
    parser.add_argument('--max_text_len', type=int, default=35, required=False,
                        help='maximum length of text')
    parser.add_argument('--n_layers', type=int, default=6, required=False,
                        help='maximum length of text')
    parser.add_argument('--d_model', type=int, default=128, required=False,
                        help='hidden size of the model')
    parser.add_argument('--n_head', type=int, default=4, required=False,
                        help='number of attention heads')
    parser.add_argument('--epochs', type=int, default=50, required=False,
                        help='number of epochs')
    parser.add_argument('--max_grad_norm', type=int, default=1, required=False,
                        help='max grad norm')
    parser.add_argument('--lr', type=float, default=.01, required=False,
                        help='learning rate')
    parser.add_argument('--lambda_', type=float, default=0.1, required=False,
                        help='L2 regularization weight')

    parser.add_argument('--classification_type', type=str, default='multiclass', required=False,
                        help='Type of classification: binary, multiclass')
    
    parser.add_argument('--train_batch_size', type=int, default=16, required=False,
                        help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, required=False,
                        help='eval batch size')

    parser.add_argument('--model_save_path', type=str, default='../models/wikitext2/', required=False,
                        help='model save path')

    parser.add_argument('--wandb_logging', action='store_true',
                        help='wandb logging needed')
    parser.add_argument('--wandb_project_name', type=str, default='WikiText 2 Language Modeling', required=False,
                        help='wandb project name')

    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='seed')
    
    args = parser.parse_args()
    print (args)

    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
    
    if args.dataset == 'wikitext':
        train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    elif args.dataset == 'ptb':
        train_txt, val_txt, test_txt = torchtext.datasets.PennTreebank.splits(TEXT)
    else:
        raise ValueError("dataset name must be either wikitext or ptb")
    
    TEXT.build_vocab(train_txt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bptt = args.max_text_len
    batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    train_data = batchify(train_txt, batch_size)
    val_data = batchify(val_txt, eval_batch_size)
    test_data = batchify(test_txt, eval_batch_size)

    encoder_version = 'ReZero'      # architecture: 'ReZero', 'pre', or 'post' (vanilla)
    nlayers = args.n_layers                  # the number of Layers
    lr = args.lr                        # Initial learning rate
    epochs = args.epochs                      # The number of epochs
    emsize = args.d_model                    # embedding dimension
    nhid = args.d_model                        # the dimension of the feedforward network model
    nhead = args.n_head                       # the number of heads in self attention
    dropout = 0.1                   # the dropout value

    setup_and_train(epochs, lr, emsize, nhid, nlayers, nhead, dropout, encoder_version, args.use_eigen, plt_jacobian = False, use_ortho=args.use_ortho)
    