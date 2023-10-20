import torch
import torch.nn as nn 
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from smelu import SmeLU
from transformers import AutoModel

class MultiHeadAttentionNew(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1, use_ortho=False):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        if use_ortho == True:
            self.q_linear = torch.nn.utils.parametrizations.orthogonal(self.q_linear)
            self.v_linear = torch.nn.utils.parametrizations.orthogonal(self.v_linear)
            self.k_linear = torch.nn.utils.parametrizations.orthogonal(self.k_linear)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None, debug=False):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
        
        if debug == True:
            return output, scores
        else:
            return output

        return output

class FeedForwardNew(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1, use_ortho=False):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayerNew(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, use_rezero=True, use_ortho=False, use_random=False):
        super().__init__()

        self.use_random = use_random
        self.n_heads = heads
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttentionNew(heads, d_model, dropout=dropout, use_ortho=use_ortho)
        self.ff = FeedForwardNew(d_model, dropout=dropout, use_ortho=use_ortho)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        if use_rezero == True:
            self.resweight = nn.Parameter(torch.Tensor([0]))
        else:
            self.resweight = 1
        
        self.all_resweights = []

    def forward(self, x, mask, debug=False):
        x2 = self.norm_1(x)
        if debug == True:
            x2, attn_score = self.attn(x2,x2,x2,mask, debug=True)
            if self.use_random == False:
                x = x + self.dropout_1(x2*self.resweight)
            else:
                x2 = self.dropout_1(x2 * self.resweight)
                y = torch.zeros_like(x)
                for n in range(self.n_heads):
                    rand_key = torch.rand(1)[0]
                    if rand_key <= 0.25:
                        y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] + x2[:,:,n::self.n_heads]
                    elif rand_key <= 0.5:
                        y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] - x2[:,:,n::self.n_heads]
                    elif rand_key <= 0.75:
                        y[:,:,n::self.n_heads] = 4 * x2[:,:,n::self.n_heads]
                    else:
                        y[:,:,n::self.n_heads] = x[:,:,n::self.n_heads]
        else:
            if self.use_random == False:
                x = x + self.dropout_1(self.attn(x2,x2,x2,mask)*self.resweight)
            else:
                x2 = self.dropout_1(self.attn(x2,x2,x2,mask)*self.resweight)
                y = torch.zeros_like(x)
                for n in range(self.n_heads):
                    rand_key = torch.rand(1)[0]
                    if rand_key <= 0.25:
                        y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] + x2[:,:,n::self.n_heads]
                    elif rand_key <= 0.5:
                        y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] - x2[:,:,n::self.n_heads]
                    elif rand_key <= 0.75:
                        y[:,:,n::self.n_heads] = 4 * x2[:,:,n::self.n_heads]
                    else:
                        y[:,:,n::self.n_heads] = x[:,:,n::self.n_heads]
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2)*self.resweight)

        if debug == True:
            return x, attn_score

        return x

class EncoderNew(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, N, heads, dropout, use_rezero=True, use_ortho=False, use_random=False):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayerNew(d_model, heads, dropout,use_rezero=use_rezero, use_ortho=use_ortho, use_random=use_random), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask, return_all_hidden_states=False, debug=False):
        x = self.embed(src)
        x = self.pe(x)
        
        all_hidden_states = []
        all_attn_scores = []
        if return_all_hidden_states == True:
            all_hidden_states.append(x)
        for i in range(self.N):
            if debug == True:
                x, attn_score = self.layers[i](x, mask, debug=True)
                all_attn_scores.append(attn_score.unsqueeze(0))
            else:
                x = self.layers[i](x, mask)

            if return_all_hidden_states == True:
                all_hidden_states.append(x)

        if return_all_hidden_states:
            if debug == True:
                return self.norm(x), all_hidden_states, torch.cat(all_attn_scores,0)
            else:
                return self.norm(x), all_hidden_states
        else:
            if debug == True:
                return self.norm(x), torch.cat(all_attn_scores, 0)
            else:
                return self.norm(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    #mask = None

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1, use_ortho=False):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        if use_ortho == True:
            self.q_linear = torch.nn.utils.parametrizations.orthogonal(self.q_linear)
            self.v_linear = torch.nn.utils.parametrizations.orthogonal(self.v_linear)
            self.k_linear = torch.nn.utils.parametrizations.orthogonal(self.k_linear)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None, debug=False):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
        
        if debug == True:
            return output, scores

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1, use_ortho=False):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class FeedForwardWithoutDropout(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

'''
class StochasticFF(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        U1 = nn.Softmax(-1)(self.linear_1.weight.transpose(0,1))
        bias1 = self.linear_1.bias
        U2 = nn.Softmax(-1)(self.linear_2.weight.transpose(0,1))
        bias2 = self.linear_2.bias

        #print (x.shape, U1.shape, bias1.shape)
        x = F.elu(torch.einsum('bmd,de->bme',x,U1) + bias1.unsqueeze(0).unsqueeze(0))
        #print (x.shape)
        x = self.dropout(x)
        x = torch.einsum('bmd,de->bme',x,U2) + bias2.unsqueeze(0).unsqueeze(0)

        return x
'''

class StochasticFF(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))/torch.linalg.norm(self.linear_1.weight, 2)
        #x = self.dropout(SmeLU()(self.linear_1(x)))/torch.linalg.norm(self.linear_1.weight, 2)
        #x = self.dropout(F.relu(self.linear_1(x)))/torch.linalg.norm(self.linear_1.weight, 2)
        #x = self.dropout(SmeLU()(self.linear_1(x)))/torch.linalg.norm(self.linear_1.weight, 2)
        x = F.relu(self.linear_2(x)/torch.linalg.norm(self.linear_2.weight, 2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    #return nn.ModuleList([module for i in range(N)])
    #return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    #return nn.ModuleList([module for i in range(N)])

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, use_rezero=True, use_ortho=False, use_random=False):
        super().__init__()

        self.use_random = False
        self.n_heads = heads
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout, use_ortho=False)
        self.ff = FeedForward(d_model, dropout=dropout, use_ortho=False)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        if use_rezero == True:
            self.resweight = nn.Parameter(torch.Tensor([0]))
        else:
            self.resweight = 1
        
    def forward(self, x, mask, debug=False):
        x2 = self.norm_1(x)
        if debug == True:
            x2, attn_score = self.attn(x2,x2,x2,mask, debug=True)
            if self.use_random == False:
                x = x + self.dropout_1(x2*self.resweight)
            else:
                x2 = self.dropout_1(x2 * self.resweight)
                y = torch.zeros_like(x)
                for n in range(self.n_heads):
                    rand_key = torch.rand(1)[0]
                    if rand_key <= 0.25:
                        y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] + x2[:,:,n::self.n_heads]
                    elif rand_key <= 0.5:
                        y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] - x2[:,:,n::self.n_heads]
                    elif rand_key <= 0.75:
                        y[:,:,n::self.n_heads] = 4 * x2[:,:,n::self.n_heads]
                    else:
                        y[:,:,n::self.n_heads] = x[:,:,n::self.n_heads]
        else:
            if self.use_random == False:
                x = x + self.dropout_1(self.attn(x2,x2,x2,mask)*self.resweight)
            else:
                x2 = self.dropout_1(self.attn(x2,x2,x2,mask)*self.resweight)
                y = torch.zeros_like(x)
                for n in range(self.n_heads):
                    rand_key = torch.rand(1)[0]
                    if rand_key <= 0.25:
                        y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] + x2[:,:,n::self.n_heads]
                    elif rand_key <= 0.5:
                        y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] - x2[:,:,n::self.n_heads]
                    elif rand_key <= 0.75:
                        y[:,:,n::self.n_heads] = 4 * x2[:,:,n::self.n_heads]
                    else:
                        y[:,:,n::self.n_heads] = x[:,:,n::self.n_heads]
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2)*self.resweight)

        if debug == True:
            return x, attn_score

        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, use_rezero=True):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)

        self.ff = FeedForward(d_model, dropout=dropout)

        if use_rezero == True:
            self.resweight = nn.Parameter(torch.Tensor([0]))
        else:
            self.resweight = 1

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask)) * self.resweight
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask)) * self.resweight
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2)) * self.resweight
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, N, heads, dropout, use_rezero=True, use_ortho=False, use_random=False):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout,use_rezero=use_rezero, use_ortho=use_ortho, use_random=use_random), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask, return_all_hidden_states=False, debug=False):
        x = self.embed(src)
        x = self.pe(x)
        
        all_hidden_states = []
        all_attn_scores = []
        if return_all_hidden_states == True:
            all_hidden_states.append(x)
        for i in range(self.N):
            if debug == True:
                x, attn_score = self.layers[i](x, mask, debug=True)
                all_attn_scores.append(attn_score.unsqueeze(0))
            else:
                x = self.layers[i](x, mask)

            if return_all_hidden_states == True:
                all_hidden_states.append(x)

        if return_all_hidden_states:
            if debug == True:
                return self.norm(x), all_hidden_states, torch.cat(all_attn_scores,0)
            else:
                return self.norm(x), all_hidden_states
        else:
            if debug == True:
                return self.norm(x), torch.cat(all_attn_scores, 0)
            else:
                return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, use_rezero=True):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout, use_rezero=use_rezero), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

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

class OrthogonalFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = torch.nn.utils.parametrizations.orthogonal(nn.Linear(d_model, d_model))
        self.linear_2 = torch.nn.utils.parametrizations.orthogonal(nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dropout(F.elu(self.linear_1(x)))
        x = F.elu(self.linear_2(x))
        return x
    
class IsoAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model = d_model
        
        self.U = torch.nn.utils.parametrizations.orthogonal(nn.Linear(d_model, d_model))
        self.V = torch.nn.utils.parametrizations.orthogonal(nn.Linear(d_model, d_model))

    def forward(self, x, eig):
        eig_ = self.V(eig)
        y = F.elu(torch.matmul(self.U(x), eig_)) #SmeLU()(torch.matmul(self.U(x), eig_)) #F.relu(torch.matmul(self.U(x), eig_))
        return y
    
class IsoEncoderSubLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, random_features=True):
        super().__init__()
        
        self.d_model = d_model
        self.attn = IsoAttention(d_model)
        self.n_heads = n_heads
        self.random_features = random_features

    def forward(self, x, resweight, eig):
        x2 = self.attn(x, eig)
        x2 = x2 * resweight
        
        if self.random_features == True:
            y = torch.zeros_like(x)
            for n in range(self.n_heads):
                rand_key = torch.rand(1)[0]
                if rand_key <= 0.25:
                    y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] + x2[:,:,n::self.n_heads]
                elif rand_key <= 0.5:
                    y[:,:,n::self.n_heads] = 1.5 * x[:,:,n::self.n_heads] - x2[:,:,n::self.n_heads]
                elif rand_key <= 0.75:
                    y[:,:,n::self.n_heads] = 4 * x2[:,:,n::self.n_heads]
                else:
                    y[:,:,n::self.n_heads] = x[:,:,n::self.n_heads]
        else:
            y = x + x2

        return y
    
class IsoEncoderLayer(nn.Module):
    def __init__(self, d_model, n_experts, random_features=True):
        super().__init__()
        
        assert d_model % n_experts == 0

        self.n_experts = n_experts
        self.attn_combine = nn.Linear(d_model, 1)
        
        self.sublayers = nn.ModuleList([IsoEncoderSubLayer(d_model,random_features=random_features) for i in range(n_experts)])
        self.ff = OrthogonalFeedForward(d_model)  #FeedForward(d_model, dropout=0.1) #StochasticFF(d_model, dropout=0.1) #OrthogonalFeedForward(d_model) #FeedForward(d_model, dropout=0.1) #StochasticFF(d_model, dropout=0.1) #FeedForward(d_model, dropout=0.1) 
        
    def forward(self, x, resweight, eig, output_expert_weights=False):
        if self.n_experts > 1:
            out = torch.cat([self.sublayers[i](x, resweight, eig).unsqueeze(1) for i in range(self.n_experts)], 1)
            #outs = [out]*self.n_experts
            bs = out.size()[0]
            all_expert_hidden_states = out #out.mean(-1)
            #out = out.contiguous().view(bs, self.n_experts, -1)

            expert_weights = F.softmax(self.attn_combine(out.max(2).values), dim=1).squeeze(-1)
            out = torch.einsum('be,bemn->bmn', expert_weights, out)
            #out = torch.bmm(out.transpose(-1,1), expert_weights).squeeze(-1).contiguous().view(bs, x.size()[1], -1)
        else:
            out = self.sublayers[0](x, resweight, eig)
            all_expert_hidden_states = out.unsqueeze(0) #out.unsqueeze(0).mean(-1)

        out = out + resweight * self.ff(out)
        #print (cosine_similarity(out.detach().cpu().numpy()[:,:,0]))
        #out = self.ff(out)
        #print (cosine_similarity(out.detach().cpu().numpy()[:,:,0]))

        if output_expert_weights == True:
            return out, expert_weights, all_expert_hidden_states
        else:
            return out
        
class IsoEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, N, n_experts, use_eigen=False, random_features=True, use_bert=False, use_embedder=True):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.use_eigen = use_eigen
        self.use_embedder = use_embedder
        if use_embedder == True:
            if use_bert == True:
                bert = AutoModel.from_pretrained("bert-base-uncased")
                self.embed = bert.embeddings.word_embeddings
                for param in self.embed.parameters():
                    param.requires_grad = False
                self.pe = ConcatenatedPositionalEncoder(self.embed.embedding_dim, max_len)
                assert d_model == self.embed.embedding_dim * 2
            else:
                self.embed = Embedder(vocab_size, d_model//2)
                self.pe = ConcatenatedPositionalEncoder(d_model//2, max_len)
        self.layers = nn.ModuleList([IsoEncoderLayer(d_model, n_experts,random_features=random_features) for i in range(N)]) #nn.ModuleList([IsoEncoderLayer(d_model, n_experts) for i in range(N)])
        #self.resweight = nn.Parameter(torch.Tensor([0.1]*N), requires_grad=True)
        # initialized as 0 as per ReZero paper
        self.resweight = nn.Parameter(torch.Tensor([0]*N), requires_grad=True)
        self.U = torch.nn.utils.parametrizations.orthogonal(nn.Linear(d_model, d_model))

        self.all_resweights = []

    def forward(self, src, output_expert_weights=False, return_all_hidden_states=False):
        if self.use_embedder == True:
            x = self.embed(src)
            x = self.pe(x)
        else:
            x = src

        #print ("Embedding Layer")
        #print (euclidean_distances(x.detach().cpu().numpy()[0]))

        all_expert_weights = []
        all_hidden_states = []
        all_expert_hidden_states = []

        if return_all_hidden_states == True:
            all_hidden_states.append(x)
        recon_loss = 0

        if self.use_eigen == True:
            M = torch.einsum('bmd,bme->bde', x,x)
            U = self.U.weight
            bias = self.U.bias
            M2 = torch.einsum('bmn,mn->bmn',M,U.transpose(0,1)) + bias 
            M2 = torch.einsum('mn,bmn->bmn',U,M2) + bias
            #eigs2 = F.relu(torch.diagonal(M2, dim1 = -2, dim2 = -1))
            #eigs2 = F.relu(torch.diagonal(M2, dim1 = -2, dim2 = -1))
            eigs2 = torch.tanh(torch.diagonal(M2, dim1 = -2, dim2 = -1))
            #eigs2 = nn.Softmax(-1)(eigs2)
            #eigs2 = torch.div(eigs2, eigs2.max(1).values)
            eigs2 = eigs2/eigs2.max(1).values.unsqueeze(1)

            eigs = []
            for bs in range(x.size(0)):
                eig_ = torch.diag(eigs2[bs])
                eigs.append(eig_.unsqueeze(0))
            
            eigs = torch.cat(eigs, 0)

            M_ = torch.einsum('bmn,mn->bmn',eigs, U) + bias
            M_ = torch.einsum('mn,bmn->bmn',U.transpose(0,1),M_) + bias
            recon_loss += nn.MSELoss(reduction='mean')(M, M_)
        else:
            eigs = []
            for bs in range(x.size(0)):
                eig = nn.Parameter(torch.rand(self.d_model), requires_grad=True).to(x.device)
                #eig = F.relu(nn.Parameter(torch.rand(self.d_model), requires_grad=True).to(x.device))
                #eig = SmeLU()(nn.Parameter(torch.rand(self.d_model), requires_grad=True).to(x.device))
                #eig = F.relu(torch.diag(eig))
                eig = torch.tanh(torch.diag(eig))
                #eig_ = eig
                #eig_ = F.relu(eig)
                eig_ = eig/torch.abs(eig).max()
                #eig_ = nn.Softmax(-1)(eig_)
                recon_loss = 0

                eigs.append(eig_.unsqueeze(0))
            
            eigs = torch.cat(eigs, 0)

        for i in range(self.N):
            #resweight = F.relu(torch.sigmoid(self.resweight[i]))/self.N #torch.sigmoid(self.resweight[i]/((i+1)**2))
            resweight = torch.tanh(self.resweight[i])/self.N

            if output_expert_weights == True:
                x, expert_weights, expert_hidden_states = self.layers[i](x, resweight, eigs, True)
                all_expert_weights.append(expert_weights.unsqueeze(0))
                all_expert_hidden_states.append(expert_hidden_states.unsqueeze(0))
            else:
                x = self.layers[i](x, resweight, eigs)

                #print ("Layer {}".format(i+1))
                #print (euclidean_distances(x.detach().cpu().numpy()[0]))
            #print (x)

            if return_all_hidden_states == True:
                all_hidden_states.append(x)
        #recon_loss  = recon_loss/self.N

        self.all_resweights.append(self.resweight.item())

        if output_expert_weights == True:
            all_expert_weights = torch.cat(all_expert_weights, 0)
            all_expert_hidden_states = torch.cat(all_expert_hidden_states, 0)
            if return_all_hidden_states == True:
                return x, recon_loss, all_expert_weights, all_hidden_states, all_expert_hidden_states, eigs
            else:
                return x, recon_loss, all_expert_weights

        return x, recon_loss

class IsoDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, N, n_experts, use_eigen=False, random_features=True):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.use_eigen = use_eigen
        self.embed = Embedder(vocab_size, d_model//2)
        self.pe = ConcatenatedPositionalEncoder(d_model//2, max_len)
        self.decoder_attn_layers = nn.ModuleList([IsoEncoderLayer(d_model, n_experts,random_features=random_features) for i in range(N)])
        self.encoder_decoder_cross_attn_layers = nn.ModuleList([IsoEncoderLayer(d_model, n_experts, random_features=random_features) for i in range(N)])
        self.resweight = nn.Parameter(torch.Tensor([0]*N), requires_grad=True)
        self.U1 = torch.nn.utils.parametrizations.orthogonal(nn.Linear(d_model, d_model))
        self.U2 = torch.nn.utils.parametrizations.orthogonal(nn.Linear(d_model, d_model))

    def forward(self, src, encoder_out=None, output_expert_weights=False):
        x = self.embed(src)
        x = self.pe(x)
        #print (cosine_similarity(x.detach().cpu().numpy()[0]))

        recon_loss = 0
        if self.use_eigen == True and encoder_out is not None:
            M_encoder = torch.einsum('bmd,bme->bde', encoder_out,encoder_out)
            U1 = self.U1.weight
            bias1 = self.U1.bias
            M2_encoder = torch.einsum('bmn,mn->bmn',M_encoder,U1.transpose(0,1)) + bias1
            M2_encoder = torch.einsum('mn,bmn->bmn',U1,M2_encoder) + bias1
            #encoder_eigs2 = F.relu(torch.diagonal(M2_encoder, dim1 = -2, dim2 = -1)) #F.relu(torch.diagonal(M2_encoder, dim1 = -2, dim2 = -1))
            encoder_eigs2 = torch.tanh(torch.diagonal(M2_encoder, dim1 = -2, dim2 = -1))
            #encoder_eigs2 = encoder_eigs2/encoder_eigs2.max(1).values.unsqueeze(1)
            #encoder_eigs2 = torch.div(encoder_eigs2,encoder_eigs2.max(1).values)
            encoder_eigs2 = encoder_eigs2/encoder_eigs2.max(1).values.unsqueeze(1)

            encoder_eigs = []
            for bs in range(encoder_out.size(0)):
                eig_ = torch.diag(encoder_eigs2[bs])
                encoder_eigs.append(eig_.unsqueeze(0))
            
            encoder_eigs = torch.cat(encoder_eigs, 0)

            M_encoder_ = torch.einsum('bmn,mn->bmn',encoder_eigs, U1) + bias1
            M_encoder_ = torch.einsum('mn,bmn->bmn',U1.transpose(0,1),M_encoder_) + bias1
            recon_loss += 0.5 * nn.MSELoss(reduction='mean')(M_encoder, M_encoder_)

        recon_loss = 0

        if self.use_eigen == True:
            M_decoder = torch.einsum('bmd,bme->bde', torch.tril(x),torch.tril(x))
            U2 = self.U2.weight
            bias2 = self.U2.bias
            M2_decoder = torch.einsum('bmn,mn->bmn',M_decoder,U2.transpose(0,1)) + bias2
            M2_decoder = torch.einsum('mn,bmn->bmn',U2,M2_decoder) + bias2
            #decoder_eigs2 = F.relu(torch.diagonal(M2_decoder, dim1 = -2, dim2 = -1)) #F.relu(torch.diagonal(M2_decoder, dim1 = -2, dim2 = -1))
            decoder_eigs2 = torch.tanh(torch.diagonal(M2_decoder, dim1 = -2, dim2 = -1))
            #decoder_eigs2 = decoder_eigs2/decoder_eigs2.max(1).values.unsqueeze(1)
            #decoder_eigs2 = torch.div(decoder_eigs2,decoder_eigs2.max(1).values)
            decoder_eigs2 = decoder_eigs2/decoder_eigs2.max(1).values.unsqueeze(1)

            decoder_eigs = []
            for bs in range(x.size(0)):
                eig_ = torch.diag(decoder_eigs2[bs])
                decoder_eigs.append(eig_.unsqueeze(0))

            decoder_eigs = torch.cat(decoder_eigs, 0)

            M_decoder_ = torch.einsum('bmn,mn->bmn',decoder_eigs, U2) + bias2
            M_decoder_ = torch.einsum('mn,bmn->bmn',U2.transpose(0,1),M_decoder_) + bias2
            recon_loss += 0.5 * nn.MSELoss(reduction='mean')(M_decoder, M_decoder_)

        else:
            if encoder_out is not None:
                encoder_eigs = []
                for bs in range(x.size(0)):
                    eig = nn.Parameter(torch.rand(self.d_model), requires_grad=True).to(x.device)
                    #eig = F.relu(torch.diag(eig))
                    eig = torch.tanh(torch.diag(eig))
                    #eig_ = F.relu(eig)
                    #eig_ = SmeLU()(eig)
                    eig_ = eig/torch.abs(eig).max()

                    encoder_eigs.append(eig_.unsqueeze(0))
                
                encoder_eigs = torch.cat(encoder_eigs, 0)

            decoder_eigs = []
            for bs in range(x.size(0)):
                eig = nn.Parameter(torch.rand(self.d_model), requires_grad=True).to(x.device)
                #eig = F.relu(torch.diag(eig))
                eig = torch.tanh(torch.diag(eig))
                #eig_ = F.relu(eig)
                #eig_ = SmeLU()(eig)
                eig_ = eig/torch.abs(eig).max()

                decoder_eigs.append(eig_.unsqueeze(0))
            
            decoder_eigs = torch.cat(decoder_eigs, 0)

        all_expert_weights = []
        
        for i in range(self.N):
            #resweight = F.relu(torch.sigmoid(self.resweight[i]))/self.N #resweight = torch.sigmoid(self.resweight[i]/((i+1)**2))
            resweight = torch.tanh(self.resweight[i])/self.N

            if output_expert_weights == True:
                x, expert_weights = self.decoder_attn_layers[i](x, resweight, decoder_eigs, True)
                if encoder_out is not None:
                    x, expert_weights = self.encoder_decoder_cross_attn_layers[i](x, resweight, encoder_eigs, True)
                
                all_expert_weights.append(expert_weights.unsqueeze(0))
            else:
                x = self.decoder_attn_layers[i](x, resweight, decoder_eigs)
                if encoder_out is not None:
                    x = self.encoder_decoder_cross_attn_layers[i](x, resweight, encoder_eigs)
            
            #print (cosine_similarity(x.detach().cpu().numpy()[0]))

        #recon_loss  = recon_loss/self.N

        if output_expert_weights == True:
            all_expert_weights = torch.cat(all_expert_weights, 0)
            return x, recon_loss, all_expert_weights

        return x, recon_loss
        