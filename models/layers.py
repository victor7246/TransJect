import torch
import torch.nn as nn 
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

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
            eigs2 = torch.tanh(torch.diagonal(M2, dim1 = -2, dim2 = -1))
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

            if return_all_hidden_states == True:
                all_hidden_states.append(x)

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
                    eig_ = eig/torch.abs(eig).max()

                    encoder_eigs.append(eig_.unsqueeze(0))
                
                encoder_eigs = torch.cat(encoder_eigs, 0)

            decoder_eigs = []
            for bs in range(x.size(0)):
                eig = nn.Parameter(torch.rand(self.d_model), requires_grad=True).to(x.device)
                #eig = F.relu(torch.diag(eig))
                eig = torch.tanh(torch.diag(eig))
                eig_ = eig/torch.abs(eig).max()

                decoder_eigs.append(eig_.unsqueeze(0))
            
            decoder_eigs = torch.cat(decoder_eigs, 0)

        all_expert_weights = []
        
        for i in range(self.N):
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

        if output_expert_weights == True:
            all_expert_weights = torch.cat(all_expert_weights, 0)
            return x, recon_loss, all_expert_weights

        return x, recon_loss
        