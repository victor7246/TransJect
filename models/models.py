import math
from pickletools import optimize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Encoder, Decoder, IsoEncoder, IsoDecoder, EncoderNew
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from pytorch_lightning import LightningModule
from .utils import ScheduledOptim, create_masks
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
from torch.optim.lr_scheduler import LambdaLR
import wandb
from transformers import get_linear_schedule_with_warmup

class TransformerForClassificationPLNew(LightningModule):
    def __init__(self, args, src_vocab, d_model, MAX_LEN, N, heads, dropout, n_out, use_rezero, use_ortho, use_random):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        #self.automatic_optimization = False
        
        self.n_out = n_out
        self.encoder = EncoderNew(src_vocab, MAX_LEN, d_model, N, heads, dropout, use_rezero, use_ortho, use_random)
        self.out = nn.Linear(d_model, n_out)

        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        #self.sched = ScheduledOptim(self.optimizer, d_model=self.args.d_model, n_warmup_steps=self.args.n_warmup, lr_mul=self.args.max_lr)

        if self.args.classification_type == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.args.classification_type == 'multiclass':
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, mask=None, debug=False):
        if debug == True:
            hidden_state, all_hidden_states, all_attn_scores = self.encoder(src, mask, True, True)
        else:
            hidden_state = self.encoder(src, mask)

        if self.n_out > 1:
            out = self.out(hidden_state.mean(1))
        else:
            out = self.out(hidden_state.mean(1))

        if debug == True:
            return out, all_hidden_states, all_attn_scores
        else:
            return out

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_

        src = batch['input_ids']
        trg = batch['out']

        #self.sched.zero_grad()
        outputs = self(src)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        #self.manual_backward(total_loss)
        #self.sched.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.trainer.optimizers[0].params['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        trg = batch['out']

        outputs = self(src)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        if self.args.classification_type == 'binary':
            outputs = torch.sigmoid(outputs)
        elif self.args.classification_type == 'multiclass':
            outputs = outputs.argmax(-1)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)   

        return {"val_loss": total_loss, "pred": outputs, "target": trg}

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        predictions = np.concatenate([x["pred"].detach().cpu().numpy() for x in batch_parts],0)
        target = np.concatenate([x["target"].detach().cpu().numpy() for x in batch_parts],0)
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        print ("Validation Acc:{}".format(accuracy_score(target.round(),predictions.round())))

        print ("Confusion: {}".format(confusion_matrix(target.round(),predictions.round())))
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation_acc": accuracy_score(target.round(),predictions.round())})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.args.n_warmup,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        #)
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        #return [optimizer], [scheduler]
        return 

class TransformerForClassification(nn.Module):
    def __init__(self, src_vocab, d_model, MAX_LEN, N, heads, dropout, n_out, use_rezero, use_ortho, use_random):
        super().__init__()
        self.n_out = n_out
        self.encoder = Encoder(src_vocab, MAX_LEN, d_model, N, heads, dropout, use_rezero, use_ortho, use_random)
        self.out = nn.Linear(d_model, n_out)
    def forward(self, src, mask=None):
        hidden_state = self.encoder(src, mask)
        if self.n_out > 1:
            out = nn.Softmax(-1)(self.out(hidden_state.mean(1)))
        else:
            out = self.out(hidden_state.mean(1))
        #print (cosine_similarity(hidden_state.mean(1).detach().cpu().numpy()))
        return out

class IsoFormerForClassification(nn.Module):
    def __init__(self, src_vocab, d_model, MAX_LEN, N, n_experts, n_out, use_eigen=False, random_features=True, pooling='mean', classification_type='binary',use_bert=False):
        super().__init__()
        self.n_out = n_out
        self.pooling = pooling
        self.encoder = IsoEncoder(src_vocab, d_model, MAX_LEN, N, n_experts, use_eigen, random_features=random_features, use_bert=use_bert)
        self.out = nn.Linear(d_model, n_out)
        self.classification_type = classification_type

    def forward(self, src, output_expert_weights=False):
        if output_expert_weights == True:
            hidden_state, recon_loss, all_expert_weights = self.encoder(src, True)
        else:
            hidden_state, recon_loss = self.encoder(src)

        if self.n_out > 1:
            if self.pooling == 'mean':
                out = nn.Softmax(-1)(self.out(hidden_state.mean(1)))
            elif self.pooling == 'max':
                out = nn.Softmax(-1)(self.out(hidden_state.max(1).values))
            elif self.pooling == 'min':
                out = nn.Softmax(-1)(self.out(hidden_state.min(1).values))
            elif self.pooling == 'first':
                out = nn.Softmax(-1)(self.out(hidden_state[:,0,:]))
        else:
            if self.pooling == 'mean':
                out = self.out(hidden_state.mean(1))
            elif self.pooling == 'max':
                out = self.out(hidden_state.max(1).values)
            elif self.pooling == 'min':
                out = self.out(hidden_state.min(1).values)
            elif self.pooling == 'first':
                out = self.out(hidden_state[:,0,:])
        #print (cosine_similarity(hidden_state.mean(1).detach().cpu().numpy()))
        if self.classification_type == 'regression':
            out = F.relu(out)
            
        if output_expert_weights == True:
            return out, recon_loss, output_expert_weights
        else:
            return out, recon_loss

class IsoFormerForGeneration(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, encoder_MAX_LEN, decoder_MAX_LEN, N, n_experts, n_out, use_eigen=False, random_features=True):
        super().__init__()
        self.n_out = n_out
        self.encoder = IsoEncoder(src_vocab, d_model, encoder_MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.decoder = IsoDecoder(trg_vocab, d_model, decoder_MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, encoder_src, decoder_src, output_expert_weights=False): 
        if output_expert_weights == True:
            hidden_state, recon_loss, all_expert_weights = self.encoder(encoder_src, True)
        else:
            hidden_state, recon_loss = self.encoder(encoder_src)

        if output_expert_weights == True:
            decoder_hidden_state, recon_loss2, all_expert_weights = self.decoder(decoder_src, hidden_state, True)
        else:
            decoder_hidden_state, recon_loss2 = self.decoder(decoder_src, hidden_state)

        out = nn.Softmax(-1)(self.out(decoder_hidden_state))
        recon_loss = 0.5*recon_loss + 0.5*recon_loss2

        if output_expert_weights == True:
            return out, recon_loss, output_expert_weights
        else:
            return out, recon_loss

class IsoFormerForGenerationPL(LightningModule):
    def __init__(self, args, src_vocab, trg_vocab, d_model, encoder_MAX_LEN, decoder_MAX_LEN, N, n_experts, n_out, use_eigen=False, random_features=True):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        #self.automatic_optimization = False
        
        self.n_out = n_out
        self.encoder = IsoEncoder(src_vocab, d_model, encoder_MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.decoder = IsoDecoder(trg_vocab, d_model, decoder_MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, encoder_src, decoder_src):
        hidden_state, recon_loss = self.encoder(encoder_src)

        decoder_hidden_state, recon_loss2 = self.decoder(decoder_src, hidden_state)

        out = nn.Softmax(-1)(self.out(decoder_hidden_state))
        recon_loss = 0.5*recon_loss + 0.5*recon_loss2
        
        return out, recon_loss

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_
        #opt = self.sched

        src = batch['input_ids']
        trg = batch['output_ids']
        decoder_src = trg[:,1:-1]
        target = trg[:,2:].contiguous().view(-1)

        #opt.zero_grad()
        outputs, recon_loss = self(src,decoder_src)

        l2_reg = 0
        #for param in self.encoder.parameters():
        #    l2_reg += torch.norm(param)
        #for param in self.decoder.parameters():
        #    l2_reg += torch.norm(param)
        #for param in self.out.parameters():
        #    l2_reg += torch.norm(param)

        total_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target, ignore_index=0) + \
                    lambda_ * l2_reg + lambda_ * recon_loss

        #self.manual_backward(total_loss)
        #opt.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        self.log("train_loss", total_loss)
        
        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        trg = batch['output_ids']
        decoder_src = trg[:,1:-1]
        target = trg[:,2:].contiguous().view(-1)

        outputs, recon_loss = self(src,decoder_src)

        total_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target, ignore_index=0)

        self.log("val_loss", total_loss)

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.args.n_warmup,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        #)
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        #return [optimizer], [scheduler]
        return optimizer

class IsoFormerForClassificationPL(LightningModule):
    def __init__(self, args, src_vocab, d_model, MAX_LEN, N, n_experts, n_out, use_eigen=False, pooling='mean', classification_type='binary', random_features=True):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        #self.automatic_optimization = False
        
        self.n_out = n_out
        self.pooling = pooling
        self.encoder = IsoEncoder(src_vocab, d_model, MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.out = nn.Linear(d_model, n_out)
        self.classification_type = classification_type

        self.best_acc = 0
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        #self.sched = ScheduledOptim(self.optimizer, d_model=self.args.d_model, n_warmup_steps=self.args.n_warmup, lr_mul=self.args.max_lr)

        if self.args.classification_type == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.args.classification_type == 'multiclass':
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, debug=False):
        if debug == True:
            hidden_state, recon_loss, all_expert_weights, all_hidden_states,all_expert_hidden_states,  eigs = self.encoder(src, True, True)
        else:
            hidden_state, recon_loss = self.encoder(src)

        if self.n_out > 1:
            if self.pooling == 'mean':
                out = nn.Softmax(-1)(self.out(hidden_state.mean(1)))
            elif self.pooling == 'max':
                out = nn.Softmax(-1)(self.out(hidden_state.max(1).values))
            elif self.pooling == 'min':
                out = nn.Softmax(-1)(self.out(hidden_state.min(1).values))
            elif self.pooling == 'first':
                out = nn.Softmax(-1)(self.out(hidden_state[:,0,:]))
        else:
            if self.pooling == 'mean':
                out = self.out(hidden_state.mean(1))
            elif self.pooling == 'max':
                out = self.out(hidden_state.max(1).values)
            elif self.pooling == 'min':
                out = self.out(hidden_state.min(1).values)
            elif self.pooling == 'first':
                out = self.out(hidden_state[:,0,:])
        
        #print (cosine_similarity(hidden_state.mean(1).detach().cpu().numpy()))
        if self.classification_type == 'regression':
            out = F.relu(out)
        
        if debug == True:
            return out, recon_loss, all_expert_weights, all_hidden_states, all_expert_hidden_states, eigs
        else:
            return out, recon_loss

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_

        src = batch['input_ids']
        trg = batch['out']

        #self.sched.zero_grad()
        outputs, recon_loss = self(src)

        l2_reg = 0
        #for param in self.encoder.parameters():
        #    l2_reg += torch.norm(param)
        #for param in self.out.parameters():
        #    l2_reg += torch.norm(param)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg) + lambda_ * l2_reg + lambda_ * recon_loss

        #self.manual_backward(total_loss)
        #self.sched.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        #print (self.trainer.optimizers[0])
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.trainer.optimizers[0].params['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        trg = batch['out']

        outputs, recon_loss = self(src)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        if self.args.classification_type == 'binary':
            outputs = torch.sigmoid(outputs)
        elif self.args.classification_type == 'multiclass':
            outputs = outputs.argmax(-1)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        
        return {"val_loss": total_loss, "pred": outputs, "target": trg}

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        predictions = np.concatenate([x["pred"].detach().cpu().numpy() for x in batch_parts],0)
        target = np.concatenate([x["target"].detach().cpu().numpy() for x in batch_parts],0)
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        acc = accuracy_score(target.round(),predictions.round())
        print ("Validation Acc:{}".format(acc))

        if acc > self.best_acc:
            self.best_acc = acc

        print ("Confusion: {}".format(confusion_matrix(target.round(),predictions.round())))
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation_acc": acc})
            wandb.log({"best_validation_acc": self.best_acc})

        #return {'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.args.n_warmup,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        #)
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        #return [optimizer], [scheduler]
        return optimizer

class IsoFormerForImageClassificationPL(LightningModule):
    def __init__(self, args, src_vocab, d_model, MAX_LEN, N, n_experts, n_out, use_eigen=False, pooling='mean', classification_type='binary', random_features=True):

        super().__init__()

        self.save_hyperparameters()
        self.args = args
        self.MAX_LEN = MAX_LEN
        #self.automatic_optimization = False
        
        self.n_out = n_out
        self.pooling = pooling
        self.encoder = IsoEncoder(src_vocab, d_model, MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.out = nn.Linear(d_model, n_out)
        self.classification_type = classification_type

        self.best_acc = 0
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        #self.sched = ScheduledOptim(self.optimizer, d_model=self.args.d_model, n_warmup_steps=self.args.n_warmup, lr_mul=self.args.max_lr)

        if self.args.classification_type == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.args.classification_type == 'multiclass':
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, debug=False):
        if debug == True:
            hidden_state, recon_loss, all_expert_weights, all_hidden_states,all_expert_hidden_states,  eigs = self.encoder(src, True, True)
        else:
            hidden_state, recon_loss = self.encoder(src)

        #if self.n_out > 1:
        #    if self.pooling == 'mean':
        #        out = nn.Softmax(-1)(self.out(hidden_state.mean(1)))
        #    elif self.pooling == 'max':
        #        out = nn.Softmax(-1)(self.out(hidden_state.max(1).values))
        #    elif self.pooling == 'min':
        #        out = nn.Softmax(-1)(self.out(hidden_state.min(1).values))
        #    elif self.pooling == 'first':
        #        out = nn.Softmax(-1)(self.out(hidden_state[:,0,:]))
        #else:
        if self.pooling == 'mean':
            out = self.out(hidden_state.mean(1))
        elif self.pooling == 'max':
            out = self.out(hidden_state.max(1).values)
        elif self.pooling == 'min':
            out = self.out(hidden_state.min(1).values)
        elif self.pooling == 'first':
            out = self.out(hidden_state[:,0,:])
        
        #print (cosine_similarity(hidden_state.mean(1).detach().cpu().numpy()))
        if self.classification_type == 'regression':
            out = F.relu(out)
        
        if debug == True:
            return out, recon_loss, all_expert_weights, all_hidden_states, all_expert_hidden_states, eigs
        else:
            return out, recon_loss

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_

        src = batch['input_ids']
        trg = batch['out']

        src = src.reshape(src.size(0),self.MAX_LEN)

        #self.sched.zero_grad()
        outputs, recon_loss = self(src)

        l2_reg = 0
        #for param in self.encoder.parameters():
        #    l2_reg += torch.norm(param)
        #for param in self.out.parameters():
        #    l2_reg += torch.norm(param)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg) + lambda_ * l2_reg + lambda_ * recon_loss

        #self.manual_backward(total_loss)
        #self.sched.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        #print (self.trainer.optimizers[0])
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.trainer.optimizers[0].params['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        trg = batch['out']

        src = src.reshape(src.size(0),self.MAX_LEN)

        outputs, recon_loss = self(src)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        if self.args.classification_type == 'binary':
            outputs = torch.sigmoid(outputs)
        elif self.args.classification_type == 'multiclass':
            outputs = outputs.argmax(-1)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        
        return {"val_loss": total_loss, "pred": outputs, "target": trg}

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        predictions = np.concatenate([x["pred"].detach().cpu().numpy() for x in batch_parts],0)
        target = np.concatenate([x["target"].detach().cpu().numpy() for x in batch_parts],0)
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        acc = accuracy_score(target.round(),predictions.round())
        print ("Validation Acc:{}".format(acc))

        if acc > self.best_acc:
            self.best_acc = acc

        print ("Confusion: {}".format(confusion_matrix(target.round(),predictions.round())))
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation_acc": acc})
            wandb.log({"best_validation_acc": self.best_acc})

        #return {'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.args.n_warmup,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        #)
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        #return [optimizer], [scheduler]
        return optimizer

class IsoFormerForSimilarityPL(LightningModule):
    def __init__(self, args, src_vocab, d_model, MAX_LEN, N, n_experts, n_out, use_eigen=False, pooling='mean', classification_type='binary', random_features=True):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        #self.automatic_optimization = False
        
        self.n_out = n_out
        self.pooling = pooling
        self.encoder = IsoEncoder(src_vocab, d_model, MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.out = nn.Linear(1, n_out)
        self.classification_type = classification_type

        self.best_acc = 0
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        #self.sched = ScheduledOptim(self.optimizer, d_model=self.args.d_model, n_warmup_steps=self.args.n_warmup, lr_mul=self.args.max_lr)

        if self.args.classification_type == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.args.classification_type == 'multiclass':
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src1, src2, debug=False):
        if debug == True:
            hidden_state1, recon_loss1, all_expert_weights1, all_hidden_states1,all_expert_hidden_states1,  eigs1 = self.encoder(src1, True, True)
        else:
            hidden_state1, recon_loss1 = self.encoder(src1)

        if debug == True:
            hidden_state2, recon_loss2, all_expert_weights2, all_hidden_states2,all_expert_hidden_states2,  eigs2 = self.encoder(src2, True, True)
        else:
            hidden_state2, recon_loss2 = self.encoder(src2)

        if self.pooling == 'mean':
            out1 = hidden_state1.mean(1)
            out2 = hidden_state2.mean(1)
        elif self.pooling == 'max':
            out1 = hidden_state1.max(1).values
            out2 = hidden_state2.max(1).values
        elif self.pooling == 'min':
            out1 = hidden_state1.min(1).values
            out2 = hidden_state2.min(1).values
        elif self.pooling == 'first':
            out1 = hidden_state1[:,0,:]
            out2 = hidden_state2[:,0,:]
        
        out = self.out((torch.sum(out1/torch.norm(out1,p=2) * out2/torch.norm(out2,p=2), dim=-1).unsqueeze(1)))
        #print (out)
        #print ((torch.sum(out1/torch.norm(out1,p=2) * out2/torch.norm(out2,p=2), dim=-1).unsqueeze(1)))

        recon_loss = 0.5*recon_loss1 + 0.5*recon_loss2

        #print (cosine_similarity(hidden_state.mean(1).detach().cpu().numpy()))
        if self.classification_type == 'regression':
            out = F.relu(out)
        
        if debug == True:
            return out, recon_loss, all_expert_weights1, all_hidden_states1, all_expert_hidden_states1, eigs1, all_expert_weights2, all_hidden_states2, all_expert_hidden_states2, eigs2
        else:
            return out, recon_loss

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_

        src1 = batch['input_ids1']
        src2 = batch['input_ids2']
        trg = batch['out']

        #self.sched.zero_grad()
        outputs, recon_loss = self(src1,src2)

        l2_reg = 0
        #for param in self.encoder.parameters():
        #    l2_reg += torch.norm(param)
        #for param in self.out.parameters():
        #    l2_reg += torch.norm(param)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        #print (outputs, trg)
        #print (self.loss_fn(outputs, trg))

        total_loss = self.loss_fn(outputs, trg) + lambda_ * l2_reg + lambda_ * recon_loss

        #self.manual_backward(total_loss)
        #self.sched.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        #print (self.trainer.optimizers[0])
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.trainer.optimizers[0].params['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src1 = batch['input_ids1']
        src2 = batch['input_ids2']

        trg = batch['out']

        outputs, recon_loss = self(src1, src2)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        if self.args.classification_type == 'binary':
            outputs = torch.sigmoid(outputs)
        elif self.args.classification_type == 'multiclass':
            outputs = outputs.argmax(-1)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        
        return {"val_loss": total_loss, "pred": outputs, "target": trg}

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        predictions = np.concatenate([x["pred"].detach().cpu().numpy() for x in batch_parts],0)
        target = np.concatenate([x["target"].detach().cpu().numpy() for x in batch_parts],0)
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        acc = accuracy_score(target.round(),predictions.round())
        print ("Validation Acc:{}".format(acc))

        if acc > self.best_acc:
            self.best_acc = acc

        print ("Confusion: {}".format(confusion_matrix(target.round(),predictions.round())))
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation_acc": acc})
            wandb.log({"best_validation_acc": self.best_acc})

        #return {'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)

        return optimizer

class IsoFormerForMLMPL(LightningModule):
    def __init__(self, args, src_vocab, d_model, MAX_LEN, N, n_experts, use_eigen=False, random_features=True):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        #self.automatic_optimization = False
        
        self.encoder = IsoEncoder(src_vocab, d_model, MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.out = nn.Linear(d_model, src_vocab)

        self.best_acc = 0

    def forward(self, src, debug=False):
        if debug == True:
            hidden_state, recon_loss, all_expert_weights, all_hidden_states, eigs = self.encoder(src, True, True)
        else:
            hidden_state, recon_loss = self.encoder(src)

        out = nn.Softmax(-1)(self.out(hidden_state))
        
        if debug == True:
            return out, recon_loss, all_expert_weights, all_hidden_states, eigs
        else:
            return out, recon_loss

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_

        input = batch['input_ids'].clone()

        rand_value = torch.rand(input.shape).to(input.device)
        rand_mask = (rand_value < 0.15) * (input != 101) * (input != 102) * (input != 0)
        mask_idx = (rand_mask.flatten() == True).nonzero().view(-1)
        input = input.flatten()
        input[mask_idx] = 103
        input = input.view(batch['input_ids'].size())

        trg = batch['output_ids'].contiguous().view(-1)

        #self.sched.zero_grad()
        outputs, recon_loss = self(input)

        l2_reg = 0

        total_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), trg, ignore_index=0) + \
                    lambda_ * l2_reg + lambda_ * recon_loss

        #print (self.trainer.optimizers[0])
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.trainer.optimizers[0].params['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        input = batch['input_ids'].clone()

        rand_value = torch.rand(input.shape).to(input.device)
        rand_mask = (rand_value < 0.15) * (input != 101) * (input != 102) * (input != 0)
        mask_idx = (rand_mask.flatten() == True).nonzero().view(-1)
        input = input.flatten()
        input[mask_idx] = 103
        input = input.view(batch['input_ids'].size())

        trg = batch['output_ids'].contiguous().view(-1)

        #self.sched.zero_grad()
        outputs, _ = self(input)

        total_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), trg, ignore_index=0)

        outputs = outputs.argmax(-1)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        
        return {"val_loss": total_loss, "pred": outputs.contiguous().view(-1), "target": trg}

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        predictions = np.concatenate([x["pred"].detach().cpu().numpy() for x in batch_parts],0)
        target = np.concatenate([x["target"].detach().cpu().numpy() for x in batch_parts],0)
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        acc = accuracy_score(target.round(),predictions.round())
        print ("Validation Acc:{}".format(acc))

        if acc > self.best_acc:
            self.best_acc = acc
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation_acc": acc})
            wandb.log({"best_validation_acc": self.best_acc})

        #return {'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)

        return optimizer

class IsoFormerDecoderPL(LightningModule):
    def __init__(self, args, trg_vocab, d_model, decoder_MAX_LEN, N, n_experts, n_out, use_eigen=False, random_features=True):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        #self.automatic_optimization = False
        
        self.n_out = n_out
        self.decoder = IsoDecoder(trg_vocab, d_model, decoder_MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.out = nn.Linear(d_model, trg_vocab)

        self.best_ppl = 1000000

    def forward(self, encoder_src, decoder_src):
        decoder_hidden_state, recon_loss2 = self.decoder(decoder_src, encoder_out=None)

        out = nn.Softmax(-1)(self.out(decoder_hidden_state))
        recon_loss = 2*recon_loss2
        
        return out, recon_loss

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_
        #opt = self.sched

        src = batch['input_ids']
        trg = batch['output_ids']
        decoder_src = trg[:,1:-1]
        target = trg[:,2:].contiguous().view(-1)

        #opt.zero_grad()
        outputs, recon_loss = self(src,decoder_src)

        l2_reg = 0
        #for param in self.encoder.parameters():
        #    l2_reg += torch.norm(param)
        #for param in self.decoder.parameters():
        #    l2_reg += torch.norm(param)
        #for param in self.out.parameters():
        #    l2_reg += torch.norm(param)

        total_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target, ignore_index=0) + \
                    lambda_ * l2_reg + lambda_ * recon_loss

        #self.manual_backward(total_loss)
        #opt.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        self.log("train_loss", total_loss)
        
        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        trg = batch['output_ids']
        decoder_src = trg[:,1:-1]
        target = trg[:,2:].contiguous().view(-1)

        outputs, recon_loss = self(src,decoder_src)

        total_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target, ignore_index=0)

        self.log("val_loss", total_loss)

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()
        perplexity = np.exp(validation_loss)

        if perplexity < self.best_ppl:
            self.best_ppl = perplexity

        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation_perplexity": perplexity})
            wandb.log({"best_validation_perplexity": self.best_ppl})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.args.n_warmup,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        #)
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        #return [optimizer], [scheduler]
        return optimizer

class TransformerForClassificationPL(LightningModule):
    def __init__(self, args, src_vocab, d_model, MAX_LEN, N, heads, dropout, n_out, use_rezero):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        #self.automatic_optimization = False
        
        self.n_out = n_out
        self.encoder = Encoder(src_vocab, MAX_LEN, d_model, N, heads, dropout, use_rezero)
        self.out = nn.Linear(d_model, n_out)

        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        #self.sched = ScheduledOptim(self.optimizer, d_model=self.args.d_model, n_warmup_steps=self.args.n_warmup, lr_mul=self.args.max_lr)

        if self.args.classification_type == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.args.classification_type == 'multiclass':
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, mask=None, debug=False):
        if debug == True:
            hidden_state, all_hidden_states, all_attn_scores = self.encoder(src, mask, True, True)
        else:
            hidden_state = self.encoder(src, mask)

        if self.n_out > 1:
            out = nn.Softmax(-1)(self.out(hidden_state.mean(1)))
        else:
            out = self.out(hidden_state.mean(1))

        if debug == True:
            return out, all_hidden_states, all_attn_scores
        else:
            return out

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_

        src = batch['input_ids']
        trg = batch['out']

        #self.sched.zero_grad()
        outputs = self(src)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        #self.manual_backward(total_loss)
        #self.sched.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.trainer.optimizers[0].params['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        trg = batch['out']

        outputs = self(src)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        if self.args.classification_type == 'binary':
            outputs = torch.sigmoid(outputs)
        elif self.args.classification_type == 'multiclass':
            outputs = outputs.argmax(-1)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)   

        return {"val_loss": total_loss, "pred": outputs, "target": trg}

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        predictions = np.concatenate([x["pred"].detach().cpu().numpy() for x in batch_parts],0)
        target = np.concatenate([x["target"].detach().cpu().numpy() for x in batch_parts],0)
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        print ("Validation Acc:{}".format(accuracy_score(target.round(),predictions.round())))

        print ("Confusion: {}".format(confusion_matrix(target.round(),predictions.round())))
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation_acc": accuracy_score(target.round(),predictions.round())})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.args.n_warmup,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        #)
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        #return [optimizer], [scheduler]
        return optimizer

class TransformerForImageClassificationPL(LightningModule):
    def __init__(self, args, src_vocab, d_model, MAX_LEN, N, heads, dropout, n_out, use_rezero, use_ortho, use_random):

        super().__init__()

        self.save_hyperparameters()
        self.args = args
        self.MAX_LEN = MAX_LEN
        #self.automatic_optimization = False
        
        self.n_out = n_out
        self.encoder = Encoder(src_vocab, MAX_LEN, d_model, N, heads, dropout, use_rezero, use_ortho, use_random)
        self.out = nn.Linear(d_model, n_out)

        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        #self.sched = ScheduledOptim(self.optimizer, d_model=self.args.d_model, n_warmup_steps=self.args.n_warmup, lr_mul=self.args.max_lr)

        if self.args.classification_type == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.args.classification_type == 'multiclass':
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, mask=None, debug=False):
        print (src)

        if debug == True:
            hidden_state, all_hidden_states, all_attn_scores = self.encoder(src, None, True, True)
        else:
            hidden_state = self.encoder(src, None)

        if self.n_out > 1:
            out = nn.Softmax(-1)(self.out(hidden_state.mean(1)))
        else:
            out = self.out(hidden_state.mean(1))

        if debug == True:
            return out, all_hidden_states, all_attn_scores
        else:
            return out

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_

        src = batch['input_ids']
        trg = batch['out']

        src = src.reshape(src.size(0),self.MAX_LEN)

        #self.sched.zero_grad()
        outputs = self(src)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        #self.manual_backward(total_loss)
        #self.sched.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.trainer.optimizers[0].params['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        trg = batch['out']

        src = src.reshape(src.size(0),self.MAX_LEN)

        outputs = self(src)

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg)

        if self.args.classification_type == 'binary':
            outputs = torch.sigmoid(outputs)
        elif self.args.classification_type == 'multiclass':
            outputs = outputs.argmax(-1)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)   

        return {"val_loss": total_loss, "pred": outputs, "target": trg}

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        predictions = np.concatenate([x["pred"].detach().cpu().numpy() for x in batch_parts],0)
        target = np.concatenate([x["target"].detach().cpu().numpy() for x in batch_parts],0)
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        print ("Validation Acc:{}".format(accuracy_score(target.round(),predictions.round())))

        print ("Confusion: {}".format(confusion_matrix(target.round(),predictions.round())))
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})
            wandb.log({"validation_acc": accuracy_score(target.round(),predictions.round())})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.args.n_warmup,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        #)
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        #return [optimizer], [scheduler]
        return optimizer

class TransformerForGenerationPL(LightningModule):
    def __init__(self,  args, src_vocab, trg_vocab, d_model, encoder_MAX_LEN, decoder_MAX_LEN, N, heads, n_out, dropout, use_rezero):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        #self.automatic_optimization = False
        
        self.encoder = Encoder(src_vocab, encoder_MAX_LEN, d_model, N, heads, dropout, use_rezero)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, use_rezero)
        self.out = nn.Linear(d_model, trg_vocab)

        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        #self.sched = ScheduledOptim(self.optimizer, d_model=self.args.d_model, n_warmup_steps=self.args.n_warmup, lr_mul=self.args.max_lr)


    def forward(self, src, trg, mask=None, trg_mask=None):
        hidden_state = self.encoder(src, mask)
        d_output = self.decoder(trg, hidden_state, mask, trg_mask)
        out = nn.Softmax(-1)(self.out(d_output))

        return out

    def training_step(self, batch, batch_idx):
        lambda_ = self.args.lambda_

        src = batch['input_ids']
        trg = batch['output_ids']
        decoder_src = trg[:,1:-1]
        target = trg[:,2:].contiguous().view(-1)

        src_mask, trg_mask = create_masks(src, decoder_src)

        #self.sched.zero_grad()
        outputs = self(src, decoder_src, src_mask, trg_mask)

        total_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target, ignore_index=0)

        #self.manual_backward(total_loss)
        #self.sched.step()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.trainer.optimizers[0].params['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        trg = batch['output_ids']
        decoder_src = trg[:,1:-1]
        target = trg[:,2:].contiguous().view(-1)

        src_mask, trg_mask = create_masks(src, decoder_src)

        #self.sched.zero_grad()
        outputs = self(src, decoder_src, src_mask, trg_mask)

        total_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target, ignore_index=0)

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)   

        return {"val_loss": total_loss, "pred": outputs, "target": trg}

    def training_epoch_end(self, batch_parts):
        training_loss = np.asarray([x['loss'].detach().cpu().numpy() for x in batch_parts]).mean()

        if self.args.wandb_logging:
            wandb.log({"training_loss": training_loss})

    def validation_epoch_end(self, batch_parts):
        validation_loss = np.asarray([x['val_loss'].detach().cpu().numpy() for x in batch_parts]).mean()
        
        if self.args.wandb_logging:
            wandb.log({"validation_loss": validation_loss})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98), eps=1e-09)
        
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=self.args.n_warmup,
        #    num_training_steps=self.trainer.estimated_stepping_batches,
        #)
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        #return [optimizer], [scheduler]
        return optimizer