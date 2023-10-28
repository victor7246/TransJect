import math
from pickletools import optimize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import IsoEncoder
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
from torch.optim.lr_scheduler import LambdaLR
import wandb
from transformers import get_linear_schedule_with_warmup

class IsoFormerForClassificationPL(LightningModule):
    def __init__(self, args, src_vocab, d_model, MAX_LEN, N, n_experts, n_out, use_eigen=False, pooling='mean', classification_type='binary', random_features=True):

        super().__init__()

        self.save_hyperparameters()
        self.args = args
        
        self.n_out = n_out
        self.pooling = pooling
        self.encoder = IsoEncoder(src_vocab, d_model, MAX_LEN, N, n_experts, use_eigen, random_features=random_features)
        self.out = nn.Linear(d_model, n_out)
        self.classification_type = classification_type

        self.best_acc = 0

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

        if self.args.classification_type == 'multiclass' and len(trg.size()) == 2:
            trg = trg[:,0]

        total_loss = self.loss_fn(outputs, trg) + lambda_ * l2_reg + lambda_ * recon_loss

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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