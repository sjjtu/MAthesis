# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler
from fastNLP import seq_len_to_mask
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
import random
from torch import autograd
import time
from autodp import privacy_calibrator
from opacus import PrivacyEngine


from util.autoencoder import Autoencoder
from util.gan import Generator, Discriminator
from util.TrainRoutine import AutoEncTrainRoutine
from util.ECGDataset import ECGDataset
    
class AeGAN:
    def __init__(self, params):
        self.params = params
        self.device = params["device"]
        self.logger = params["logger"]

        self.ae = AutoEncTrainRoutine(emb_dim=self.params["hidden_dim"])
        self.ae.model.to(self.device)
        """
        self.decoder_optm = torch.optim.Adam(
            params=self.ae.decoder.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
        )
        self.encoder_optm = torch.optim.Adam(
            params=self.ae.encoder.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
        )
        """
        
        self.loss_con = nn.MSELoss(reduction='none')
        self.loss_dis = nn.NLLLoss(reduction='none')
        self.loss_mis = nn.BCELoss(reduction='none')
        
        self.generator = Generator(self.params["noise_dim"], self.params["hidden_dim"], self.params["layers"]).to(self.device)
        self.discriminator = Discriminator(self.params["embed_dim"]).to(self.device)
        self.discriminator_optm = torch.optim.RMSprop(
            params=self.discriminator.parameters(),
            lr=self.params['gan_lr'],
            alpha=self.params['gan_alpha'],
        )
        self.generator_optm = torch.optim.RMSprop(
            params=self.generator.parameters(),
            lr=self.params['gan_lr'],
            alpha=self.params['gan_alpha'],
        )

    def load_ae(self, pretrained_dir=None):
        if pretrained_dir is not None:
            path = pretrained_dir
        else:
            path = '{}/ae.dat'.format(self.params["root_dir"])
        self.logger.info("load: "+path)
        self.ae.load_model(path)
        return self.ae.model

    def load_generator(self, pretrained_path=None):
        path = pretrained_path
        self.logger.info("load: "+path)
        self.generator.load_state_dict(torch.load(path, map_location=self.device))

    def sta_loss(self, data, target):
        loss = 0
        n = len(self.static_processor.models)
        st = 0
        for model in self.static_processor.models:
            ed = st + model.length - int(model.missing)
            use = 1
            if model.missing:
                loss += torch.mean(self.loss_mis(data[:, ed], target[:, ed])) 
                use = 0.1 + target[:, ed:ed+1]
             
            if model.which == "categorical":
                loss += torch.mean(use * self.loss_dis((data[:, st:ed]+1e-8).log(), target[:,st:ed]).unsqueeze(-1))
            elif model.which =="binary" :
                loss += torch.mean(use * self.loss_mis(data[:, st:ed], target[:, st:ed]))
            else:
                loss += torch.mean(use * self.loss_con(data[:, st:ed], target[:, st:ed]))
 
            st += model.length
        return loss/n
    
    def dyn_loss(self, data, target, seq_len):
        loss = []
        n = len(self.dynamic_processor.models)
        st = 0
        for model in self.dynamic_processor.models:
            ed = st + model.length - int(model.missing)
            use = 1
            if model.missing:
                loss.append(self.loss_mis(data[:, :, ed], target[:, :, ed]).unsqueeze(-1))
                use = 0.1 + target[:, :, ed:ed+1]
                
            if model.which == "categorical":
                loss.append(use * self.loss_dis((data[:, :, st:ed]+1e-8).log(), target[:, :, st:ed]).unsqueeze(-1))
            elif model.which =="binary" :
                loss.append(use * self.loss_mis(data[:, :, st:ed], target[:, :, st:ed]))
            else:
                #print(data.size(), target.size(), st, ed)
                loss.append(use * self.loss_con(data[:, :, st:ed], target[:, :, st:ed]))
            st += model.length
        loss = torch.cat(loss, dim=-1)
        mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, mask.unsqueeze(-1))
        return torch.mean(loss)
        
    def train_ae(self, train_ds_path, val_ds_path, n_epochs=20, lr=5e-4, batch_size=1):
        return self.ae.train_model(train_ds_path=train_ds_path, val_ds_path=val_ds_path, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    
    def train_gan(self, train_ds_path, iterations=15000, d_update=5, eps=None):
        batch_size = self.params["gan_batch_size"]
        
        train_ds = ECGDataset(train_ds_path)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True) # will reshuffle at every epoch
        if eps!=None:
             privacy_engine = PrivacyEngine()
             self.discriminator, self.discriminator_optm, train_dl = privacy_engine.make_private(
                                                 module=self.discriminator,
                                                 optimizer=self.discriminator_optm,
                                                 data_loader=train_dl,
                                                 noise_multiplier=8.0,
                                                 max_grad_norm=2.0,
                                                 poisson_sampling=False
                                                
                                             )
        self.discriminator.train()
        self.generator.train()
        self.ae.model.train()
        
        
        history = dict(d_loss=[], g_loss=[])

        for iteration in range(iterations):
            
            avg_d_loss = 0
            t1 = time.time()

            toggle_grad(self.generator, False)
            toggle_grad(self.discriminator, True)
            self.generator.train()
            self.discriminator.train()
            
            for j in range(d_update):
                for batch_nr, (batch_x, batch_y) in enumerate(train_dl):        
                    self.discriminator_optm.zero_grad()
                    z = torch.randn(batch_size, self.params['noise_dim']).to(self.device)

                    sta = None
                    dyn = batch_x.to(self.device)
                    real_rep = self.ae.model.encoder(dyn).squeeze()
                    # if eps!=None:
                    #     delta=1e-5
                    #     privacy_param = privacy_calibrator.gaussian_mech(eps, delta)
                    #     sensitivity = 2 / batch_size
                    #     noise_std_for_privacy = privacy_param['sigma'] * sensitivity
                    #     noise = noise_std_for_privacy * torch.randn(real_rep.shape).to(self.device)
                    #     real_rep = real_rep + noise

                    d_real = self.discriminator(real_rep)
                    dloss_real = -d_real.mean()
                    #y = d_real.new_full(size=d_real.size(), fill_value=1)
                    #dloss_real = F.binary_cross_entropy_with_logits(d_real, y)
                    dloss_real.backward()
                    
                    """
                    dloss_real.backward(retain_graph=True)
                    reg = 10 * compute_grad2(d_real, real_rep).mean()
                    reg.backward()
                    """

                    # On fake data
                    with torch.no_grad():
                        x_fake = self.generator(z)

                    x_fake.requires_grad_()
                    d_fake = self.discriminator(x_fake)
                    dloss_fake = d_fake.mean()
                    """
                    y = d_fake.new_full(size=d_fake.size(), fill_value=0)
                    dloss_fake = F.binary_cross_entropy_with_logits(d_fake, y)
                    """
                    dloss_fake.backward()
                    """
                    dloss_fake.backward(retain_graph=True)
                    reg = 10 * compute_grad2(d_fake, x_fake).mean()
                    reg.backward()
                    """
                    if eps!=None: self.discriminator.disable_hooks()
                    reg = 10 * self.wgan_gp_reg(real_rep, x_fake)
                    #reg.backward()
                    if eps!=None: self.discriminator.enable_hooks()

                    # Clip weights of discriminator ## taken from githubt issue
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-10, 10)

                    #self.discriminator_optm.step()
                    d_loss = dloss_fake + dloss_real
                    avg_d_loss += d_loss.item()
                    break
     
            avg_d_loss/=d_update

            toggle_grad(self.generator, True)
            toggle_grad(self.discriminator, False)
            self.generator.train()
            self.discriminator.train()
            self.generator_optm.zero_grad()
            z = torch.randn(batch_size, self.params['noise_dim']).to(self.device)
            fake = self.generator(z)
            if eps!=None: self.discriminator.disable_hooks() ## taken from https://github.com/pytorch/opacus/issues/31
            g_loss = -torch.mean(self.discriminator(fake))
            """
            d_fake = self.discriminator(fake)
            y = d_fake.new_full(size=d_fake.size(), fill_value=1)
            g_loss = F.binary_cross_entropy_with_logits(d_fake, y)
            """
            g_loss.backward()
            if eps!=None: self.discriminator.enable_hooks() ## see above
            self.generator_optm.step()

            history["d_loss"].append(avg_d_loss)
            history["g_loss"].append(g_loss.item())

            if iteration % 100 == 0:
                self.logger.info('[Iteration %d/%d] [%f] [D loss: %f] [G loss: %f] [%f]' % (
                    iteration, iterations, time.time()-t1, avg_d_loss, g_loss.item(), reg.item()
                ))
        #torch.save(self.generator.state_dict(), '{}/generator.dat'.format(self.params["root_dir"]))    

        return history          
    
    def synthesize(self, n, seq_len=24, batch_size=500):
        self.ae.model.decoder.eval()
        self.generator.eval()
        
        z = torch.randn(n, self.params['noise_dim']).to(self.device)
        hidden =self.generator(z)
        dynamics = self.ae.decode_data(hidden)
    
        return dynamics

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=self.device).view(batch_size, -1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg
    
    def eval_ae(self, dataset):
        batch_size = self.params["gan_batch_size"]
        idxs = list(range(len(dataset)))
        batch = DataSetIter(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())
        res = []
        h = []
        for batch_x, batch_y in batch:
            with torch.no_grad():
                sta = None
                dyn = batch_x["dyn"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)
                hidden = self.ae.encoder(sta, dyn, seq_len)
                dynamics = self.ae.decoder.generate_dynamics(hidden, seq_len[0])
                h.append(hidden)
                for i in range(len(dyn)):
                    #dyn = self.dynamic_processor.inverse_transform(dynamics[i]).values.tolist()
                    dyn = dynamics[i].tolist()
                    res.append(dyn)
        h = torch.cat(h, dim=0).cpu().numpy()
        assert len(h) == len(res)
        return res, h
    
# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
