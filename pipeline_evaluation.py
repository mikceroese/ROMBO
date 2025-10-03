#################################################
# Test routine for a Random Embedding Bayesian  #
# Optimization ablation study.                	#
#                                               #
# Author: Miguel Marcos				            #
#################################################

# Import zone

from os.path import join
from os import getcwd

import math
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch
import scipy.stats as sp

from bayesoptmodule import BayesOptContinuous

from powgan import Generator, Encoder, VarAutoencoder # The model is defined here
from powgan import clip_samples, write_sample, samples_to_multitrack
from powgan import used_pitch_classes, tonal_distance

from utils import *

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Model instantiation

wd = getcwd() # Working directory

cp_path_gen = join(wd,'museGANgen_DBG_chroma_256_25k.pt') # Path to the generator checkpoint
gen = Generator()
load_checkpoint(cp_path_gen,gen)
if torch.cuda.is_available():
    gen = gen.cuda()
gen.eval()

enc_mf = Encoder()
dec_mf = Generator() #Not used, just here to prevent weight corruption during loading
enc_asl = Encoder()
dec_asl = Generator() #Not used, just here to prevent weight corruption during loading
if torch.cuda.is_available():
    enc_mf = enc_mf.cuda()
    dec_mf = dec_mf.cuda()
    enc_asl = enc_asl.cuda()
    dec_asl = dec_asl.cuda()
enc_mf.eval()
enc_asl.eval()

vae_path_mf = join(wd,'autoencoder_DBG_chroma256_mf.pt') # Path to the vae checkpoint
vae_path_asl = join(wd,'autoencoder_DBG_chroma256_asl.pt') # Path to the vae checkpoint
vae_mf = VarAutoencoder(enc_mf,dec_mf)
vae_asl = VarAutoencoder(enc_asl,dec_asl)
load_checkpoint(vae_path_asl,vae_asl)
load_checkpoint(vae_path_mf,vae_mf)
if torch.cuda.is_available():
    vae_mf = vae_mf.cuda()
    vae_asl = vae_asl.cuda()
vae_mf.eval()
vae_asl.eval()

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Class definitions

class AssymetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
      super(AssymetricLoss, self).__init__()

      self.gamma_neg = gamma_neg
      self.gamma_pos = gamma_pos
      self.clip = clip
      self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
      self.eps = eps

      # For normalizaton
      self.zero = 0.444895
      self.one = 9.730950

    def forward(self, x, y):
      """
      Parameters
      ----------
      x: input logits
      y: targets
      """

      # Calculating probabilities
      x_sigmoid = torch.sigmoid(x)
      xs_pos = x_sigmoid
      xs_neg = 1 - x_sigmoid

      # Asymmetric Clipping
      if self.clip is not None and self.clip > 0:
          xs_neg = (xs_neg + self.clip).clamp(max=1)

      # Basic CE calculation
      los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
      los_neg = (1-y) * torch.log(xs_neg.clamp(min=self.eps))
      loss = los_pos + los_neg

      # Asymmetric Focusing
      if self.gamma_neg > 0 or self.gamma_pos > 0:
        if self.disable_torch_grad_focal_loss:
          torch.set_grad_enabled(False)
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y) # pt = p if t > 0 else 1 - p
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.disable_torch_grad_focal_loss:
          torch.set_grad_enabled(True)
        loss *= one_sided_w

      return (-loss.sum([1,2,3]).mean() - self.zero)/(self.one-self.zero)

class BayesOptMuseGAN(BayesOptContinuous):

    def __init__(self, rembo_dim, target, mat_A=None, clipping = False, loss='$\mathcal{L}_{MF}$', dec='REMBO'):
       
        super().__init__(rembo_dim)

        self.target_sample = target
        self.clipping = clipping

        self.best_asl = 100000.0
        self.asl_curve = []
        self.best_mf = 100000.0
        self.mf_curve = []

        self.target_np = self.target_sample.cpu().detach().numpy()
        self.upcs = used_pitch_classes(self.target_np,verbose=False)
        self.td = tonal_distance(self.target_np,verbose=False)
        self.sr = np.mean(self.target_np,axis=(0,2,3))
        self.target_features = np.concatenate([self.upcs,self.td,self.sr])

        self.n = rembo_dim
        if mat_A is None:
            self.mat_A = np.random.randn(self.n,true_dim)
        else:
            self.mat_A = mat_A
        Q , _ = np.linalg.qr(self.mat_A.T)
        self.mat_Q = Q.T

        self.asl = AssymetricLoss(gamma_neg=16,gamma_pos=4,disable_torch_grad_focal_loss=True)
        if torch.cuda.is_available():
            self.asl = self.asl.cuda()

        self.loss = loss
        self.dec = dec

    def uniform_to_normal(self,query):

        # IMPORTANT: Box-Muller needs an even number of variables
        # If the program outputs an index-out-of-bounds error, it's
        # probably because of this.

        # Box-Muller transform
        even = np.arange(0,query.shape[-1],2)
        q_even = query[even]
        q_even[q_even==0] += 1e-6
        Rs = np.sqrt(-2*np.log(q_even))
        thetas = 2*math.pi*(query[even+1])
        cos = np.cos(thetas)
        sin = np.sin(thetas)
        query = np.stack([Rs*cos,Rs*sin],-1).flatten()

        return query

    def to_high_dim(self,query):

        if self.dec == 'ND$_{MF}$':
            q = self.uniform_to_normal(query)
            q = np_to_tensor(q).view(1,-1)
            q = vae_mf.get_highdim(q)
        elif self.dec == 'ND$_{AS}$':
            q = self.uniform_to_normal(query)
            q = np_to_tensor(q).view(1,-1)
            q = vae_asl.get_highdim(q)
        elif self.dec == 'ROMBO':
            q = self.uniform_to_normal(query)
            q = np.matmul(q,self.mat_Q)
            q = np_to_tensor(q)
        elif self.dec == 'REMBO':
            q = (query*2-1)*np.sqrt(self.n)
            q = np.matmul(q,self.mat_A)
            q = np_to_tensor(q)
        else: # self.dec == 'RANDOM':
            q = torch.normal(0,1,size=true_dim)
        
        return q

    def generate_sample(self,query):
        
        q = self.to_high_dim(query)
        if torch.cuda.is_available():
            q = q.cuda()
        sample = gen(q)
        if self.clipping:
            sample = clip_samples(sample,note_thresholds)
        return sample
 
    def evaluateSample(self,query):

        score = None

        with torch.no_grad():
            sample = self.generate_sample(query)
            mf_loss = self.compare_features(sample)
            if mf_loss < self.best_mf:
                self.best_mf = mf_loss
            asl_loss = self.asl(sample,self.target_sample).item()
            if asl_loss < self.best_asl:
                self.best_asl = asl_loss

            self.mf_curve.append(self.best_mf)
            self.asl_curve.append(self.best_asl)

            if self.loss=='$\mathcal{L}_{MF}$':
                score = mf_loss
            else:
                score = asl_loss 


        return score    

    def compare_features(self,sample):
        sample_np = sample.cpu().detach().numpy()
        upcs = used_pitch_classes(sample_np,verbose=False)
        td = tonal_distance(sample_np,verbose=False)
        sr = np.mean(sample_np,axis=(0,2,3))
        sample_features = np.concatenate([upcs,td,sr])
        score = np.linalg.norm(sample_features-self.target_features)

        return score
        
# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Parameters

loss_names = {"$\mathcal{L}_{MF}$":"Music Features L2",
              "$\mathcal{L}_{AS}$":"Assymetric Loss",
              }

losses = ["$\mathcal{L}_{MF}$",
          "$\mathcal{L}_{AS}$"
          ]

decs = ['RANDOM',
        'REMBO',
        'ROMBO',
        #'ND$_{MF}$',
        #'ND$_{AS}$'
        ]
        
n_targets = 20
colors = ['r','g','y','b']
rembo_dim = 10
clipping = True
n_init = 10
n_samples = 60

results = {}
for l in losses:
    results[l] = {}
    for d in decs:
        results[l][d] = {}

note_thresholds = [0.60828567, 0.55597573, 0.54794814]
true_dim = 256 # True dimension of the model's input

params = {}
params['init_method'] = 2 # Sobol
params['n_iter_relearn'] = 1
params['l_type'] = 'mcmc'
params['verbose_level'] = 0 # Quiet
params['n_init_samples'] = n_init
params['n_iterations'] = n_samples

sf2_path = "SGM.sf2"

loss_mat = np.zeros((len(losses),len(decs),len(losses)))

seed = 1743007583 #seed = int(time.time())
print("Seed:",seed)
np.random.seed(seed)

bar = tqdm(total=len(losses)*len(decs)*n_targets)

qs = np.random.randn(n_targets,true_dim)
As = np.random.randn(n_targets,rembo_dim,true_dim)

lb = np.zeros((rembo_dim,)) # Lower bounds
ub = np.ones((rembo_dim,)) # Upper bounds

fig1 = plt.figure()
subfigs1 = fig1.subfigures(2,1)
axs1 = []
axs1_1 = subfigs1[0].subplots(1,len(losses), sharey=True)
axs1.append(axs1_1)
axs1_2 = subfigs1[1].subplots(1,len(losses), sharey=True)
axs1.append(axs1_2)
fig1.suptitle("Historic best through optimization")
for i in range(len(losses)):
    axs1[i][0].set_ylabel(losses[i])
for i in range(len(losses)):
    axs1[0][i].set_xlabel("Iterations\noptimizing "+losses[i])

fig2 = plt.figure()
subfigs2 = fig2.subfigures(2,1)
axs2 = []
axs2_1 = subfigs2[0].subplots(1,len(losses), sharey=True)
axs2_1[0].set_yscale('log')
axs2.append(axs2_1)
axs2_2 = subfigs2[1].subplots(1,len(losses), sharey=True)
axs2_2[0].set_yscale('log')
axs2.append(axs2_2)
fig2.suptitle("Historic best through optimization")
for i in range(len(losses)):
    axs2[i][0].set_ylabel(losses[i])
for i in range(len(losses)):
    axs2[0][i].set_xlabel("Iterations\noptimizing "+losses[i])

for i_l, loss in enumerate(losses):

    for i_d, dec in enumerate(decs):

        color = colors[i_d]
        symbol = ','
        label = dec

        mf_curves = []
        asl_curves = []

        for t in range(n_targets):

            q = qs[t]
            q = np_to_tensor(q)
            if torch.cuda.is_available():
                q = q.cuda()
            target_sample = gen(q)
            target_sample = clip_samples(target_sample,note_thresholds)

            """
            target_name = "samples/target_"+str(t)
            m = samples_to_multitrack(tensor_to_np(target_sample))
            write_sample(m,sf2_path,target_name,write_wav=False)
            """

            bo = BayesOptMuseGAN(rembo_dim,target_sample,mat_A=As[t],clipping=clipping,loss=loss,dec=dec)

            bo.params = params
            bo.upper_bound = ub
            bo.lower_bound = lb

            mvalue, x_out, error = bo.optimize()
            if error:
                print("Something went wrong,skipping")
                continue

            """
            sample = bo.generate_sample(x_out)
            target_name = "samples/closest_"+str(t)+"_"+loss+"_"+dec
            m = samples_to_multitrack(tensor_to_np(sample))
            write_sample(m,sf2_path,target_name,write_wav=False)
            """

            mf_curves.append(bo.mf_curve)
            asl_curves.append(bo.asl_curve)

            bar.update()

        res = np.asarray(mf_curves)
        num_runs = res.shape[0]
        res_mean = np.mean(res,axis=0)
        res_std = np.std(res,axis=0)

        loss_mat[i_l][i_d][0] = res_mean[-1]

        results[loss][dec]['mf']={'mean':res_mean.tolist(),'std':res_std.tolist()}
    
        it = range(res_mean.shape[0])
        t_limits = sp.t.interval(0.95,num_runs) / np.sqrt(num_runs)

        if dec == 'REMBO' or dec == 'ROMBO' or dec == 'RANDOM':

            axs1[0][i_l].plot(it,res_mean, color+symbol+'-', linewidth=2, label=label)
            axs1[0][i_l].fill(np.concatenate([it,it[::-1]]),
                    np.concatenate([res_mean + t_limits[0] * res_std,
                                    (res_mean + t_limits[1] * res_std)[::-1]]),
                    alpha=.3, fc=color, ec='None')
            
            res = np.asarray(asl_curves)
            res_mean = np.mean(res,axis=0)
            res_std = np.std(res,axis=0)

            loss_mat[i_l][i_d][1] = res_mean[-1]

            axs1[1][i_l].plot(it,res_mean, color+symbol+'-', linewidth=2, label=label)
            axs1[1][i_l].fill(np.concatenate([it,it[::-1]]),
                    np.concatenate([res_mean + t_limits[0] * res_std,
                                    (res_mean + t_limits[1] * res_std)[::-1]]),
                alpha=.3, fc=color, ec='None')
            
        if dec != 'REMBO' and dec != 'RANDOM':

            res = np.asarray(mf_curves)
            res_mean = np.mean(res,axis=0)
            res_std = np.std(res,axis=0)

            axs2[0][i_l].plot(it,res_mean, color+symbol+'-', linewidth=2, label=label)
            axs2[0][i_l].fill(np.concatenate([it,it[::-1]]),
                    np.concatenate([res_mean + t_limits[0] * res_std,
                                    (res_mean + t_limits[1] * res_std)[::-1]]),
                    alpha=.3, fc=color, ec='None')
            
            res = np.asarray(asl_curves)
            res_mean = np.mean(res,axis=0)
            res_std = np.std(res,axis=0)

            loss_mat[i_l][i_d][1] = res_mean[-1]

            axs2[1][i_l].plot(it,res_mean, color+symbol+'-', linewidth=2, label=label)
            axs2[1][i_l].fill(np.concatenate([it,it[::-1]]),
                    np.concatenate([res_mean + t_limits[0] * res_std,
                                    (res_mean + t_limits[1] * res_std)[::-1]]),
                alpha=.3, fc=color, ec='None')

        results[loss][dec]['asl']={'mean':res_mean.tolist(),'std':res_std.tolist()}

handles, labels = axs1[0][0].get_legend_handles_labels()
fig1.legend(handles,labels)
handles, labels = axs2[0][0].get_legend_handles_labels()
fig2.legend(handles,labels)
    
bar.close()
print(loss_mat)
with open("results.json", "w") as f:
    json.dump(results,f)

plt.show(block=True)