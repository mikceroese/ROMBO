#################################################
# Program for using the bayesopt interface  	#
#                                               #
# The program loads a model trained over scores #
# for a few epochs and then tries to create one #
# that best fits the human feedback.            #
#                                               #
# Author: Miguel Marcos				            #
#################################################

# Import zone

from os.path import join
from os import getcwd, mkdir

import math
import numpy as np
import matplotlib.pyplot as plt
import logging as log
from datetime import datetime

import torch
import torch.nn as nn

import bayesopt
from bopt_dict import BayesOptDict
from bayesoptmodule import BayesOptContinuous

from powgan import Generator # The model is defined here
from powgan import clip_samples, samples_to_multitrack, write_sample
from pianoroll_gui import *

from utils import *

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Model instantiation

wd = getcwd() # Working directory
cp_path_gen = join(wd,'powgan_gen.pt') # Path to the generator checkpoint
gen = Generator()
load_checkpoint(cp_path_gen,gen)
if torch.cuda.is_available():
    gen = gen.cuda()
gen.eval()

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Prepare subject folder for results

date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S") # Date and time string for results storing
sd = join(wd,date_str) # Subject directory
mkdir(sd)

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Parameters

note_thresholds = [0.60828567, 0.55597573, 0.54794814]

params = {}
params['n_init_samples'] = 21 # Probably use a small number if you plan to use human feedback
params['init_method'] = 2 # Sobol
params['noise'] = 1e-2 # Default 1e-6, if human feedback is used, pump it up
params['n_iterations'] = 0
params['n_iter_relearn'] = 1
params['l_type'] = 'mcmc'
params['load_save_flag'] = 2 # 1 - Load, 2 - Save, 3 - Load and save
params['save_filename'] = join(sd,"musegan_bopt")+".txt"
params['verbose_level'] = 5
params['log_filename'] = join(sd,"musegan_bopt_log")+".txt"

n = 10 # Problem is 10-dimensional
true_n = 256 # True dimension of the model's input
lb = np.zeros((n,)) # Lower bounds
ub = np.ones((n,)) # Upper bounds


# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# In/Out

sf2_path = join(wd,"SGM.sf2") # Soundfont file, needed to create .wav files

results_name = join(sd,"results.npz")
sample_name = join(sd,'sample')
wav_name = sample_name+'.wav'
mid_sample_name = join(sd,'mid_sample')
mid_name = mid_sample_name+'.wav'

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Problem definition

class BayesOptPownGANgui(BayesOptContinuous):

    def __init__(self, n, mat_A):

        super().__init__(n)

        # Set maximum and minimum scores
        self.min_score = 0.0
        self.max_score = 10.0
        self.best_score = -1.0
        self.mid_score = -1.0

        self.current_iter = 1
        self.best_iter = 1

        # Set REMBO matrix and problem dimension
        self.mat_A = mat_A
        self.n = n

        # Precalculate variances for remapping
        # (See 'remap_query' below)
        self.vars = np.sum(np.square(mat_A),axis=0)
        self.stds = np.sqrt(self.vars)

        # Initialize GUI
        self.gui = PianorollGUI(wav_name, use_ref=True, ref_path=mid_name)

        # Show two random generations so the user knows what to expect
        with torch.no_grad():
            sample_one = self.generate_sample(np.random.uniform(size=n))
            sample_two = self.generate_sample(np.random.uniform(size=n))
            m_one = samples_to_multitrack(tensor_to_np(sample_one))
            write_sample(m_one,sf2_path,sample_name)
            m_two = samples_to_multitrack(tensor_to_np(sample_two))
            write_sample(m_two,sf2_path,mid_sample_name)
            self.gui.show_current_sample(m_one)
            self.gui.show_target_sample(m_two)
        
        self.gui.wait_for_input()

        mid_query = np.zeros(n)
        mid_query[np.arange(0,n,2)] = 1 - 1e-6
        self.mid_score = self.evaluateSample(mid_query)

    def uniform_to_normal(self,query):

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

    def remap_query(self,query):

        # Map query to the high-dimensional space
        q = np.matmul(query,self.mat_A)

        # Queries are mapped to a ~N(0,1) distribution
        # using the Box-Muller transform.
        # Therefore, the vector-matrix product (q)
        # results in a vector of sums of Normal distributions,
        # which are Normal themselves.

        # Each component i of the resulting vector will follow a Normal
        # distribution with mean 0 and variance V[i] equal
        # to the sum of the squares of the i-th column of mat_A.

        # Thus, we can remap each of this components to ~N(0,1)
        # using the precalculated standard deviations:

        q = np.divide(q,self.stds)

        # Clip the resulting query
        q = np.clip(q,-4.0,4.0)

        return q

    def generate_sample(self,query):

        q = self.uniform_to_normal(query)
        q = self.remap_query(q)
        q = np_to_tensor(q)
        if torch.cuda.is_available():
            q = q.cuda()
        sample = gen(q)
        sample = clip_samples(sample,note_thresholds)
        return sample

    def evaluateSample(self,query):

        with torch.no_grad():
            sample = self.generate_sample(query)

        m = samples_to_multitrack(tensor_to_np(sample))
        write_sample(m,sf2_path,sample_name)

        self.gui.show_current_sample(m)
        self.gui.update_sample_text(self.current_iter)

        score = self.gui.wait_for_input()

        if self.best_score <= score:
            self.best_score = score
            write_sample(m,sf2_path,mid_sample_name)
            self.gui.show_target_sample(m)
            self.gui.update_ref_text(self.current_iter,score)
            self.best_iter = self.current_iter

        self.current_iter+=1

        score = 1-(score-self.min_score)/(self.max_score-self.min_score)
        return score    

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Optimization

mat_A = np.random.randn(n,true_n)

bo = BayesOptPowGANgui(n, mat_A)

bo.params = params
bo.lower_bound = lb 
bo.upper_bound = ub

mvalue, x_out, error = bo.optimize()
print("Finished initial models.")
print("Result", mvalue, "at", x_out)

with torch.no_grad():
    sample = bo.generate_sample(x_out)
    m = samples_to_multitrack(tensor_to_np(sample))
    write_sample(m,sf2_path,sample_name)

bo_dict = BayesOptDict(params['save_filename'])
for i in range(21):
    mid_sample = np.round(np.random.uniform(size=n),3)
    mid_sample[np.arange(0,n,2)] = 1 - 1e-6
    bo_dict.add_sample(mid_sample,bo.mid_score)

bo_dict.set_init_samples(params['n_init_samples']+21)
bo_dict.set_num_iter(42)
bo_dict.save_txt(params['save_filename'])

params['n_init_samples'] += 21
params['n_iterations'] = 42
params['load_save_flag'] = 3 # 1 - Load, 2 - Save, 3 - Load and save
params['load_filename'] = params['save_filename']

bo.params = params

mvalue, x_out, error = bo.optimize()
print("Finished iterations.")
print("Result", mvalue, "at", x_out)
print("Final score:",(1-mvalue)*10)
bo.gui.close_window()

with torch.no_grad():
    sample = bo.generate_sample(x_out)
    m = samples_to_multitrack(tensor_to_np(sample))
    write_sample(m,sf2_path,sample_name)

with torch.no_grad():
    mid_sample = bo.generate_sample(mid_sample)
    mid_m = samples_to_multitrack(tensor_to_np(mid_sample))
    write_sample(mid_m,sf2_path,mid_sample_name)

gui = PianorollGUI(wav_name, use_ref=True, ref_path=mid_name) 
gui.show_current_sample(m)
gui.update_sample_text(bo.best_iter)
gui.show_target_sample(mid_m)
gui.update_ref_text(1,(1-bo.mid_score)*(bo.max_score-bo.min_score)+bo.min_score)
gui.wait_for_input()

np.savez(results_name,sample=sample,score=(1-mvalue)*10,best_iter=bo.best_iter,query=x_out,rembo=mat_A)
print("Best results saved at",results_name)
