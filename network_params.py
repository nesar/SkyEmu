#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 2019

@author: cguilloteau
"""

DataDir = '../Data/'

batch = 32
kernel_size = 4
n_conv = 1
filters = 16
interm_dim = 128
latent_dim = 12
epochs = 1000

epsilon_mean = 0.
epsilon_std = 1e-4

learning_rate = 1e-4
decay_rate = 1e-1

ntrain = 512
ntest = 64
nx = 32
ny = 32
