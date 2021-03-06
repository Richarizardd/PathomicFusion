import os
import logging
import numpy as np
import random
import pickle

import torch

# Env
from networks import define_net
from data_loaders import *
from options import parse_args
from train_test import train, test


### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
data_ivy_path = '%s/splits/%s_%s_%d%s.pkl' % (opt.dataroot, opt.roi_dir_ivy, opt.k, opt.use_vgg_features, '_rnaseq')
print("Loading %s" % data_ivy_path)
data_ivy = pickle.load(open(data_ivy_path, 'rb'))
data = data_ivy['ivy_split']

### 3. Sets-Up Main Loop
print("*******************************************")
print("************* Ivy Validation **************")
print("*******************************************")
load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%s.pt' % (opt.model_name, opt.k))
model_ckpt = torch.load(load_path, map_location=device)

#### Loading Env
model_state_dict = model_ckpt['model_state_dict']
if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

model = define_net(opt, None)
if isinstance(model, torch.nn.DataParallel): model = model.module

print('Loading the model from %s' % load_path)
model.load_state_dict(model_state_dict)
model.eval()


### 3.2 Evalutes Train + Test Error, and Saves Model
loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device)
print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))

### 3.3 Saves Model
pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%s_%s_pred_test.pkl' % (opt.model_name, opt.roi_dir_ivy, opt.k)), 'wb'))