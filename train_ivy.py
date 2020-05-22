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
model, optimizer, metric_logger = train(opt, data, device, int(opt.k))

### 3.2 Evalutes Train + Test Error, and Saves Model
loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(opt, model, data, 'train', device)
loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device)

### 3.2 Evalutes Train + Test Error, and Saves Model
loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device)
print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))

### 3.3 Saves Model
if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
    model_state_dict = model.module.cpu().state_dict()
else:
    model_state_dict = model.cpu().state_dict()

torch.save({
	'split':opt.k,
    'opt': opt,
    'epoch': opt.niter+opt.niter_decay,
    'data': data,
    'model_state_dict': model_state_dict,
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metric_logger}, 
    os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%s.pt' % (opt.model_name, opt.k)))


### 3.3 Saves Model
#pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))