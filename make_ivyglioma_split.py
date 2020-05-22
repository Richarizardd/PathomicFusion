### data_loaders.py
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

# Env
from networks import define_net
from utils import getCleanAllDataset, getCleanIvyGlioma
import torch
from torchvision import transforms
from options import parse_gpuids

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot_ivy', type=str, default='./data/IvyGlioma/', help="datasets")
    parser.add_argument('--dataroot_tcga', type=str, default='./data/TCGA_GBMLGG/', help="datasets")
    parser.add_argument('--which_structures', type=str, default='default', help="parse ivyglioma")
    parser.add_argument('--bulk', type=int, default=1, help="to average")
    parser.add_argument('--use_vgg_features', type=int, default=1, help='use model')
    parser.add_argument('--graph_feat_type', type=str, default='cpc', help="graph features to use")

    parser.add_argument('--roi_dir_ivy', type=str, default='IvyGBMA_st', help="to average")
    parser.add_argument('--roi_dir_tcga', type=str, default='all_st_patches_512', help="to average")
    parser.add_argument('--model_name', type=str, default='path', help='mode')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/TCGA_GBMLGG/', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='surv_15_rnaseq', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--k', type=str, default='15', help='mode')

    parser.add_argument('--mode', type=str, default='path', help='mode')
    parser.add_argument('--task', type=str, default='surv', help='surv | grad')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--batch_size', type=int, default=1, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--init_type', type=str, default='none', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')

    opt = parser.parse_known_args()[0]
    opt = parse_gpuids(opt)
    return opt

opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
ivy, tcga, genes_overlap = getCleanIvyGlioma(dataroot=opt.dataroot_ivy)
metadata_ivy, all_dataset_ivy = ivy
metadata_tcga, all_dataset_tcga = tcga

### Creates a mapping from Ivy Glioma Tumor ID -> Image ROI
img_fnames_ivy = os.listdir(os.path.join(opt.dataroot_ivy, opt.roi_dir_ivy))
pat2img_ivy = {}
for pat, img_fname in zip(["-".join(img_fname.split('-')[:3]) for img_fname in img_fnames_ivy], img_fnames_ivy):
    if pat not in pat2img_ivy.keys(): pat2img_ivy[pat] = []
    pat2img_ivy[pat].append(img_fname)

### Creates a mapping from TCGA ID -> Image ROI
img_fnames_tcga = os.listdir(os.path.join(opt.dataroot_tcga, opt.roi_dir_tcga))
pat2img_tcga = {}
for pat, img_fname in zip([img_fname[:12] for img_fname in img_fnames_tcga], img_fnames_tcga):
    if pat not in pat2img_tcga.keys(): pat2img_tcga[pat] = []
    pat2img_tcga[pat].append(img_fname)


### Dictionary file containing split information
data_dict = {}
data_dict['data_ivy'] = all_dataset_ivy
data_dict['data_tcga'] = all_dataset_tcga

### Extracting K-Fold Splits
pnas_splits = pd.read_csv(opt.dataroot_tcga+'pnas_splits.csv')
pnas_splits.columns = ['TCGA ID']+[str(k) for k in range(1, 16)]
pnas_splits.index = pnas_splits['TCGA ID']
pnas_splits = pnas_splits.drop(['TCGA ID'], axis=1)

### get path_feats
def get_vgg_features(model, device, img_path):
    if model is None:
        return img_path
    else:
        x_path = Image.open(img_path).convert('RGB')
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        x_path = torch.unsqueeze(normalize(x_path), dim=0)
        features, hazard = model(x_path=x_path.to(device))
        return features.cpu().detach().numpy()

### method for constructing aligned TCGA
def getAlignedMultimodalData_TCGA(opt, model, device, all_dataset, pat_split, pat2img, genes_overlap):
    x_patname, x_path, x_grph, x_omic, e, t = [], [], [], [], [], []

    for pat_name in pat_split:
        if pat_name not in all_dataset.index: continue

        for img_fname in pat2img[pat_name]:
            grph_fname = img_fname.rstrip('.png')+'.pt'
            assert grph_fname in os.listdir(os.path.join(opt.dataroot_tcga, '%s_%s' % (opt.roi_dir_tcga, opt.graph_feat_type)))
            assert all_dataset[all_dataset['TCGA ID'] == pat_name].shape[0] == 1

            x_patname.append(pat_name)
            x_path.append(get_vgg_features(model, device, os.path.join(opt.dataroot_tcga, opt.roi_dir_tcga, img_fname)))
            x_grph.append(os.path.join(opt.dataroot_tcga, '%s_%s' % (opt.roi_dir_tcga, opt.graph_feat_type), grph_fname))
            x_omic.append(np.array(all_dataset[all_dataset['TCGA ID'] == pat_name][genes_overlap]))
            e.append(int(all_dataset[all_dataset['TCGA ID']==pat_name]['censored']))
            t.append(int(all_dataset[all_dataset['TCGA ID']==pat_name]['Survival months']))

    return x_patname, x_path, x_grph, x_omic, e, t

### method for constructing aligned Ivy
def getAlignedMultimodalData_Ivy(opt, model, device, all_dataset, pat_split, pat2img, genes_overlap):
    x_patname, x_path, x_grph, x_omic, e, t = [], [], [], [], [], []

    for pat_name in pat_split:
        if pat_name not in all_dataset.index: continue

        for img_fname in pat2img[pat_name]:

            grph_fname = img_fname.rstrip('.jpg')+'.pt'
            assert grph_fname in os.listdir(os.path.join(opt.dataroot_ivy, '%s_%s' % (opt.roi_dir_ivy, opt.graph_feat_type), 'pt_bi'))
            assert all_dataset[all_dataset['tumor_name'] == pat_name].shape[0] == 1

            x_patname.append(pat_name)
            x_path.append(get_vgg_features(model, device, os.path.join(opt.dataroot_ivy, opt.roi_dir_ivy, img_fname)))
            x_grph.append(os.path.join(opt.dataroot_ivy, '%s_%s' % (opt.roi_dir_ivy, opt.graph_feat_type), grph_fname))
            x_omic.append(np.array(all_dataset[all_dataset['tumor_name'] == pat_name][genes_overlap]))
            e.append(int(all_dataset[all_dataset['tumor_name']==pat_name]['censored']))
            t.append(int(all_dataset[all_dataset['tumor_name']==pat_name]['Survival months']))

    return x_patname, x_path, x_grph, x_omic, e, t

model = None
if opt.use_vgg_features:
    load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%s.pt' % (opt.model_name, opt.k))
    model_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = model_ckpt['model_state_dict']
    if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata
    model = define_net(opt, None)
    if isinstance(model, torch.nn.DataParallel): model = model.module
    print('Loading the model from %s' % load_path)
    model.load_state_dict(model_state_dict)
    model.eval()

pat_train = pnas_splits.index[pnas_splits[opt.k] == 'Train']
pat_test = list(all_dataset_ivy['tumor_name'])

out = getAlignedMultimodalData_TCGA(opt, model, device, all_dataset_tcga, pat_train, pat2img_tcga, genes_overlap)
train_x_patname, train_x_path, train_x_grph, train_x_omic, train_e, train_t = out
out = getAlignedMultimodalData_Ivy(opt, model, device, all_dataset_ivy, pat_test, pat2img_ivy, genes_overlap)
test_x_patname, test_x_path, test_x_grph, test_x_omic, test_e, test_t = out

train_x_omic, train_e, train_t = np.array(train_x_omic).squeeze(axis=1), np.array(train_e, dtype=np.float64), np.array(train_t, dtype=np.float64)
test_x_omic, test_e, test_t = np.array(test_x_omic).squeeze(axis=1), np.array(test_e, dtype=np.float64), np.array(test_t, dtype=np.float64)
    
scaler = preprocessing.StandardScaler().fit(train_x_omic)
train_x_omic = scaler.transform(train_x_omic)
test_x_omic = scaler.transform(test_x_omic)

train_data = {'x_patname': train_x_patname,
              'x_path':np.array(train_x_path),
              'x_grph':train_x_grph,
              'x_omic':train_x_omic,
              'e':np.array(train_e, dtype=np.float64), 
              't':np.array(train_t, dtype=np.float64),
              'g':np.array(train_t, dtype=np.float64)}

test_data = {'x_patname': test_x_patname,
             'x_path':np.array(test_x_path),
             'x_grph':test_x_grph,
             'x_omic':test_x_omic,
             'e':np.array(test_e, dtype=np.float64),
             't':np.array(test_t, dtype=np.float64),
             'g':np.array(train_t, dtype=np.float64)}

dataset = {'train':train_data, 'test':test_data}
data_dict['ivy_split'] = dataset

pickle.dump(data_dict, open('%s/splits/%s_%s_%d%s.pkl' % (opt.dataroot_ivy, opt.roi_dir_ivy, opt.k, opt.use_vgg_features, '_rnaseq'), 'wb'))