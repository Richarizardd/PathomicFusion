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
import torch
from torchvision import transforms
from options import parse_gpuids

from tqdm import tqdm

# python make_kirc_splits.py --roi_dir KIRC_st --use_vgg_features 1 --exp_name surv_15 --gpu_ids 1

def getCleanKIRC(dataroot='./data/TCGA_KIRC/', rnaseq_cutoff=3922, cnv_cutoff=7.0, mut_cutoff=5.0, use_ag=False):
    ### Clinical variables
    clinical = pd.read_table(os.path.join(dataroot, './kirc_tcga_pan_can_atlas_2018_clinical_data.tsv'), index_col=2)
    clinical.index.name = None
    clinical = clinical[['Center of sequencing', 'Overall Survival Status', 'Overall Survival (Months)', 'Diagnosis Age', 'Sex']].copy()
    clinical = clinical.rename(columns={'Center of sequencing':'CoS', 'Overall Survival Status':'censored', 'Overall Survival (Months)':'OS_month', 'Diagnosis Age':'Age'})
    clinical['Sex'] = clinical['Sex'].replace({'Male':0, 'Female': 1})
    clinical['censored'] = clinical['censored'].replace('LIVING', 0) # actually uncensored
    clinical['censored'] = clinical['censored'].replace('DECEASED', 1) # actually uncensored
    clinical['train'] = 0
    train_cohort = list(clinical['CoS'].value_counts().index[0:2]) + list(clinical['CoS'].value_counts().index[-16:])
    clinical.loc[clinical['CoS'].isin(train_cohort), 'train'] = 1
    clinical = clinical.sort_values(['train', 'CoS'], ascending=False)

    ### Select RNAseq Features
    rnaseq = pd.read_table(os.path.join(dataroot, 'data_RNA_Seq_v2_mRNA_median_Zscores.txt'), index_col=0)
    rnaseq = rnaseq[rnaseq.index.notnull()]
    rnaseq = rnaseq.drop(['Entrez_Gene_Id'], axis=1)
    rnaseq.index.name = None
    rnaseqDEGs = pd.read_csv(os.path.join(dataroot, 'dataDEGs_kirc.csv'), index_col=0)
    rnaseqDEGs = rnaseqDEGs.sort_values(['PValue', 'logFC'], ascending=False)
    rnaseq_cutoff = min(rnaseqDEGs.shape[0], rnaseq_cutoff)
    rnaseq = rnaseq.loc[rnaseq.index.intersection(rnaseqDEGs.head(rnaseq_cutoff).index)].T
    rnaseq.columns = [g+"_rnaseq" for g in rnaseq.columns]

    ### Select CNV Features
    cnv = pd.read_table(os.path.join(dataroot, 'data_CNA.txt'), index_col=0)
    cnv = cnv[cnv.index.notnull()]
    cnv = cnv.drop(['Entrez_Gene_Id'], axis=1)
    cnv.index.name = None
    cnv_freq = pd.read_table(os.path.join(dataroot, 'CNA_Genes.txt'), index_col=0)
    cnv_freq = cnv_freq[['CNA', 'Profiled Samples', 'Freq']]
    cnv_freq['Freq'] = cnv_freq['Freq'].str.rstrip('%').astype(float)
    cnv_cutoff = cnv_freq.shape[0] if isinstance(cnv_cutoff, str) else cnv_cutoff
    cnv_freq = cnv_freq[cnv_freq['Freq'] >= cnv_cutoff]
    cnv = cnv.loc[cnv.index.intersection(cnv_freq.index)].T
    cnv.columns = [g+"_cnv" for g in cnv.columns]
                                 
    mut = clinical[['OS_month']].copy()
    for tsv in os.listdir(os.path.join(dataroot, 'muts')):
        if tsv.endswith('.tsv'):
            mut_samples = pd.read_table(os.path.join(dataroot, 'muts', tsv))['Patient ID']
            mut_gene = tsv.split('_')[2].rstrip('.tsv')+'_mut'
            mut[mut_gene] = 0
            mut.loc[mut.index.isin(mut_samples), mut_gene] = 1
    mut = mut.drop(['OS_month'], axis=1)
    
    all_dataset = clinical.join(rnaseq.join(cnv, how='inner').join(mut, how='inner'), how='inner')
    all_dataset.index = all_dataset.index.str[:-3]
    splits = pd.read_csv(os.path.join(dataroot, 'kirc_splits.csv'), index_col=0)
    all_dataset = all_dataset.loc[splits.index]

    metadata = ['CoS', 'censored', 'OS_month', 'train']
    if use_ag is False:
        metadata.extend(['Age', 'Sex'])

    return metadata, all_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data/TCGA_KIRC/', help="datasets")
    parser.add_argument('--roi_dir', type=str, default='KIRC_st')
    parser.add_argument('--graph_feat_type', type=str, default='cpc', help="graph features to use")

    parser.add_argument('--rnaseq_cutoff', type=int, default=240)
    parser.add_argument('--cnv_cutoff', type=float, default=7.0)
    parser.add_argument('--mut_cutoff', type=float, default=5.0)
    parser.add_argument('--use_vgg_features', type=int, default=0)
    parser.add_argument('--use_rnaseq', type=int, default=0)
    parser.add_argument('--use_ag', type=int, default=0)

    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/TCGA_KIRC/', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='surv_15_rnaseq_3922', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='path', help='mode')
    parser.add_argument('--model_name', type=str, default='path', help='mode')
    parser.add_argument('--task', type=str, default='surv', help='surv | grad')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--batch_size', type=int, default=32, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--init_type', type=str, default='none', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')

    opt = parser.parse_known_args()[0]
    opt = parse_gpuids(opt)
    return opt

opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
metadata, all_dataset = getCleanKIRC(opt.dataroot, opt.rnaseq_cutoff, opt.cnv_cutoff, opt.mut_cutoff, opt.use_ag)

### Creates a mapping from TCGA ID -> Image ROI
img_fnames = os.listdir(os.path.join(opt.dataroot, opt.roi_dir))
pat2img = {}
for pat, img_fname in zip([img_fname[:12] for img_fname in img_fnames], img_fnames):
    if pat not in pat2img.keys(): pat2img[pat] = []
    if int(img_fname.split('_')[2]) < 3:
        pat2img[pat].append(img_fname)


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

### method for constructing aligned
def getAlignedMultimodalData(opt, model, device, all_dataset, pat_split, pat2img):
    x_patname, x_path, x_grph, x_omic, e, t, g = [], [], [], [], [], [], []

    for pat_name in pat_split:
        if pat_name not in all_dataset.index: continue

        for img_fname in pat2img[pat_name]:
            grph_fname = img_fname.rstrip('.png')+'.pt'
            assert grph_fname in os.listdir(os.path.join(opt.dataroot, '%s_%s' % (opt.roi_dir, opt.graph_feat_type), 'pt_bi'))
            assert all_dataset[all_dataset.index == pat_name].shape[0] == 1

            x_patname.append(img_fname)
            x_path.append(get_vgg_features(model, device, os.path.join(opt.dataroot, opt.roi_dir, img_fname)))
            x_grph.append(os.path.join(opt.dataroot, '%s_%s' % (opt.roi_dir, opt.graph_feat_type), 'pt_bi', grph_fname))
            #x_grph.append('NaN')
            x_omic.append(np.array(all_dataset[all_dataset.index == pat_name].drop(metadata, axis=1)))
            e.append(int(all_dataset[all_dataset.index==pat_name]['censored']))
            t.append(int(all_dataset[all_dataset.index==pat_name]['OS_month']))
            g.append(-1)
            #g.append(int(all_dataset[all_dataset.index==pat_name]['Grade']))

    return x_patname, x_path, x_grph, x_omic, e, t, g

### Split Information
cv_splits = {}
splits = pd.read_csv(os.path.join(opt.dataroot, 'kirc_splits.csv'), index_col=0)
splits.columns = [str(k) for k in range(1, 16)]

for k in tqdm(splits.columns):
    print('Creating Split %s' % k)
    pat_train = splits.index[splits[k] == 'Train']
    pat_test = splits.index[splits[k] == 'Test']
    cv_splits[int(k)] = {}

    model = None
    if opt.use_vgg_features:
        load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%s.pt' % (opt.model_name, k))
        model_ckpt = torch.load(load_path, map_location=device)
        model_state_dict = model_ckpt['model_state_dict']
        if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata
        model = define_net(opt, None)
        if isinstance(model, torch.nn.DataParallel): model = model.module
        print('Loading the model from %s' % load_path)
        model.load_state_dict(model_state_dict)
        model.eval()

    train_x_patname, train_x_path, train_x_grph, train_x_omic, train_e, train_t, train_g = getAlignedMultimodalData(opt, model, device, all_dataset, pat_train, pat2img)
    test_x_patname, test_x_path, test_x_grph, test_x_omic, test_e, test_t, test_g = getAlignedMultimodalData(opt, model, device, all_dataset, pat_test, pat2img)

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
                  'g':np.array(train_g, dtype=np.float64)}

    test_data = {'x_patname': test_x_patname,
                 'x_path':np.array(test_x_path),
                 'x_grph':test_x_grph,
                 'x_omic':test_x_omic,
                 'e':np.array(test_e, dtype=np.float64),
                 't':np.array(test_t, dtype=np.float64),
                 'g':np.array(test_g, dtype=np.float64)}

    dataset = {'train':train_data, 'test':test_data}
    cv_splits[int(k)] = dataset


### Dictionary file containing split information
data_dict = {}
data_dict['all_dataset'] = all_dataset
data_dict['split'] = cv_splits

pickle.dump(data_dict, open('%s/splits/%s_%d%s.pkl' % (opt.dataroot, opt.roi_dir, opt.use_vgg_features, '_ag' if opt.use_ag else ''), 'wb'))