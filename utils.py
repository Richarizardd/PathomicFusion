# Base / Native
import math
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Numerical / Array
import lifelines
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
from imblearn.over_sampling import RandomOverSampler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
from PIL import Image
import pylab
import scipy
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import average_precision_score, auc, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from scipy import interp
mpl.rcParams['axes.linewidth'] = 3 #set the value globally

# Torch
import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch



################
# Regularization
################
def regularize_weights(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg


def regularize_path_weights(model, reg_type=None):
    l1_reg = None
    
    for W in model.classifier.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    for W in model.linear.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    return l1_reg


def regularize_MM_weights(model, reg_type=None):
    l1_reg = None

    if model.module.__hasattr__('omic_net'):
        for W in model.module.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_h_path'):
        for W in model.module.linear_h_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_h_omic'):
        for W in model.module.linear_h_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_h_grph'):
        for W in model.module.linear_h_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_z_path'):
        for W in model.module.linear_z_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_z_omic'):
        for W in model.module.linear_z_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_z_grph'):
        for W in model.module.linear_z_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_o_path'):
        for W in model.module.linear_o_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_o_omic'):
        for W in model.module.linear_o_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('linear_o_grph'):
        for W in model.module.linear_o_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('encoder1'):
        for W in model.module.encoder1.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('encoder2'):
        for W in model.module.encoder2.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('classifier'):
        for W in model.module.classifier.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
        
    return l1_reg


def regularize_MM_omic(model, reg_type=None):
    l1_reg = None

    if model.module.__hasattr__('omic_net'):
        for W in model.module.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    return l1_reg



################
# Network Initialization
################
def init_weights(net, init_type='orthogonal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)           # multi-GPUs

    if init_type != 'max' and init_type != 'none':
        print("Init Type:", init_type)
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        print("Init Type: Self-Normalizing Weights")
    return net



################
# Freeze / Unfreeze
################
def unfreeze_unimodal(opt, model, epoch):
    if opt.mode == 'graphomic':
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == 'pathomic':
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
    elif opt.mode == 'pathgraph':
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "pathgraphomic":
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "omicomic":
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
    elif opt.mode == "graphgraph":
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


def print_if_frozen(module):
    for idx, child in enumerate(module.children()):
        for param in child.parameters():
            if param.requires_grad == True:
                print("Learnable!!! %d:" % idx, child)
            else:
                print("Still Frozen %d:" % idx, child)


def unfreeze_vgg_features(model, epoch):
    epoch_schedule = {30:45}
    unfreeze_index = epoch_schedule[epoch]
    for idx, child in enumerate(model.features.children()):
        if idx > unfreeze_index:
            print("Unfreezing %d:" %idx, child)
            for param in child.parameters(): 
                param.requires_grad = True
        else:
            print("Still Frozen %d:" %idx, child)
            continue



################
# Collate Utils
################
def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)    
    transposed = zip(*batch)
    return [Batch.from_data_list(samples, []) if type(samples[0]) is torch_geometric.data.data.Data else default_collate(samples) for samples in transposed]



################
# Survival Utils
################
def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] < hazards[i]: concord += 0.5

    return(concord/total)


def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))



################
# Data Utils
################
def addHistomolecularSubtype(data):
    """
    Molecular Subtype: IDHwt == 0, IDHmut-non-codel == 1, IDHmut-codel == 2
    Histology Subtype: astrocytoma == 0, oligoastrocytoma == 1, oligodendroglioma == 2, glioblastoma == 3
    """
    subtyped_data = data.copy()
    subtyped_data.insert(loc=0, column='Histomolecular subtype', value=np.ones(len(data)))
    idhwt_ATC = np.logical_and(data['Molecular subtype'] == 0, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhwt_ATC, 'Histomolecular subtype'] = 'idhwt_ATC'
    
    idhmut_ATC = np.logical_and(data['Molecular subtype'] == 1, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhmut_ATC, 'Histomolecular subtype'] = 'idhmut_ATC'
    
    ODG = np.logical_and(data['Molecular subtype'] == 2, data['Histology'] == 2)
    subtyped_data.loc[ODG, 'Histomolecular subtype'] = 'ODG'
    return subtyped_data


def changeHistomolecularSubtype(data):
    """
    Molecular Subtype: IDHwt == 0, IDHmut-non-codel == 1, IDHmut-codel == 2
    Histology Subtype: astrocytoma == 0, oligoastrocytoma == 1, oligodendroglioma == 2, glioblastoma == 3
    """
    data = data.drop(['Histomolecular subtype'], axis=1)
    subtyped_data = data.copy()
    subtyped_data.insert(loc=0, column='Histomolecular subtype', value=np.ones(len(data)))
    idhwt_ATC = np.logical_and(data['Molecular subtype'] == 0, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhwt_ATC, 'Histomolecular subtype'] = 'idhwt_ATC'
    
    idhmut_ATC = np.logical_and(data['Molecular subtype'] == 1, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhmut_ATC, 'Histomolecular subtype'] = 'idhmut_ATC'
    
    ODG = np.logical_and(data['Molecular subtype'] == 2, data['Histology'] == 2)
    subtyped_data.loc[ODG, 'Histomolecular subtype'] = 'ODG'
    return subtyped_data


def getCleanGBMLGG(dataroot='./data/TCGA_GBMLGG/', ignore_missing_moltype=False, ignore_missing_histype=False, use_rnaseq=False, use_ag=False):
    ### 1. Joining all_datasets.csv with grade data. Looks at columns with misisng samples
    metadata = ['Histology', 'Grade', 'Molecular subtype', 'TCGA ID', 'censored', 'Survival months']
    all_dataset = pd.read_csv(os.path.join(dataroot, 'all_dataset.csv')).drop('indexes', axis=1)
    all_dataset.index = all_dataset['TCGA ID']

    all_grade = pd.read_csv(os.path.join(dataroot, 'grade_data.csv'))
    all_grade['Histology'] = all_grade['Histology'].str.replace('astrocytoma (glioblastoma)', 'glioblastoma', regex=False)
    all_grade.index = all_grade['TCGA ID']
    all_grade = all_grade.rename(columns={'Age at diagnosis': 'Age'})
    all_grade['Gender'] = all_grade['Gender'].replace({'male':0, 'female': 1})
    assert pd.Series(all_dataset.index).equals(pd.Series(sorted(all_grade.index)))

    all_dataset = all_dataset.join(all_grade[['Histology', 'Grade', 'Molecular subtype', 'Age', 'Gender']], how='inner')
    cols = all_dataset.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    all_dataset = all_dataset[cols]

    if use_rnaseq:
        gbm = pd.read_csv(os.path.join(dataroot, 'mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt'), sep='\t', skiprows=1, index_col=0)
        lgg = pd.read_csv(os.path.join(dataroot, 'mRNA_Expression_Zscores_RSEM.txt'), sep='\t', skiprows=1, index_col=0)
        gbm = gbm[gbm.columns[~gbm.isnull().all()]]
        lgg = lgg[lgg.columns[~lgg.isnull().all()]]
        glioma_RNAseq = gbm.join(lgg, how='inner').T
        glioma_RNAseq = glioma_RNAseq.dropna(axis=1)
        glioma_RNAseq.columns = [gene+'_rnaseq' for gene in glioma_RNAseq.columns]
        glioma_RNAseq.index = [patname[:12] for patname in glioma_RNAseq.index]
        glioma_RNAseq = glioma_RNAseq.iloc[~glioma_RNAseq.index.duplicated()]
        glioma_RNAseq.index.name = 'TCGA ID'
        all_dataset = all_dataset.join(glioma_RNAseq, how='inner')

    pat_missing_moltype = all_dataset[all_dataset['Molecular subtype'].isna()].index
    pat_missing_idh = all_dataset[all_dataset['idh mutation'].isna()].index
    pat_missing_1p19q = all_dataset[all_dataset['codeletion'].isna()].index
    print("# Missing Molecular Subtype:", len(pat_missing_moltype))
    print("# Missing IDH Mutation:", len(pat_missing_idh))
    print("# Missing 1p19q Codeletion:", len(pat_missing_1p19q))
    assert pat_missing_moltype.equals(pat_missing_idh)
    assert pat_missing_moltype.equals(pat_missing_1p19q)
    pat_missing_grade =  all_dataset[all_dataset['Grade'].isna()].index
    pat_missing_histype = all_dataset[all_dataset['Histology'].isna()].index
    print("# Missing Histological Subtype:", len(pat_missing_histype))
    print("# Missing Grade:", len(pat_missing_grade))
    assert pat_missing_histype.equals(pat_missing_grade)

    ### 2. Impute Missing Genomic Data: Removes patients with missing molecular subtype / idh mutation / 1p19q. Else imputes with median value of each column. Fills missing Molecular subtype with "Missing"
    if ignore_missing_moltype: 
        all_dataset = all_dataset[all_dataset['Molecular subtype'].isna() == False]
    for col in all_dataset.drop(metadata, axis=1).columns:
        all_dataset['Molecular subtype'] = all_dataset['Molecular subtype'].fillna('Missing')
        all_dataset[col] = all_dataset[col].fillna(all_dataset[col].median())

    ### 3. Impute Missing Histological Data: Removes patients with missing histological subtype / grade. Else imputes with "missing" / grade -1
    if ignore_missing_histype: 
        all_dataset = all_dataset[all_dataset['Histology'].isna() == False]
    else:
        all_dataset['Grade'] = all_dataset['Grade'].fillna(1)
        all_dataset['Histology'] = all_dataset['Histology'].fillna('Missing')
    all_dataset['Grade'] = all_dataset['Grade'] - 2

    ### 4. Adds Histomolecular subtype
    ms2int = {'Missing':-1, 'IDHwt':0, 'IDHmut-non-codel':1, 'IDHmut-codel':2}
    all_dataset[['Molecular subtype']] = all_dataset[['Molecular subtype']].applymap(lambda s: ms2int.get(s) if s in ms2int else s)
    hs2int = {'Missing':-1, 'astrocytoma':0, 'oligoastrocytoma':1, 'oligodendroglioma':2, 'glioblastoma':3}
    all_dataset[['Histology']] = all_dataset[['Histology']].applymap(lambda s: hs2int.get(s) if s in hs2int else s)
    all_dataset = addHistomolecularSubtype(all_dataset)
    metadata.extend(['Histomolecular subtype'])

    if use_ag == 0:
        metadata.extend(['Age', 'Gender'])

    all_dataset['censored'] = 1 - all_dataset['censored']
    return metadata, all_dataset

def getCleanKIRC(dataroot='./', rnaseq_cutoff='all', cnv_cutoff=7.0, mut_cutoff=5.0):
    ### Clinical variables
    clinical = pd.read_table(os.path.join(dataroot, './kirc_tcga_pan_can_atlas_2018_clinical_data.tsv'), index_col=2)
    clinical.index.name = None
    clinical['censored'] = clinical['Overall Survival Status']
    clinical['censored'] = clinical['censored'].replace('LIVING', 1)
    clinical['censored'] = clinical['censored'].replace('DECEASED', 0)
    clinical['censored'] = 1-clinical['censored']

    ### Select RNAseq Features
    rnaseq = pd.read_table(os.path.join(dataroot, 'data_RNA_Seq_v2_mRNA_median_Zscores.txt'), index_col=0)
    rnaseq = rnaseq[rnaseq.index.notnull()]
    rnaseq = rnaseq.drop(['Entrez_Gene_Id'], axis=1)
    rnaseq.index.name = None
    rnaseqDEGs = pd.read_csv(os.path.join(dataroot, 'dataDEGs_kirc.csv'), index_col=0)
    rnaseqDEGs = rnaseqDEGs.sort_values(['PValue', 'logFC'], ascending=False)
    rnaseq_cutoff = rnaseqDEGs.shape[0] if isinstance(rnaseq_cutoff, str) else rnaseq_cutoff
    rnaseq = rnaseq.loc[rnaseq.index.intersection(rnaseqDEGs.index)].T
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
                                 
    mut = clinical[['Patient ID']].copy()
    for tsv in os.listdir(os.path.join(dataroot, 'muts')):
        if tsv.endswith('.tsv'):
            mut_samples = pd.read_table(os.path.join(dataroot, 'muts', tsv))['Patient ID']
            mut_gene = tsv.split('_')[2].rstrip('.tsv')+'_mut'
            mut[mut_gene] = 0
            mut.loc[mut.index[:-3].isin(mut_samples), mut_gene] = 1
    mut = mut.drop(['Patient ID'], axis=1)
    
    omic_features = rnaseq.join(cnv, how='inner').join(mut, how='inner')
    return omic_features


def getCleanIvyGlioma(dataroot='./data/IvyGlioma/', folder='genomic'):
    tumor_details = pd.read_csv(os.path.join(dataroot, folder, 'tumor_details.csv'))
    bulk_rnaseq = pd.read_table(os.path.join(dataroot, folder, 'RNA-seq_Ivybulktumor.txt'))
    bulk_rnaseq.index = bulk_rnaseq['genesymbol'].str.lower()
    bulk_rnaseq.index.name = None
    bulk_rnaseq = bulk_rnaseq.drop(bulk_rnaseq.columns[:5], axis=1)
    bulk_rnaseq = bulk_rnaseq.drop(['Unnamed: 22', 'Unnamed: 29'], axis=1)
    bulk_rnaseq = bulk_rnaseq.drop(['W5-1-1', 'W10-1-1', 'W22-1-1', 'W33-1-1'], axis=1)
    bulk_rnaseq = bulk_rnaseq.T
    bulk_rnaseq.columns = [g.lower()+'_rnaseq' for g in bulk_rnaseq.columns]

    ignore_missing_moltype, ignore_missing_histype, use_rnaseq = False, False, True
    metadata_tcga, all_dataset_tcga = getCleanAllDataset(dataroot='./data/TCGA_GBMLGG/', 
                                     ignore_missing_moltype=ignore_missing_moltype, 
                                     ignore_missing_histype=ignore_missing_histype, 
                                     use_rnaseq=use_rnaseq)

    all_dataset_tcga.columns = list(all_dataset_tcga.columns[:7]) + [g.lower() for g in all_dataset_tcga.columns[7:]]
    fpkm_table_mean.insert(loc=0, column='codeletion', value=0)
    fpkm_table_mean.insert(loc=1, column='idh mutation', value=0)
    genes_overlap = list(set(all_dataset_tcga.columns).intersection(set(bulk_rnaseq.columns)))

    all_dataset_ivy = tumor_details#[tumor_details['tumor_name'].isin(best_patients)]
    all_dataset_ivy = all_dataset_ivy[~all_dataset_ivy['survival_days'].isna()]
    all_dataset_ivy['Survival months'] = all_dataset_ivy['survival_days'] / 30
    all_dataset_ivy['censored'] = 1
    all_dataset_ivy.index = all_dataset_ivy['tumor_name']
    all_dataset_ivy.index.name = None
    all_dataset_ivy_feats = bulk_rnaseq[genes_overlap]
    all_dataset_ivy_feats.index.name = None
    all_dataset_ivy = all_dataset_ivy.join(all_dataset_ivy_feats, how='inner')
    all_dataset_ivy.index.name = 'tumor_name'
    metadata_ivy = all_dataset_ivy.columns[:12]
    return (metadata_ivy, all_dataset_ivy), (metadata_tcga, all_dataset_tcga), genes_overlap


def getCleanIvyGlioma_dep(dataroot='./data/IvyGlioma/', folder='genomic', which_structures='default', bulk=True):
    row_genes = pd.read_csv(os.path.join(dataroot, folder, 'rows-genes.csv'))
    column_samples = pd.read_csv(os.path.join(dataroot, folder, 'columns-samples.csv'))
    fpkm_table = pd.read_csv(os.path.join(dataroot, folder, 'fpkm_table.csv'), index_col=0)
    tumor_details = pd.read_csv(os.path.join(dataroot, folder, 'tumor_details.csv'))

    fpkm_table.index = [g.lower() for g in row_genes['gene_symbol']]
    fpkm_table = fpkm_table.T
    assert fpkm_table.index.all(column_samples['rna_well_id'])

    if which_structures == 'default':
        which_structures = ['CT-reference-histology', 'CTmvp-reference-histology', 'CTpan-reference-histology', 'LE-reference-histology', 'IT-reference-histology']
    elif  which_structures == 'all':
        which_structures = column_samples['structure_abbreviation'].unique()
    elif which_structres == 'two':
        which_structures = ['CT-reference-histology', 'CTmvp-reference-histology']
    else:
        print("Error")

    select_structures = column_samples['structure_abbreviation'].isin(which_structures)
    fpkm_table = fpkm_table[np.array(select_structures)]
    column_samples = column_samples[np.array(select_structures)]
    column_samples = column_samples.sort_values(['tumor_name', 'block_name', 'structure_abbreviation'])

    fpkm_table_mean = fpkm_table.copy()
    fpkm_table_mean.index = column_samples['tumor_name']
    if bulk:
        fpkm_table_mean = fpkm_table_mean.groupby('tumor_name').mean()

    ignore_missing_moltype, ignore_missing_histype, use_rnaseq = False, False, True
    metadata_tcga, all_dataset_tcga = getCleanAllDataset(dataroot='./data/TCGA_GBMLGG/', 
                                     ignore_missing_moltype=ignore_missing_moltype, 
                                     ignore_missing_histype=ignore_missing_histype, 
                                     use_rnaseq=use_rnaseq)

    all_dataset_tcga.columns = list(all_dataset_tcga.columns[:7]) + [g.lower() for g in all_dataset_tcga.columns[7:]]
    fpkm_table_mean.columns = [g.lower()+'_rnaseq' for g in fpkm_table.columns]
    fpkm_table_mean.insert(loc=0, column='codeletion', value=0)
    fpkm_table_mean.insert(loc=1, column='idh mutation', value=0)
    genes_overlap = list(set(all_dataset_tcga.columns).intersection(set(fpkm_table_mean.columns)))

    fpkm_table_mean.shape

    metadata_tcga = [metadata_tcga[-1]] + list(metadata_tcga[:-1])
    all_dataset_tcga.columns = metadata_tcga + list(all_dataset_tcga.columns[len(metadata_tcga):])
    #all_dataset_tcga = all_dataset_tcga.loc[:,~all_dataset_tcga.columns.duplicated()]
    best_patients = fpkm_table_mean.index
    all_dataset_ivy = tumor_details[tumor_details['tumor_name'].isin(best_patients)]
    all_dataset_ivy = all_dataset_ivy[~all_dataset_ivy['survival_days'].isna()]
    all_dataset_ivy['Survival months'] = all_dataset_ivy['survival_days'] / 30
    all_dataset_ivy['censored'] = 1
    all_dataset_ivy.index = all_dataset_ivy['tumor_name']
    all_dataset_ivy.index.name = None
    all_dataset_ivy_feats = fpkm_table_mean[genes_overlap]
    all_dataset_ivy_feats.index.name = None
    all_dataset_ivy = all_dataset_ivy.join(all_dataset_ivy_feats)
    all_dataset_ivy.index.name = 'tumor_name'
    metadata_ivy = all_dataset_ivy.columns[:12]
    return (metadata_ivy, all_dataset_ivy), (metadata_tcga, all_dataset_tcga), genes_overlap


################
# Analysis Utils
################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)