import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd


# output positive and negative samples for training, validation and testing

np.random.seed(453289)
num_miRNA = 495
num_disease = 383
miRNA_disease = np.load('miRNA_disease.npy')
train_val_test_idx = np.load('train_val_test_idx.npz')
train_idx = train_val_test_idx['train_idx']
val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']

neg_candidates = []
counter = 0
for i in range(num_miRNA):
    for j in range(num_disease):
        if counter < len(miRNA_disease):
            if i == miRNA_disease[counter, 0] and j == miRNA_disease[counter, 1]:
                counter += 1
            else:
                neg_candidates.append([i, j])
        else:
            neg_candidates.append([i, j])
neg_candidates = np.array(neg_candidates)

idx = np.random.choice(len(neg_candidates), len(val_idx) + len(test_idx), replace=False)
val_neg_candidates = neg_candidates[sorted(idx[:len(val_idx)])]
test_neg_candidates = neg_candidates[sorted(idx[len(val_idx):])]

train_miRNA_disease = miRNA_disease[train_idx]
train_neg_candidates = []
counter = 0
for i in range(num_miRNA):
    for j in range(num_disease):
        if counter < len(train_miRNA_disease):
            if i == train_miRNA_disease[counter, 0] and j == train_miRNA_disease[counter, 1]:
                counter += 1
            else:
                train_neg_candidates.append([i, j])
        else:
            train_neg_candidates.append([i, j])
train_neg_candidates = np.array(train_neg_candidates)

np.savez(save_prefix + 'train_val_test_neg_miRNA_disease.npz',
         train_neg_miRNA_disease=train_neg_candidates,
         val_neg_miRNA_disease=val_neg_candidates,
         test_neg_miRNA_disease=test_neg_candidates)
np.savez(save_prefix + 'train_val_test_pos_miRNA_disease.npz',
         train_pos_miRNA_disease=miRNA_disease[train_idx],
         val_pos_miRNA_disease=miRNA_disease[val_idx],
         test_pos_miRNA_disease=miRNA_disease[test_idx])
