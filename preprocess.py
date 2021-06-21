import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd


save_prefix = ''

miRNA_disease = pd.read_csv('miRNA_disease_id.txt', encoding='utf-8', delimiter='\t', names=['miRNAID', 'diseaseID'])
miRNA_adjacent = pd.read_csv('miRNASim_id.txt', encoding='utf-8', delimiter='\t', names=['miRNAID', 'adjacentID'])
miRNA_Sim = pd.read_csv('miRNASim_score.txt', encoding='utf-8', delimiter='\t', names=['similarity'])
disease_gene= pd.read_csv('gene-disease_id.txt', encoding='utf-8', delimiter='\t', names=['geneID', 'diseaseID','score'])
num_miRNA = 495
num_disease = 383
num_gene = 3790

train_val_test_idx = np.load('train_val_test_idx.npz')
train_idx = train_val_test_idx['train_idx']
val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']

miRNA_disease = miRNA_disease.loc[train_idx].reset_index(drop=True)

# build the adjacency matrix
# 0 for miRNA, 1 for disease, 2 for gene
dim = num_miRNA + num_disease + num_gene

type_mask = np.zeros((dim), dtype=int)
type_mask[num_miRNA:num_miRNA+num_disease] = 1
type_mask[num_miRNA+num_disease:] = 2

adjM = np.zeros((dim, dim), dtype=int)
for _, row in miRNA_disease.iterrows():
    mid = row['miRNAID']
    did = num_miRNA + row['diseaseID']
    adjM[mid, did] = 100
    adjM[did, mid] = 100
    
miRNA_adjacent_Sim = pd.concat([miRNA_adjacent,miRNA_Sim],axis=1)
for _, row in miRNA_adjacent_Sim.iterrows():
    mid = int(row['miRNAID'])
    aid = int(row['adjacentID'])
    adjM[mid, aid] = row['similarity']*100
    
for _, row in disease_gene.iterrows():
    did = num_miRNA + int(row['diseaseID'])
    gid = num_miRNA + num_disease + int(row['geneID'])
    adjM[did, gid] = row['score']*100
    adjM[gid, did] = row['score']*100

miRNA_disease_list = {i: adjM[i, num_miRNA:num_miRNA+num_disease].nonzero()[0] for i in range(num_miRNA)}
disease_miRNA_list = {i: adjM[num_miRNA + i, :num_miRNA].nonzero()[0] for i in range(num_disease)}
miRNA_adjacent_list = {i: adjM[i, :num_miRNA].nonzero()[0] for i in range(num_miRNA)}
disease_gene_list = {i: adjM[num_miRNA + i, num_miRNA+num_disease:].nonzero()[0] for i in range(num_disease)}
gene_disease_list = {i: adjM[num_miRNA + num_disease + i, num_miRNA:num_miRNA+num_disease].nonzero()[0] for i in range(num_gene)}

# 0-1-0
m_d_m = []
for d, m_list in disease_miRNA_list.items():
    m_d_m.extend([(m1, d, m2) for m1 in m_list for m2 in m_list])
m_d_m = np.array(m_d_m)
m_d_m[:, 1] += num_miRNA
sorted_index = sorted(list(range(len(m_d_m))), key=lambda i : m_d_m[i, [0, 2, 1]].tolist())
m_d_m = m_d_m[sorted_index]

# 1-2-1
d_g_d = []
for g, d_list in gene_disease_list.items():
    d_g_d.extend([(d1, g, d2) for d1 in d_list for d2 in d_list])
d_g_d = np.array(d_g_d)
d_g_d += num_miRNA
d_g_d[:, 1] += num_disease
sorted_index = sorted(list(range(len(d_g_d))), key=lambda i : d_g_d[i, [0, 2, 1]].tolist())
d_g_d = d_g_d[sorted_index]

# 0-1-2-1-0
m_d_g_d_m = []
for d1, g, d2 in d_g_d:
    if len(disease_miRNA_list[d1 - num_miRNA]) == 0 or len(disease_miRNA_list[d2 - num_miRNA]) == 0:
        continue
    candidate_m1_list = np.random.choice(len(disease_miRNA_list[d1 - num_miRNA]), int(0.5 * len(disease_miRNA_list[d1 - num_miRNA])), replace=False)
    candidate_m1_list = disease_miRNA_list[d1 - num_miRNA][candidate_m1_list]
    candidate_m2_list = np.random.choice(len(disease_miRNA_list[d2 - num_miRNA]), int(0.5 * len(disease_miRNA_list[d2 - num_miRNA])), replace=False)
    candidate_m2_list = disease_miRNA_list[d2 - num_miRNA][candidate_m2_list]
    m_d_g_d_m.extend([(m1, d1, g, d2, m2) for m1 in candidate_m1_list for m2 in candidate_m2_list])
m_d_g_d_m = np.array(m_d_g_d_m)
sorted_index = sorted(list(range(len(m_d_g_d_m))), key=lambda i : m_d_g_d_m[i, [0, 4, 1, 2, 3]].tolist())
m_d_g_d_m = m_d_g_d_m[sorted_index]

# 0-0
# m_m = miRNA_adjacent.to_numpy(dtype=np.int32) - 1
m_m = np.array(miRNA_adjacent)
sorted_index = sorted(list(range(len(m_m))), key=lambda i : m_m[i].tolist())
m_m = m_m[sorted_index]

# 1-0-1
d_m_d = []
for m, d_list in miRNA_disease_list.items():
    d_m_d.extend([(d1, m, d2) for d1 in d_list for d2 in d_list])
d_m_d = np.array(d_m_d)
d_m_d[:, [0, 2]] += num_miRNA
sorted_index = sorted(list(range(len(d_m_d))), key=lambda i : d_m_d[i, [0, 2, 1]].tolist())
d_m_d = d_m_d[sorted_index]

# 1-0-0-1
d_m_m_d = []
for m1, m2 in m_m:
    d_m_m_d.extend([(d1, m1, m2, d2) for d1 in miRNA_disease_list[m1] for d2 in miRNA_disease_list[m2]])
d_m_m_d = np.array(d_m_m_d)
d_m_m_d[:, [0, 3]] += num_miRNA
sorted_index = sorted(list(range(len(d_m_m_d))), key=lambda i : d_m_m_d[i, [0, 3, 1, 2]].tolist())
d_m_m_d = d_m_m_d[sorted_index]

expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]
]
# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

metapath_indices_mapping = {(0, 1, 0): m_d_m,
                            (0, 1, 2, 1, 0): m_d_g_d_m,
                            (0, 0): m_m,
                            (1, 0, 1): d_m_d,
                            (1, 2, 1): d_g_d,
                            (1, 0, 0, 1): d_m_m_d}

# write all things
target_idx_lists = [np.arange(num_miRNA), np.arange(num_disease)]
offset_list = [0, num_miRNA]
for i, metapaths in enumerate(expected_metapaths):
    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]
        
        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        #np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)
        
        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]
                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
                else:
                    out_file.write('{}\n'.format(target_idx))
                left = right

scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
np.save(save_prefix + 'node_types.npy', type_mask)
