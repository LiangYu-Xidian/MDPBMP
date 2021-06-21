import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.pytorchtools import EarlyStopping
from utils.data import load_MDPBMP_data
from utils.tools import index_generator, parse_minibatch_MDPBMP
from model import MDPBMP_lp

# Params
num_ntype = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001 
etypes_lists = [[[0, 1], [0, 2, 3, 1], [None]],
                [[1, 0], [2, 3], [1, None, 0]]]  #关系种类    
miRNA_masks = [[True, True, False],
             [True, False, True]]   #验证集：是否包含miRNA_disease链接
no_masks = [[False] * 3, [False] * 3]   #测试集
#num_miRNA = 1206
#num_disease = 894
num_miRNA = 495
num_disease = 383
expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]
]


def run_model_MDPBMP(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    adjlists_ua, edge_metapath_indices_list_ua, _, type_mask, train_val_test_pos_miRNA_disease, train_val_test_neg_miRNA_disease = load_MDPBMP_data()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    if feats_type == 0:   #all id vectors / Default
        for i in range(num_ntype):    
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    elif feats_type == 1:   #all zero vector
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))    

    train_pos_miRNA_disease = train_val_test_pos_miRNA_disease['train_pos_miRNA_disease']
    val_pos_miRNA_disease = train_val_test_pos_miRNA_disease['val_pos_miRNA_disease']
    test_pos_miRNA_disease = train_val_test_pos_miRNA_disease['test_pos_miRNA_disease']
    train_neg_miRNA_disease = train_val_test_neg_miRNA_disease['train_neg_miRNA_disease']
    val_neg_miRNA_disease = train_val_test_neg_miRNA_disease['val_neg_miRNA_disease']
    test_neg_miRNA_disease = train_val_test_neg_miRNA_disease['test_neg_miRNA_disease']
    y_true_test = np.array([1] * len(test_pos_miRNA_disease) + [0] * len(test_neg_miRNA_disease))

    auc_list = []
    ap_list = []
    for _ in range(repeat):
        net = MDPBMP_lp(
            [3, 3], 4, etypes_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_miRNA_disease))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_miRNA_disease), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_miRNA_disease_batch = train_pos_miRNA_disease[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_miRNA_disease), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_miRNA_disease_batch = train_neg_miRNA_disease[train_neg_idx_batch].tolist()


                train_miRNA_disease_batch = train_pos_miRNA_disease_batch + train_neg_miRNA_disease_batch

                train_g_lists, train_indices_lists, train_idx_batch_mapped_lists = parse_minibatch_MDPBMP(
                    adjlists_ua, edge_metapath_indices_list_ua, train_miRNA_disease_batch, device, neighbor_samples, miRNA_masks, num_miRNA)

                t1 = time.time()
                dur1.append(t1 - t0)

                [embedding_miRNA, embedding_disease], _ = net(
                    (train_g_lists, features_list, type_mask, train_indices_lists, train_idx_batch_mapped_lists))
                
                pos_embedding_miRNA, neg_embedding_miRNA = embedding_miRNA.chunk(2, dim=0)
                pos_embedding_disease, neg_embedding_disease = embedding_disease.chunk(2, dim=0)


                pos_embedding_miRNA = pos_embedding_miRNA.view(-1, 1, pos_embedding_miRNA.shape[1])
                pos_embedding_disease = pos_embedding_disease.view(-1, pos_embedding_disease.shape[1], 1)
                neg_embedding_miRNA = neg_embedding_miRNA.view(-1, 1, neg_embedding_miRNA.shape[1])
                neg_embedding_disease = neg_embedding_disease.view(-1, neg_embedding_disease.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_miRNA, pos_embedding_disease)
                neg_out = -torch.bmm(neg_embedding_miRNA, neg_embedding_disease)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()  #在反向传播之前需要将优化器中的梯度值清零，因为在默认情况下反向传播的梯度值会进行累加
                train_loss.backward()  #进行反性传播，计算损失函数对于网络参数的梯度值
                optimizer.step()       #按照梯度值与优化器的定义来改变网络参数值，使其朝着输出更好结果的方向改变

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_miRNA_disease_batch = val_pos_miRNA_disease[val_idx_batch].tolist()
                    val_neg_miRNA_disease_batch = val_neg_miRNA_disease[val_idx_batch].tolist()
                   

                    val_miRNA_disease_batch = val_pos_miRNA_disease_batch + val_neg_miRNA_disease_batch
                    val_g_lists, val_indices_lists, val_idx_batch_mapped_lists = parse_minibatch_MDPBMP(
                        adjlists_ua, edge_metapath_indices_list_ua, val_miRNA_disease_batch, device, neighbor_samples, no_masks, num_miRNA)

                    [embedding_miRNA, embedding_disease], _ = net(
                        (val_g_lists, features_list, type_mask, val_indices_lists, val_idx_batch_mapped_lists))

                    pos_embedding_miRNA, neg_embedding_miRNA = embedding_miRNA.chunk(2, dim=0)
                    pos_embedding_disease, neg_embedding_disease = embedding_disease.chunk(2, dim=0)

                    pos_embedding_miRNA = pos_embedding_miRNA.view(-1, 1, pos_embedding_miRNA.shape[1])
                    pos_embedding_disease = pos_embedding_disease.view(-1, pos_embedding_disease.shape[1], 1)
                    neg_embedding_miRNA = neg_embedding_miRNA.view(-1, 1, neg_embedding_miRNA.shape[1])
                    neg_embedding_disease = neg_embedding_disease.view(-1, neg_embedding_disease.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_miRNA, pos_embedding_disease)
                    neg_out = -torch.bmm(neg_embedding_miRNA, neg_embedding_disease)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_miRNA_disease), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []

        with torch.no_grad():

            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_miRNA_disease_batch = test_pos_miRNA_disease[test_idx_batch].tolist()
                test_neg_miRNA_disease_batch = test_neg_miRNA_disease[test_idx_batch].tolist()
                test_miRNA_disease_batch = test_pos_miRNA_disease_batch + test_neg_miRNA_disease_batch
                test_g_lists, test_indices_lists, test_idx_batch_mapped_lists = parse_minibatch_MDPBMP(
                    adjlists_ua, edge_metapath_indices_list_ua, test_miRNA_disease_batch, device, neighbor_samples, no_masks, num_miRNA)

                [embedding_miRNA, embedding_disease], [h_miRNA, h_disease] = net(
                    (test_g_lists, features_list, type_mask, test_indices_lists, test_idx_batch_mapped_lists))
                
                pos_embedding_miRNA, neg_embedding_miRNA = embedding_miRNA.chunk(2, dim=0)
                pos_embedding_disease, neg_embedding_disease = embedding_disease.chunk(2, dim=0)

                pos_embedding_miRNA = pos_embedding_miRNA.view(-1, 1, pos_embedding_miRNA.shape[1])
                pos_embedding_disease = pos_embedding_disease.view(-1, pos_embedding_disease.shape[1], 1)
                neg_embedding_miRNA = neg_embedding_miRNA.view(-1, 1, neg_embedding_miRNA.shape[1])
                neg_embedding_disease = neg_embedding_disease.view(-1, neg_embedding_disease.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_miRNA, pos_embedding_disease).flatten()
                neg_out = torch.bmm(neg_embedding_miRNA, neg_embedding_disease).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))


        
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        np.savetxt('1y_true_test.txt', y_true_test)
        np.savetxt('1y_proba_test.txt', y_proba_test)
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    # print('----------------------------------------------------------------')
    # print('Link Prediction Tests Summary')
    # print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    # print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MDPBMP testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector;' +
                         '2 - all lncRNA vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='max-pooling', help='Type of the aggregator. Default is max-pooling.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='MDPBMP', help='Postfix for the saved model and result. Default is MDPBMP.')

    args = ap.parse_args()
    run_model_MDPBMP(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)