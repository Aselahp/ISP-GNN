
import numpy as np
import time
import networkx as nx
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import heapq as hp

from graph_data import GraphData
from data_reader import DataReader
from models import GNN


# Experiment parameters
'''
----------------------------
Dataset  |   batchnorm_dim
----------------------------
MUTAG    |     28
PTC_MR   |     64
BZR      |     57
COX2     |     56
COX2_MD  |     36
BZR-MD   |     33
PROTEINS |    620
IMDB-B   |    136
D&D      |   5748
'''
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', help='Select CPU/CUDA for training.')
parser.add_argument('--dataset', default='PTC_MR', help='Dataset name.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.') 
parser.add_argument('--wdecay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of MLP layers for GraphSN.')
parser.add_argument('--batchnorm_dim', type=int, default=64, help='Batchnormalization dimension for GraphSN layer.')
parser.add_argument('--dropout_1', type=float, default=0.5, help='Dropout rate for concatenation the outputs.') 
parser.add_argument('--dropout_2', type=float, default=0.1,  help='Dropout rate for MLP layers in GraphSN.')
parser.add_argument('--n_folds', type=int, default=10, help='Number of folds in cross validation.')
parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
parser.add_argument('--log_interval', type=int, default=10 , help='Log interval for visualizing outputs.')
parser.add_argument('--seed', type=int, default=117, help='Random seed.')

args = parser.parse_args("")
print('Loading data')
dataset_fold_idx_path = './data/%s/' % args.dataset.upper() #+ 'fold_idx/'
datareader = DataReader(name=args.dataset.upper(),
                        data_dir='./data/%s/' % args.dataset.upper(),
                         fold_dir=dataset_fold_idx_path,
                         rnd_state=np.random.RandomState(args.seed),
                         folds=args.n_folds,                    
                         use_cont_node_attr=False,
                         generate_features=False)

#datareader = DataReader(data_dir='./data/%s/' % args.dataset.upper(),
#                        fold_dir=None,
#                        rnd_state=np.random.RandomState(args.seed),
#                        folds=args.n_folds,                    
#                        use_cont_node_attr=False)

dataset_length = len(datareader.data['adj_list'])
acc_folds = []
accuracy_arr = np.zeros((10, args.epochs), dtype=float)
for fold_id in range(args.n_folds):
    print('\nFOLD', fold_id)
    loaders = []
    for split in ['train', 'test']:
        gdata = GraphData(fold_id=fold_id,
                             datareader=datareader,
                             split=split)

        loader = torch.utils.data.DataLoader(gdata, 
                                             batch_size=args.batch_size,
                                             shuffle=split.find('train') >= 0,
                                             num_workers=args.threads)
        loaders.append(loader)
    
    model = GNN(input_dim=loaders[0].dataset.features_dim,
                hidden_dim=args.hidden_dim,
                output_dim=loaders[0].dataset.n_classes,
                n_layers=args.n_layers, 
                batchnorm_dim=args.batchnorm_dim, 
                dropout_1=args.dropout_1, 
                dropout_2=args.dropout_2).to(args.device)

    print('\nInitialize model')
    print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wdecay,
                betas=(0.5, 0.999))
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)

    def train(train_loader):
        scheduler.step()
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data[4])
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                    epoch, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))

    def test(test_loader):
        model.eval()
        start = time.time()
        test_loss, correct, n_samples = 0, 0, 0
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            output = model(data)
            loss = loss_fn(output, data[4], reduction='sum') 
            test_loss += loss.item()
            n_samples += len(output)
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

        time_iter = time.time() - start

        test_loss /= n_samples

        acc = 100. * correct / n_samples
        print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                              test_loss, 
                                                                                              correct, 
                                                                                              n_samples, acc))
        return acc

    loss_fn = F.cross_entropy
    max_acc = 0.0
    for epoch in range(args.epochs):
        train(loaders[0])
        acc = test(loaders[1])
        accuracy_arr[fold_id][epoch] = acc
        max_acc = max(max_acc, acc)
    acc_folds.append(max_acc)

print(acc_folds)
print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, np.mean(acc_folds), np.std(acc_folds)))

mean_validation = accuracy_arr.mean(axis=0)
maximum_epoch = np.argmax(mean_validation)
average = np.mean(accuracy_arr[:, maximum_epoch])
standard_dev = np.std(accuracy_arr[:, maximum_epoch])
print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, average, standard_dev))
