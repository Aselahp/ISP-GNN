import numpy as np
import os
import math
from os.path import join as pjoin
import torch
from sklearn.model_selection import StratifiedKFold
import itertools
import networkx as nx
from sklearn.preprocessing import normalize
import statistics

from sklearn.preprocessing import MinMaxScaler
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, to_dense_adj
import timeit
from torch_geometric.data import Data
from torch_geometric.utils.sparse import dense_to_sparse


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
    
def map_arrays_to_onehot(input_arrays):
    # Create a mapping of unique arrays to indices
    unique_arrays = {}
    for arr in input_arrays:
        arr_tuple = tuple(map(tuple, arr)) if isinstance(arr, np.ndarray) else tuple(arr)
        if arr_tuple not in unique_arrays:
            unique_arrays[arr_tuple] = len(unique_arrays)
    
    # Calculate minimal bits needed to represent unique arrays
    num_unique = len(unique_arrays)
    bits_needed = max(1, math.ceil(math.log2(num_unique)))
    
    # Create binary encoded representations
    binary_arrays = []
    for arr in input_arrays:
        arr_tuple = tuple(map(tuple, arr)) if isinstance(arr, np.ndarray) else tuple(arr)
        index = unique_arrays[arr_tuple]
        
        # Convert index to binary representation
        binary = [int(b) for b in format(index, f'0{bits_needed}b')]
        
        binary_arrays.append(np.array(binary).reshape(1, -1))
    
    return binary_arrays

    
def compute_augmented_features_onehot(features, max_value):

    N = features.shape[0]
    K = features.shape[1]

    one_hot_features = np.zeros((N, K * (max_value + 1)))

    for i in range(N):
        for k in range(K):
            value = int(features[i, k])
            one_hot_index = k * (max_value + 1) + value
            one_hot_features[i, one_hot_index] = 1

    return one_hot_features

def compute_augmented_features(adj, K):
    # Ensure the adjacency matrix is square
    assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square."

    N = adj.shape[0]
    augmented_features = np.zeros((N, K))

    # Initialize A^k with A
    Ak = np.eye(N)

    for k in range(1, K + 1):
        Ak = np.dot(Ak, adj)  # Compute A^k
        cycle_counts = np.diag(Ak)  # Extract diagonal for cycle counts
        augmented_features[:, k - 1] = cycle_counts

    return augmented_features

class DataReader():
    '''
    Class to read the txt files containing all data of the dataset
    '''
    def __init__(self,
                 name,
                 data_dir,
                 fold_dir,
                 rnd_state=None,
                 use_cont_node_attr=False,
                 folds=10,
                 generate_features=True):

        self.data_dir = data_dir
        self.fold_dir = fold_dir
        self.folds = folds
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        
        if(generate_features==False):
            self.loadData()
        else:
            self.loadData_TURepository(name)
        
        
    def loadData(self):
        files = os.listdir(self.data_dir)
        if self.fold_dir!=None:
            fold_files = os.listdir(self.fold_dir)
        data = {}
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0], 
                                                 nodes, graphs, fn=lambda s: int(s.strip()))  
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)                      
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))
        
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0], 
                                                   nodes, graphs, 
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
            
        features, n_edges, degrees = [], [], []
        
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj)  # total sum of edges
            assert n % 2 == 0, n
            n_edges.append( int(n / 2) )  # undirected edges, so need to divide by 2
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(data['features'][sample_id]))
                        
        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)
        features_min = features_all.min()
        features_dim = int(features_all.max() - features_min + 1)  # number of possible values
        
        features_onehot = []
        for i, x in enumerate(features):
            feature_onehot = np.zeros((len(x), features_dim))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)
            
        #ID_GNN start
        cycle_features = []
        cycle_features_onehot = []
        for sample_id, adj in enumerate(data['adj_list']):
            augmented_features = compute_augmented_features(adj, 3)
            cycle_features.append(augmented_features)
            
            
        features_all = np.concatenate(cycle_features)
        features_max = features_all.max()
        
        cycle_features_onehot = []
        for item in cycle_features:
            cycle_features_onehot.append(compute_augmented_features_onehot(item, int(features_max)))
        
        
        updated_features_onehot = []
        for item in range(len(features_onehot)):
            updated_features_onehot.append(np.concatenate((features_onehot[item], cycle_features_onehot[item]), axis = 1))
            #for PROTEINS dataset
            #updated_features_onehot.append(np.concatenate((features_onehot[item], cycle_features_onehot[item]), axis = 1))
       
        
        features_onehot = updated_features_onehot
        
        features_dim = len(features_onehot[0][0])

        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]     
        #ID-GNN end
            
        
            
        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']        # graph class labels
        labels -= np.min(labels)        # to start from 0
        N_nodes_max = np.max(shapes)    

        classes = np.unique(labels)
        n_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), 
                                                              np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), 
                                                              np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), 
                                                                  np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        for u in np.unique(features_all):
            print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        #random splits
        train_ids, test_ids = self.split_ids(np.arange(N_graphs), rnd_state=self.rnd_state, folds=self.folds)
        
        #read splits from text file 
        #train_ids, test_ids = self.split_ids_from_text(fold_files, rnd_state=self.rnd_state, folds=self.folds)
        
        #stratified splits 
        #train_ids, test_ids = self.stratified_split_data(labels, self.rnd_state, folds)
        
        # Create train sets
        splits = []
        for fold in range(self.folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})
            
        #Tracer()()

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits 
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes
        
        self.data = data
        
        
    def loadData_TURepository(self, name):
        dir_path = os.path.dirname(os.path.realpath('__file__'))
        path = os.path.join(dir_path, self.data_dir, name)
        dataset = TUDataset(path, name= name)
        
        data = {}
        data_list = []
        
        if dataset.data.x is None:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.data.x = NormalizedDegree(mean, std)
                
            for data in dataset:
                data.x = dataset.transform(data).x
                data_list.append(Data(edge_index=data.edge_index, num_nodes=data.x.shape[0], x=data.x, y=data.y))
            
        features = []
        features_onehot = []
        targets = []
        adj_list = []
        max_num_node = 0
        features_dim = 0
        
        for data in data_list:
            features.append(data.x.tolist())
            features_onehot.append(data.x.numpy())
            targets.append(data.y.numpy())
            adj_tensor = to_dense_adj(data['edge_index'])
            adj_reshaped= torch.reshape(adj_tensor, (adj_tensor.shape[1], adj_tensor.shape[1]))
            adj_list.append(adj_reshaped.numpy())
            features_dim = data.x.shape[1]
            if (data.num_nodes > max_num_node):
                max_num_node = data.num_nodes
                
        files = os.listdir(self.data_dir)
        if self.fold_dir!=None:
            fold_files = os.listdir(self.fold_dir)
            
            
        '''skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        splits = []
        for train_idxs, val_idxs in skf.split(dataset, dataset.data.y):
            splits.append({'train': train_idxs,'test': val_idxs})'''
            
        train_ids, test_ids = self.split_ids_from_text(fold_files, rnd_state=self.rnd_state, folds=self.folds)
        # Create train sets
        splits = []
        for fold in range(self.folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})
         
        try:
            non_edge_features_dim = data['non_edge_feature_list'][0].shape[1]
            edge_features_dim = data['edge_feature_list'][0].shape[1]
        except:
           non_edge_features_dim = 0
           edge_features_dim = 0




        data['adj_list'] = adj_list
        
        #ID_GNN start
        '''cycle_features = []
        cycle_features_onehot = []
        for sample_id, adj in enumerate(data['adj_list']):
            augmented_features = compute_augmented_features(adj, 2)
            cycle_features.append(augmented_features)
            
            
        features_all = np.concatenate(cycle_features)
        features_max = features_all.max()
        
        cycle_features_onehot = []
        for item in cycle_features:
            cycle_features_onehot.append(compute_augmented_features_onehot(item, int(features_max)))
        
        
        updated_features_onehot = []
        for item in range(len(features_onehot)):
            updated_features_onehot.append(np.concatenate((features_onehot[item], cycle_features_onehot[item]), axis = 1))
       
        features_dim = features_dim + cycle_features_onehot[0].shape[1]
        
        features_onehot = updated_features_onehot

        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]'''     
        #ID-GNN end
        
        data['features'] =features               
        data['features_onehot'] = features_onehot
        data['targets'] = targets
        data['splits'] = splits 
        data['N_nodes_max'] = max_num_node  # max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = len(np.unique(targets))
        
        self.data = data

    def split_ids(self, ids_all, rnd_state=None, folds=10):
        n = len(ids_all)
        ids = ids_all[rnd_state.permutation(n)]
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids
    
    def split_ids_from_text(self, files, rnd_state=None, folds=10):
        
        train_ids = []
        test_ids = []
        
        test_file_list = sorted([s for s in files if "test" in s])
        train_file_list = sorted([s for s in files if "train" in s])

        for fold in range(folds):
            with open(pjoin(self.fold_dir, train_file_list[fold]), 'r') as f:
                train_samples = [int(line.strip()) for line in f]

            train_ids.append(np.array(train_samples))
            
            with open(pjoin(self.fold_dir, test_file_list[fold]), 'r') as f:
                test_samples = [int(line.strip()) for line in f]

            test_ids.append(np.array(test_samples))

        return train_ids, test_ids
    
    def stratified_split_data(self, labels, seed, folds):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        
        train_ids = []
        test_ids = []
        for fold_idx in range(folds):
            train_idx, test_idx = idx_list[fold_idx]
            train_ids.append(np.array(train_idx))
            test_ids.append(np.array(test_idx))

        return train_ids, test_ids
    
    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
            
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        
        return adj_list
        
    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst