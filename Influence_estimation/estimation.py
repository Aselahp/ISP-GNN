import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F
import time
import random
import pickle
import pandas as pd
import scipy.sparse as sp
import os
from torch.optim import Adam
from est_models import ISP_GIN


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1

dataset_name = 'power_grid'
diffusion_model = 'IC'
seed_rate = 10

n = 2

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def estimation_loss(y, y_hat):
    forward_loss = F.mse_loss(y_hat.squeeze(), y, reduction='sum')
    return forward_loss

print(f"\n{'='*80}")
print(f"Running experiment with Dataset: {dataset_name}, Diffusion Model: {diffusion_model}, Seed Rate: {seed_rate}")
print(f"{'='*80}\n")

file_path = f'data/{dataset_name}_mean_{diffusion_model}{10*seed_rate}.SG'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found.")

with open(file_path, 'rb') as f:
    graph = pickle.load(f)

adj, dataset = graph['adj'], graph['inverse_pairs']

adj = torch.Tensor(adj.toarray()).to_sparse()
adj = adj.to(device)
feature_matrix = torch.ones(adj.shape[0], 1)
feature_matrix = feature_matrix.to(device)

edge_index = adj.coalesce().indices()

batch_size = 1 # 1, 5, 10

kf = KFold(n_splits=10, shuffle=True, random_state=42)
val_error = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/10')
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
    
    model = ISP_GIN(
        in_channels=feature_matrix.shape[1], 
        hidden_channels=32, 
        out_channels=1, 
        dropout=0.5, 
        num_layers=n, 
        num_nodes=feature_matrix.shape[0]
    ).to(device)

    optimizer = Adam([{'params': model.parameters()}], lr=0.005, weight_decay=5e-4)
    
    best_val_error = float('inf')
    epochs_without_improvement = 0
    patience = 25
    max_epochs = 100
    
    for epoch in range(max_epochs):
        begin = time.time()
        total_overall = 0
        model.train()

        for batch_idx, data_pair in enumerate(train_loader):
            optimizer.zero_grad()
            
            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)
            
            loss = 0
            for i, x_i in enumerate(x):
                y_i = y[i]
                
                x_hat = feature_matrix
                y_hat, curv_loss = model(x_hat, edge_index)
                total = estimation_loss(y_i, y_hat) + curv_loss
                            
                loss += total

            total_overall += loss.item()
            loss = loss/x.size(0)
            loss.backward()
            optimizer.step()
            
        end = time.time()
        
        if (epoch + 1) % 1 == 0:
            print("Epoch: {}".format(epoch+1), 
                "\tTotal: {:.4f}".format(total_overall / len(train_subset)),
                "\tTime: {:.4f}".format(end - begin)
                )
        
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            val_mae = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, data_pair in enumerate(val_loader):
                    x = data_pair[:, :, 0].float().to(device)
                    y = data_pair[:, :, 1].float().to(device)

                    x_hat = feature_matrix
                    y_hat, curv_loss = model(x_hat, edge_index)
                    val_mae += np.abs(y_hat.squeeze() - y[0]).sum()/x[0].shape[0]
            
            val_mae /= len(val_loader)
            
            if val_mae < best_val_error:
                best_val_error = val_mae
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 10
            
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                break
            
            model.train()
    
    val_mae = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, data_pair in enumerate(val_loader):
            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)

            x_hat = feature_matrix
            y_hat, curv_loss = model(x_hat, edge_index)
            val_mae += np.abs(y_hat.squeeze() - y[0]).sum()/x[0].shape[0]
    
    val_mae /= len(val_loader)
    val_error.append(val_mae)
    print('Final Validation Loss: ', val_mae)

mean = np.mean(val_error)
std_dev = np.std(val_error, ddof=1)

print(f"\n{'='*80}")
print(f"Final Results for {dataset_name} with {diffusion_model} (Seed Rate {seed_rate}):")
print(f"Mean: {mean:.6f}")
print(f"Standard Deviation: {std_dev:.6f}")
print(f"{'='*80}\n")

results = pd.DataFrame({
    'Dataset': [dataset_name],
    'Diffusion_Model': [diffusion_model],
    'Seed_Rate': [seed_rate],
    'Mean_Error': [mean],
    'Std_Dev': [std_dev],
    'Val_Errors': [val_error]
})

results.to_csv(f'ISP_GNN_{dataset_name}_{diffusion_model}.csv', index=False)
print(f"Results saved to ISP_GNN_{dataset_name}_{diffusion_model}.csv")

torch.cuda.empty_cache()