import os 
import random
import torch
import numpy as np
from datetime import datetime

from lib.metrics import mae_torch

def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = mae_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def disp(x, name):
    print(f'{name} shape: {x.shape}')

def get_model_params(model_list):
    model_parameters = []
    for m in model_list:
        if m != None:
            model_parameters += list(m.parameters())
    return model_parameters

def get_log_dir(args):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)
    return log_dir 

def load_graph(adj_file, device='cpu'):
    '''Loading graph in form of edge index.'''
    graph = np.load(adj_file)['adj_mx']
    graph = torch.tensor(graph, device=device, dtype=torch.float)
    return graph

# def dwa(L_old, L_new, T=2):
#     '''
#     L_old: list.
#     '''
#     L_old = torch.tensor(L_old, dtype=torch.float32)
#     L_new = torch.tensor(L_new, dtype=torch.float32)
#     N = len(L_new) # task number
#     r =  L_old / L_new
    
#     w = N * torch.softmax(r / T, dim=0)
#     return w.numpy()
def dwa(L_old, L_new, T=2):
    '''
    L_old: list.
    '''
    L_old = torch.tensor(L_old, dtype=torch.float32)
    L_new = torch.tensor(L_new, dtype=torch.float32)
    N = len(L_new) # task number
    
    # 🌟 修复消融实验中 0/0 导致 NaN 的问题
    r = torch.ones_like(L_old) # 默认变化比例为 1.0
    mask = L_new > 1e-6        # 找出非 0 的 loss 项
    r[mask] = L_old[mask] / L_new[mask] # 只有非 0 的项才做除法
    
    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()
