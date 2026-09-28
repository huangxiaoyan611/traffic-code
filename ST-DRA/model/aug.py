import copy
import numpy as np 
import torch 

def sim_global(flow_data, sim_type='cos'):
    """Calculate the global similarity of traffic flow data.
    :param flow_data: tensor, original flow [n,l,v,c] or location embedding [n,v,c]
    :param type: str, type of similarity, attention or cosine. ['att', 'cos']
    :return sim: tensor, symmetric similarity, [v,v]
    """
    if len(flow_data.shape) == 4:
        n,l,v,c = flow_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1 # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
    elif len(flow_data.shape) == 3:
        n,v,c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1 # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
    else:
        raise ValueError('sim_global only support shape length in [3, 4] but got {}.'.format(len(flow_data.shape)))

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5 
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')
    
    return sim


def aug_topology(sim_mx, input_graph, percent=0.2):
    device = input_graph.device
    sim_mx = sim_mx.to(device)
    
    ## edge dropping
    drop_percent = percent / 2
    index_list = input_graph.nonzero() 
    edge_num = int(index_list.shape[0] / 2)
    edge_mask = (input_graph > 0).tril(diagonal=-1)
    add_drop_num = int(edge_num * drop_percent / 2) 
    aug_graph = input_graph.clone() # 使用 clone 代替 deepcopy

    # 优化点：在 GPU 上计算概率
    drop_prob = torch.softmax(sim_mx[edge_mask], dim=0)
    # torch.multinomial 比 np.random.choice 在 GPU 上快得多
    drop_list = torch.multinomial(1.0 - drop_prob, add_drop_num, replacement=False)
    drop_index = index_list[drop_list]
    
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = 0
    aug_graph[drop_index[:, 1], drop_index[:, 0]] = 0

    ## edge adding
    node_num = input_graph.shape[0]
    # 这里的 meshgrid 在大图时很占内存，如果节点数 > 5000 需注意
    x, y = torch.meshgrid(torch.arange(node_num, device=device), 
                          torch.arange(node_num, device=device), indexing='ij')
    mask = y < x
    all_possible_edges = torch.stack([x[mask], y[mask]], dim=1)

    add_prob_all = sim_mx[mask]
    add_prob = torch.softmax(add_prob_all, dim=0)
    add_list = torch.multinomial(add_prob, add_drop_num, replacement=False)
    
    add_index = all_possible_edges[add_list]
    aug_graph[add_index[:, 0], add_index[:, 1]] = 1
    aug_graph[add_index[:, 1], add_index[:, 0]] = 1
    
    return aug_graph

def aug_traffic(t_sim_mx, flow_data, percent=0.2):
    l, n, v = t_sim_mx.shape
    device = flow_data.device
    mask_num = int(n * l * v * percent)
    aug_flow = flow_data.clone()

    # 确保在同一设备上计算掩码
    mask_prob = (1. - t_sim_mx.permute(1, 0, 2).reshape(-1))
    mask_prob = mask_prob / mask_prob.sum()
    
    # 采样掩码索引
    mask_list = torch.multinomial(mask_prob, mask_num, replacement=False)

    # 快速填充掩码
    aug_flow.view(-1, flow_data.shape[-1])[mask_list] = 0

    return aug_flow





    