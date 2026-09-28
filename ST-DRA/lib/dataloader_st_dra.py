import torch
from lib.dataloader import get_dataloader as original_get_dataloader

def get_dataloader_st_dra(data_dir, dataset, batch_size, test_batch_size, scalar_type='Standard'):
    # 1. 复用原版的数据加载逻辑
    dataloader = original_get_dataloader(data_dir, dataset, batch_size, test_batch_size, scalar_type)
    
    # 2. 提取训练集的全量数据 (形如 [Total_Samples, Seq_Len, Num_Nodes, Features])
    x_train = dataloader['train'].dataset.tensors[0]
    
    # 3. ST-DRA 的历史锚点：计算全局历史均值，作为常态交通模式模板
    # 在时间维度(dim=0)上取平均，得到一个代表"常态交通模式"的标准模板
    x_history_anchor = x_train.mean(dim=0, keepdim=True) # 形状: [1, Seq_Len, Num_Nodes, Features]
    
    # 将其挂载到 dataloader 字典中供 Trainer 使用
    dataloader['x_history_anchor'] = x_history_anchor
    
    return dataloader
