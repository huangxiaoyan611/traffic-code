import warnings 
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')
import yaml 
import argparse
import traceback
import time
import os
# GPU selection is controlled by the launcher through CUDA_VISIBLE_DEVICES.
# Do not hard-code a device here: it prevents independent jobs from using
# different GPUs concurrently.
import torch

# ST-DRA modules
from model.st_dra_model import STDRAModel

from model.trainer import Trainer
from lib.dataloader_st_dra import get_dataloader_st_dra

# 导入原有工具
from lib.utils import init_seed, get_model_params, load_graph 

def model_supervisor(args):
    init_seed(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("--- 加载 ST-DRA 数据集 ---")
    dataloader = get_dataloader_st_dra(
        data_dir=args.data_dir, 
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        test_batch_size=args.test_batch_size,
    )
    graph = load_graph(args.graph_file, device=args.device)
    args.num_nodes = len(graph)
    
    print("--- 初始化 ST-DRA 模型 ---")
    model = STDRAModel(args).to(args.device)
    model_parameters = get_model_params([model])
    optimizer = torch.optim.Adam(
        params=model_parameters, lr=args.lr_init, eps=1.0e-8, weight_decay=0, amsgrad=False
    )

    trainer = Trainer(
    model=model, optimizer=optimizer, dataloader=dataloader, graph=graph, args=args
)
    
    results = None
    try:
        if args.mode == 'train':
            print("--- 🚀 启动 ST-DRA 训练 ---")
            results = trainer.train() 
        elif args.mode == 'test':
            state_dict = torch.load(args.best_path, map_location=torch.device(args.device))
            model.load_state_dict(state_dict['model'])
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                        graph, trainer.logger, trainer.args)
    except Exception as e:
        trainer.logger.info(traceback.format_exc())
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='configs/NYCBike1.yaml', type=str)
    parser.add_argument('--seed', default=None, type=int,
                        help='Optional seed override without editing the YAML file.')
    parser.add_argument('--log_dir', default=None, type=str,
                        help='Optional output-directory override without editing the YAML file.')
    parser.add_argument('--expert_num', default=None, type=int,
                        help='Optional total number of routing experts.')
    parser.add_argument('--topk', default=None, type=int,
                        help='Optional number of experts activated per node.')
    parser.add_argument('--mode', default=None, choices=['train', 'test'],
                        help='Optional mode override for checkpoint re-evaluation.')
    parser.add_argument('--best_path', default=None, type=str,
                        help='Checkpoint path used when --mode test is selected.')
    cli_args = parser.parse_args()
    
    print(f'\n========== 正在运行 ST-DRA 架构 ==========')
    time.sleep(1)
    config_path = os.path.abspath(cli_args.config_filename)
    with open(config_path, encoding='utf-8') as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)
    args = argparse.Namespace(**configs)
    args.config_filename = config_path
    if cli_args.seed is not None:
        args.seed = cli_args.seed
    if cli_args.log_dir is not None:
        args.log_dir = cli_args.log_dir
    if cli_args.expert_num is not None:
        args.expert_num = cli_args.expert_num
    if cli_args.topk is not None:
        args.topk = cli_args.topk
    if cli_args.mode is not None:
        args.mode = cli_args.mode
    if cli_args.best_path is not None:
        args.best_path = os.path.abspath(cli_args.best_path)
    model_supervisor(args)
