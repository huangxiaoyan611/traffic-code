import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import masked_mae_loss
from model.layers import TemporalHeterogeneityLoss, MLP
from model.aug import aug_topology, aug_traffic

# Import the ST-DRA encoder with temporal decoupling and topology routing.
from model.temporal_decoupling import STDRAEncoder

class PrototypeConsistencyAlignment(nn.Module):
    
    def __init__(self, c_in, prototype_count, tau=0.5):
        super(PrototypeConsistencyAlignment, self).__init__()
        self.tau = tau
        self.prototypes = nn.Parameter(torch.randn(prototype_count, c_in))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, z1, z2):
        z1 = F.normalize(z1.view(-1, z1.size(-1)), dim=1)
        z2 = F.normalize(z2.view(-1, z2.size(-1)), dim=1)
        p = F.normalize(self.prototypes, dim=1)

        out1 = torch.matmul(z1, p.T) / self.tau
        out2 = torch.matmul(z2, p.T) / self.tau

        q1 = F.softmax(out1, dim=-1)
        q2 = F.softmax(out2, dim=-1)

        loss_consist = - torch.mean(torch.sum(q1.detach() * F.log_softmax(out2, dim=-1), dim=-1)) \
                       - torch.mean(torch.sum(q2.detach() * F.log_softmax(out1, dim=-1), dim=-1))

        avg_q = (torch.mean(q1, dim=0) + torch.mean(q2, dim=0)) / 2.0
        loss_collapse = torch.sum(avg_q * torch.log(avg_q + 1e-8)) 

        return loss_consist + loss_collapse


class STDRAModel(nn.Module):
    def __init__(self, args):
        super(STDRAModel, self).__init__()
        self.args = args
        self.use_pca = getattr(args, 'use_pca', True)
        
        # The encoder contains the DTD and DTR modules.
        self.encoder = STDRAEncoder(
            args=args,
            Kt=3, Ks=3, 
            blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
            input_length=args.input_length, 
            num_nodes=args.num_nodes, 
            droprate=args.dropout,
            decomp_kernel=3
        )
        
        self.mlp = MLP(args.d_model, args.d_output)
        
        self.temporal_loss_module = TemporalHeterogeneityLoss(
            args.d_model, args.batch_size, args.num_nodes, args.device
        )
        self.prototype_alignment = PrototypeConsistencyAlignment(
            args.d_model, args.prototype_count, args.prototype_temperature
        )
        
        self.mae = masked_mae_loss(mask_value=5.0)
    
    def forward(self, view1, graph, deterministic=False):
        # View 1: the clean graph passes through the ST-DRA encoder.
        repr1 = self.encoder(view1, graph) 

        # During final testing, the prediction head uses repr1 only.  Avoid
        # constructing a random augmented view so test metrics are fully
        # deterministic without changing the training/validation objective.
        if deterministic:
            return repr1, None

        # Build the data-driven second view.
        s_sim_mx = self.encoder.s_sim_mx.cpu()
        graph2 = aug_topology(s_sim_mx, graph, percent=self.args.percent*2)
        t_sim_mx = self.encoder.t_sim_mx.cpu()
        view2 = aug_traffic(t_sim_mx, view1, percent=self.args.percent)
        
        # View 2: the augmented graph passes through the same encoder.
        repr2 = self.encoder(view2, graph2)
        
        return repr1, repr2

    def predict(self, repr1, repr2=None):
        return self.mlp(repr1)

    def loss(self, repr1, repr2, y_true, scaler, loss_weights):
        y_pred = scaler.inverse_transform(self.predict(repr1))
        y_true = scaler.inverse_transform(y_true)
 
        l_pred = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
                 (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        
        # 时序异质性对比损失
        l_temporal = self.temporal_loss_module(repr1, repr2)
        
        # ---------------- Ablation control: PCA module ----------------
        if self.use_pca:
            l_alignment = self.prototype_alignment(repr1, repr2)
        else:
            # When disabled, the alignment loss does not affect optimization.
            l_alignment = torch.tensor(0.0, device=repr1.device)
        # -------------------------------------------------------
        
        # 若是 tensor 且有 item() 方法则取值，否则直接取值
        sep_loss = [
            l_pred.item() if hasattr(l_pred, 'item') else l_pred, 
            l_temporal.item() if hasattr(l_temporal, 'item') else l_temporal,
            l_alignment.item() if hasattr(l_alignment, 'item') else l_alignment
        ]
        
        total_loss = loss_weights[0] * l_pred + loss_weights[1] * l_temporal + loss_weights[2] * l_alignment
        
        return total_loss, sep_loss
