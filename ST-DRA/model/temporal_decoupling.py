

# ST-DRA dynamic topology routing implementation.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import (
    TemporalConvLayer,
    SpatioConvLayer,
    Pooler,
    DynamicTopologyRoutingLayer,
)
from model.aug import sim_global

class MovingAverage(nn.Module):
    """
    DTD component 1: moving average for 4D spatiotemporal tensors.
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: (N, L, V, C)
        N, L, V, C = x.shape
        # 转换维度以适应 1D Pooling: (N*V*C, 1, L)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(N * V * C, 1, L)

        # 在时间序列两端进行 Padding
        front = x_reshaped[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x_reshaped[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_padded = torch.cat([front, x_reshaped, end], dim=2)

        x_avg = self.avg(x_padded) # (N*V*C, 1, L)
        # 还原回原有的 (N, L, V, C) 形状
        x_avg = x_avg.reshape(N, V, C, L).permute(0, 3, 1, 2)
        return x_avg

class TemporalDecoupling(nn.Module):
    """
    DTD component 2: separate trend and seasonal/residual components.
    """
    def __init__(self, kernel_size):
        super(TemporalDecoupling, self).__init__()
        self.moving_average = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_average(x)
        res = x - moving_mean
        return res, moving_mean

class STDRAEncoder(nn.Module):
    """
    ST-DRA spatiotemporal encoder with DTD and DTR ablation controls.
    """
    def __init__(self, args, Kt, Ks, blocks, input_length, num_nodes, droprate=0.1, decomp_kernel=3):
        super(STDRAEncoder, self).__init__()
        self.Ks = Ks
        
        # The complete ST-DRA configuration is enabled by default.
        self.use_dtd = getattr(args, 'use_dtd', True)
        self.use_dtr = getattr(args, 'use_dtr', True)

        # Expose the expert-pool size and Top-k as configuration parameters
        # for the routing sensitivity experiments.  The defaults preserve the
        # original ST-DRA setting.
        self.expert_num = int(getattr(args, 'expert_num', 4))
        self.topk = int(getattr(args, 'topk', 2))
        if self.expert_num < 1:
            raise ValueError('expert_num must be positive')
        if self.topk < 1 or self.topk > self.expert_num:
            raise ValueError('topk must satisfy 1 <= topk <= expert_num')

        #  加上这行高亮打印代码 
        print(f"\n{'='*50}")
        print(f"🚀 [构建模型架构] DTD: {self.use_dtd} | DTR: {self.use_dtr} | K: {self.expert_num} | Top-k: {self.topk}")
        print(f"{'='*50}\n")
        
        c = blocks[0]
        
        # ---------------- DTD ablation control ----------------
        if self.use_dtd:
            self.decomp = TemporalDecoupling(kernel_size=decomp_kernel)
            self.tconv11_seasonal = TemporalConvLayer(Kt, c[0], c[1], "GLU")
            self.tconv11_trend = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        else:
            # Without DTD, use a single temporal convolution branch.
            self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        # -------------------------------------------------------

        self.pooler = Pooler(input_length - (Kt - 1), c[1])
        self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv13 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        c = blocks[1]
        self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        
        # ---------------- DTR ablation control ----------------
        if self.use_dtr:
            self.sconv22 = DynamicTopologyRoutingLayer(
                c_in=c[1], c_out=c[1], node_num=num_nodes,
                expert_num=self.expert_num, topk=self.topk
            )
        else:
            # Without DTR, fall back to the static graph convolution.
            self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
        # ----------------------------------------------------------
        
        self.tconv23 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)
        
        self.s_sim_mx = None
        self.t_sim_mx = None
        
        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)
        self.receptive_field = input_length + Kt - 1

    def forward(self, x0, graph):
        lap_mx = self._cal_laplacian(graph)
        Lk = self._cheb_polynomial(lap_mx, self.Ks)
        
        in_len = x0.size(1) 
        if in_len < self.receptive_field:
            x_pad = F.pad(x0, (0, 0, 0, 0, self.receptive_field - in_len, 0))
        else:
            x_pad = x0
            
        # ---------------- DTD forward path ----------------
        # if self.use_dtd:
        #     seasonal_init, trend_init = self.decomp(x_pad)
        #     seasonal_in = seasonal_init.permute(0, 3, 1, 2) 
        #     trend_in = trend_init.permute(0, 3, 1, 2)       
        #     x_seasonal = self.tconv11_seasonal(seasonal_in)
        #     x_trend = self.tconv11_trend(trend_in)
        #     x = x_seasonal + x_trend 
        # ---------------- DTD forward path ----------------
        if self.use_dtd:
            seasonal_init, trend_init = self.decomp(x_pad)
            
            # 🌟 移到这里：刚分解完就保存，绝对不会错！
            if not self.training:
                self.vis_original = x_pad.detach().cpu()
                self.vis_trend = trend_init.detach().cpu()
                self.vis_seasonal = seasonal_init.detach().cpu()
                
            seasonal_in = seasonal_init.permute(0, 3, 1, 2) 
            trend_in = trend_init.permute(0, 3, 1, 2)       
            x_seasonal = self.tconv11_seasonal(seasonal_in)
            x_trend = self.tconv11_trend(trend_in)
            x = x_seasonal + x_trend 
        else:
            x_in = x_pad.permute(0, 3, 1, 2)
            x = self.tconv11(x_in)
        # -----------------------------------------------------------
        # else:
        #     x_in = x_pad.permute(0, 3, 1, 2)
        #     x = self.tconv11(x_in)
        # # -----------------------------------------------------------

        x, x_agg, self.t_sim_mx = self.pooler(x)
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')

        ## ST block 1
        x = self.sconv12(x, Lk) 
        x = self.tconv13(x)  
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        ## ST block 2
        x = self.tconv21(x)
        
        # DTR ignores Lk; the static fallback uses it.
        x = self.sconv22(x, Lk) 
        
        x = self.tconv23(x)
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ## out block
        x = self.out_conv(x) 
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1))) 
        return x
        



    # 以下保持原有拉普拉斯和切比雪夫多项式计算不变
    def _cheb_polynomial(self, laplacian, K):
        N = laplacian.size(0)  
        multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float) 
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]
        return multi_order_laplacian

    def _cal_laplacian(self, graph):
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        graph = graph + I 
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        L = I - torch.mm(torch.mm(D, graph), D)
        return L
