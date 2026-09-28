# ST-DRA layer definitions.
import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

from model.aug import sim_global

########################################
## Spatial Heterogeneity Modeling
########################################
class PrototypeAssignmentModel(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''
    def __init__(self, c_in, prototype_count, batch_size, tau=0.5):
        super(PrototypeAssignmentModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, prototype_count, bias=False)
        
        self.tau = tau
        self.d_model = c_in
        self.batch_size = batch_size

        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        """Compute the contrastive loss of batched data.
        :param z1, z2 (tensor): shape nlvc
        :param loss: contrastive loss
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)
        
        # l2norm avoids numerical issues in balanced prototype assignment.
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model))) # nd -> nk, assignment q, embedding z
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model))) # nd -> nk
        with torch.no_grad():
            q1 = balanced_prototype_assignment(zc1.detach())
            q2 = balanced_prototype_assignment(zc2.detach())
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2
    
@torch.no_grad()
def balanced_prototype_assignment(out, epsilon=0.05, iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

#######################################
# Temporal Heterogeneity Modeling
#######################################
class TemporalHeterogeneityLoss(nn.Module):
    '''Temporal heterogeneity modeling in a contrastive manner.
    '''
    def __init__(self, c_in, batch_size, num_nodes, device):
        super(TemporalHeterogeneityLoss, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in)) # representation weights
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in)) 
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        
        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        self.lbl = lbl.to(device)
        
        self.n = batch_size

    def forward(self, z1, z2):
        '''
        :param z1, z2 (tensor): shape nlvc, i.e., (batch_size, seq_len, num_nodes, feat_dim)
        :return loss: loss of generative branch. nclv
        '''
        h = (z1 * self.W1 + z2 * self.W2).squeeze(1) # nlvc->nvc
        s = self.read(h) # s: summary of h, nc

        # select another region in batch
        idx = torch.randperm(self.n)
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, self.lbl)
        return loss

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        '''Apply an average on graph.
        :param h: hidden representation, (batch_size, num_nodes, feat_dim)
        :return s: summary, (batch_size, feat_dim)
        '''
        s = torch.mean(h, dim=1)
        s = self.sigm(s) 
        return s

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1) # similar to score of CPC

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        '''
        :param s: summary, (batch_size, feat_dim)
        :param h_rl: real hidden representation (w.r.t summary),
            (batch_size, num_nodes, feat_dim)
        :param h_fk: fake hidden representation
        :return logits: prediction scores, (batch_size, num_nodes, 2)
        '''
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        # score of real and fake, (batch_size, num_nodes)
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2) 
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits

####################
## Spatial & Temporal Convolutions
####################
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''Align the input and output.
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """
        :param x: (n,c,l,v)
        :return: (n,c,l-kt+1,v)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]  
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  
        return torch.relu(self.conv(x) + x_in)  

class SpatioConvLayer(nn.Module):
    """ 原版静态图卷积层 """
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks)) # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)  
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b 
        x_in = self.align(x) 
        return torch.relu(x_gc + x_in)

class DynamicTopologyRoutingLayer(nn.Module):
    """
    ST-DRA's node-level sparse dynamic topology routing layer.
    """
    def __init__(self, c_in, c_out, node_num, expert_num=4, topk=2, graph_gen_dim=32):
        super(DynamicTopologyRoutingLayer, self).__init__()
        self.expert_num = expert_num
        self.topk = topk
        self.node_num = node_num
        
        # 1. Dynamic graph topology expert pool.
        # 每个专家负责生成一种特定的路网拓扑状态
        self.E1 = nn.Parameter(torch.randn(expert_num, node_num, graph_gen_dim))
        self.E2 = nn.Parameter(torch.randn(expert_num, graph_gen_dim, node_num))
        
        # 2. Node-level routing network.
        self.routing_network = nn.Sequential(
            nn.Linear(c_in, graph_gen_dim),
            nn.ReLU(),
            nn.Linear(graph_gen_dim, expert_num)
        )
        
        # 3. 每个图专家的独立特征提取权重
        self.expert_weights = nn.Parameter(torch.FloatTensor(expert_num, c_in, c_out))
        nn.init.kaiming_uniform_(self.expert_weights, a=math.sqrt(5))
        
        self.align = Align(c_in, c_out)

    def forward(self, x, Lk=None):
        
        # --- Step 1: sparse node-level routing ---
        # # 聚合时间维度的特征来进行路由判断
        x_route = x.mean(dim=2).transpose(1, 2) # (B, V, C)
        route_logits = self.routing_network(x_route)     # (B, V, expert_num)

    
        # 计算每个专家的权重，并进行 Top-K 掩码隔离
        route_probs = F.softmax(route_logits, dim=-1)
        if not self.training:
            self.vis_route_probs = route_probs.detach().cpu()
        topk_probs, topk_indices = torch.topk(route_probs, self.topk, dim=-1)
        mask = torch.zeros_like(route_probs).scatter_(-1, topk_indices, 1.0)
        
        # 过滤掉没被选中的专家，并重新归一化权重
        route_weights = route_probs * mask
        route_weights = route_weights / (route_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # --- Step 2: parallel topology-expert graph convolutions ---
        # 动态生成 expert_num 个邻接矩阵: (expert_num, V, V)
        adj_matrices = torch.softmax(torch.matmul(self.E1, self.E2), dim=-1)
        
        out_experts = []
        for i in range(self.expert_num):
            adj = adj_matrices[i]               # (V, V)
            weight = self.expert_weights[i]     # (C, C_out)
            
            # 空间信息聚合: x_c = A * X
            # 为了计算方便，将 x 转置为 (B, L, C, V)
            x_c = torch.einsum("vw, blcw -> blcv", adj, x.transpose(1, 2))
            
            # 特征变换: X_out = X_c * W
            x_out = torch.einsum("co, blcv -> blov", weight, x_c).transpose(1, 2) # (B, C_out, L, V)
            out_experts.append(x_out)
            
        # (expert_num, B, C_out, L, V)
        out_experts = torch.stack(out_experts, dim=0)
        
        # --- Step 3: aggregate with routing weights ---
        # route_weights: (B, V, expert_num) 扩展以匹配 out_experts
        route_weights_expanded = route_weights.permute(2, 0, 1).unsqueeze(2).unsqueeze(3)
        final_out = torch.sum(out_experts * route_weights_expanded, dim=0) # (B, C_out, L, V)
        
        return torch.relu(final_out + self.align(x))

        


class Pooler(nn.Module):
    '''Pooling the token representations of region time series into the region level.
    '''
    def __init__(self, n_query, d_model, agg='avg'):
        """
        :param n_query: number of query
        :param d_model: dimension of model 
        """
        super(Pooler, self).__init__()

        ## attention matirx
        self.att = FCLayer(d_model, n_query) 
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2) # softmax on the seq_length dim, nclv

        self.d_model = d_model
        self.n_query = n_query 
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')
        
    def forward(self, x):
        """
        :param x: key sequence of region embeding, nclv
        :return x: hidden embedding used for conv, ncqv
        :return x_agg: region embedding for spatial similarity, nvc
        :return A: temporal attention, lnv
        """
        x_in = self.align(x)[:, :, -self.n_query:, :] # ncqv
        # calculate the attention matrix A using key x   
        A = self.att(x) # x: nclv, A: nqlv 
        A = F.softmax(A, dim=2) # nqlv

        # calculate region embeding using attention matrix A
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2) # ncqv->ncv
        x_agg = torch.einsum('ncv->nvc', x_agg) # ncv->nvc

        # calculate the temporal simlarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2)) # A: lnqv->lnv
        return torch.relu(x + x_in), x_agg.detach(), A.detach()

########################################
## An MLP predictor
########################################
class MLP(nn.Module):
    def __init__(self, c_in, c_out): 
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2))) # nlvc->nclv
        x = self.fc2(x).permute(0, 2, 3, 1) # nclv->nlvc
        return x

class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)  

    def forward(self, x):
        return self.linear(x)

####################
## Reference ST encoder with dynamic topology routing.
####################
class STEncoder(nn.Module):
    def __init__(self, Kt, Ks, blocks, input_length, num_nodes, droprate=0.1):
        super(STEncoder, self).__init__()
        self.Ks=Ks
        c = blocks[0]
        self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        self.pooler = Pooler(input_length - (Kt - 1), c[1])
    
        self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv13 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        c = blocks[1]
        self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        
        # ST block 2 uses the ST-DRA dynamic topology routing layer.
        self.sconv22 = DynamicTopologyRoutingLayer(c_in=c[1], c_out=c[1], node_num=num_nodes, expert_num=4, topk=2)
        
        self.tconv23 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)
        
        self.s_sim_mx = None
        self.t_sim_mx = None
        
        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)
        self.receptive_field = input_length + Kt -1

    def forward(self, x0, graph):
        lap_mx = self._cal_laplacian(graph)
        Lk = self._cheb_polynomial(lap_mx, self.Ks)
        
        in_len = x0.size(1) # x0, nlvc
        if in_len < self.receptive_field:
            x = F.pad(x0, (0,0,0,0,self.receptive_field-in_len,0))
        else:
            x = x0
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes), nclv 
        
        ## ST block 1 (保持原样)
        x = self.tconv11(x)    # nclv
        x, x_agg, self.t_sim_mx = self.pooler(x)
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')

        x = self.sconv12(x, Lk)   # nclv
        x = self.tconv13(x)  
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        ## ST block 2 (deep features with dynamic topology routing)
        x = self.tconv21(x)
        # The dynamic routing layer ignores Lk and infers topology from features.
        x = self.sconv22(x, Lk) 
        x = self.tconv23(x)
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ## out block
        x = self.out_conv(x) # ncl(=1)v
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1))) # nlvc
        return x # nl(=1)vc

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
