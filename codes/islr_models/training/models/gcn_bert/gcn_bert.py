import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class GC_Layer(nn.Module):
    def __init__(self, hidden_features, num_joints):
        super(GC_Layer, self).__init__()
        self.fc = nn.Linear(hidden_features, hidden_features)
        self.adj = nn.Parameter(torch.eye(num_joints))  # 96 joints, learnable adjacency
        nn.init.xavier_uniform_(self.adj)

    def forward(self, x):
        x = x.float()
        out = torch.matmul(self.adj, x)
        out = self.fc(out)  # Linear layer
        out = torch.tanh(out)  # Non-linearity (Tanh)
        return out

class GCNBlock(nn.Module):
    def __init__(self, hidden_features=2, num_layers=2, num_joints=96, dropout=0.5):
        super(GCNBlock, self).__init__()
        self.num_joints = num_joints
        self.hidden_features = hidden_features

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(GC_Layer(hidden_features, num_joints))
            self.bns.append(nn.BatchNorm1d(num_joints * hidden_features))
            self.acts.append(nn.Tanh())
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x):
        for gc, bn, act, do in zip(self.layers, self.bns, self.acts, self.dropouts):
            x = gc(x)  # Shape: [B, N, F]
            #assert not torch.isnan(x).any(), "NaN after GC_Layer"
            b, n, f = x.shape
            x = bn(x.view(b, -1)).view(b, n, f)
            x = act(x)
            x = do(x)
        return x

class StackedGCN(nn.Module):
    def __init__(self, hidden_features=2, num_blocks=2, layers_per_block=10, num_joints=96):
        super(StackedGCN, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = GCNBlock(
                hidden_features,
                num_layers=layers_per_block,
                num_joints=num_joints
            )
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            residual = x
            x = block(x)
            #assert not torch.isnan(x).any(), "NaN inside StackedGCN block"
            x = x + residual  
        return x

class SpatialGCN(nn.Module):
    def __init__(self, hidden_features=2, num_blocks=2, layers_per_block=10, num_joints=96):
        super(SpatialGCN, self).__init__()
        self.gcn = StackedGCN(
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            num_joints=num_joints
        )

    def forward(self, x):
        batch_size, frames, joints, coords = x.shape
        x = x.view(batch_size * frames, joints, coords)
        x = self.gcn(x)
        x = x.view(batch_size, frames, joints, -1)  # [batch_size, frames, joints, hidden_features]
        return x

class LearnedPosEmbed(nn.Module):
    def __init__(self, seq_len=50, dropout: float = 0.1):
        super().__init__()
        self.positional_bias = nn.Parameter(torch.randn(1, seq_len, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_bias[:, :x.size(1), :]
        return self.dropout(x)

class GCN_BERT(nn.Module):
    def __init__(self, num_classes, hidden_features=2, seq_len=50, num_joints=96,nhead=4):
        super(GCN_BERT, self).__init__()
        self.gcn = SpatialGCN(hidden_features=2, num_blocks=2, layers_per_block=3, num_joints=num_joints)

        self.positional_encoder = LearnedPosEmbed(seq_len=seq_len, dropout=0.1)
        encoder_layer = TransformerEncoderLayer(
            d_model=num_joints*hidden_features,
            nhead=nhead,
            dim_feedforward=num_joints*hidden_features*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=4)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_features*num_joints))

        self.fc_spatial = nn.Linear(num_joints*hidden_features, num_classes)
        self.fc_temporal = nn.Linear(num_joints*hidden_features, num_classes)

    def forward(self, x, padding_mask=None):
        #Reshaping input  # input: [batch, T, K*2]
        x = x.float()
        batch_size, frames, joints_times_2 = x.shape    # The last dimension of (joints, coords_xy) had previously been flattened
        joints = joints_times_2 // 2
        x = x.view(batch_size, frames, joints, 2) # 2 because x,y coordinates
        s = self.gcn(x)     # s: [batch, T, K, F]  
        spatial_mean = s.mean(dim=1).reshape(batch_size, -1) # [batch, K*F]

        spatial_features = s.view(batch_size, frames, joints*2)# [batch, frames, 96*2]
        p = self.positional_encoder(spatial_features) # S concatenated with learned position embeddings

        cls_token = self.cls_token.expand(batch_size, 1, -1)
        p = torch.cat([cls_token, p], dim=1) # Adding classification token at the start of input
        if padding_mask is not None:
            cls_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_padding, padding_mask], dim=1)

        p = self.encoder(p, src_key_padding_mask=padding_mask)
        cls_token = p[:, 0] #y_cls
        spatial_logits = self.fc_spatial(spatial_mean) #U 
        temporal_logits = self.fc_temporal(cls_token)  #V

        out = spatial_logits + temporal_logits # U + V
        return out