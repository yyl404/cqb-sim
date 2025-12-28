import torch
import torch.nn as nn
import numpy as np

class MapEncoder(nn.Module):
    """
    使用 CNN 提取局部地图信息。
    输入为 (Batch, L*L) 的展平向量，内部 reshape 为 (Batch, 1, L, L) 进行卷积。
    """
    def __init__(self, map_size=20, embed_dim=256):
        super().__init__()
        self.map_size = map_size
        
        # 一个轻量级的 CNN 结构
        self.cnn = nn.Sequential(
            # Input: (B, 1, 20, 20)
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (16, 10, 10)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (32, 5, 5)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), # -> (64, 3, 3)
            nn.ReLU()
        )
        
        # 计算 CNN 输出展平后的维度: 64 * 3 * 3 = 576
        self.flat_dim = 64 * 3 * 3
        
        # 投影到 Transformer 的维度
        self.proj = nn.Linear(self.flat_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (Batch, 400)
        B = x.shape[0]
        # Reshape to image: (B, 1, 20, 20)
        x = x.view(B, 1, self.map_size, self.map_size)
        
        feat = self.cnn(x)
        feat = feat.flatten(1) # (B, 576)
        out = self.proj(feat)  # (B, embed_dim)
        out = self.norm(out)
        return out

class EntityEncoder(nn.Module):
    """
    使用 MLP 处理智能体观测向量 (Self, Friend, Enemy)。
    """
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class CQBTransformerPolicy(nn.Module):
    """
    基于 Transformer 的 CQB 策略网络。
    融合 CNN 地图特征 + MLP 实体特征。
    """
    def __init__(
        self, 
        map_size=20,
        self_dim=9, 
        friend_dim=10, 
        enemy_dim=8, 
        embed_dim=256, 
        nhead=4, 
        num_layers=2,
        action_dim=5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. 模态编码器 (Encoders)
        self.map_encoder = MapEncoder(map_size, embed_dim)
        
        self.self_proj = EntityEncoder(self_dim, embed_dim)
        self.friend_proj = EntityEncoder(friend_dim, embed_dim) # 对应 friend_dim
        self.enemy_proj = EntityEncoder(enemy_dim, embed_dim)   # 对应 enemy_dim
        
        # 类型嵌入 (Type Embeddings) - 可选，用于帮助 Transformer 区分不同类型的 Token
        # 0: Map, 1: Self, 2: Friend, 3: Enemy
        self.type_embed = nn.Embedding(4, embed_dim)

        # 2. Transformer Backbone
        # batch_first=True: 输入形状为 (Batch, Seq_Len, Dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Heads (Actor & Critic)
        # 我们使用 Transformer 输出序列中对应 "Self" 的 token 进行决策
        self.actor_mean = nn.Linear(embed_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) # 可学习的 log_std
        self.critic = nn.Linear(embed_dim, 1)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, obs_dict):
        """
        Args:
            obs_dict: 包含 batch 数据的字典
                - 'self': (B, 9)
                - 'spatial': (B, 400)
                - 'team': (B, N_team, 10)
                - 'enemy': (B, N_enemy, 8)
        Returns:
            action_mean: (B, 5)
            action_std: (B, 5)
            value: (B, 1)
        """
        # 1. 提取特征并投影到相同维度
        # Map: (B, embed_dim) -> unsqueeze -> (B, 1, embed_dim)
        map_emb = self.map_encoder(obs_dict['spatial']).unsqueeze(1)
        
        # Self: (B, embed_dim) -> unsqueeze -> (B, 1, embed_dim)
        self_emb = self.self_proj(obs_dict['self']).unsqueeze(1)
        
        # Team: (B, N_team, 10) -> (B, N_team, embed_dim)
        team_emb = self.friend_proj(obs_dict['team'])
        
        # Enemy: (B, N_enemy, 8) -> (B, N_enemy, embed_dim)
        enemy_emb = self.enemy_proj(obs_dict['enemy'])
        
        # 2. 添加类型嵌入 (Type Embedding)
        # 为每个模态添加固定的标识向量，帮助网络区分来源
        B = map_emb.shape[0]
        
        map_emb = map_emb + self.type_embed(torch.tensor(0, device=map_emb.device))
        self_emb = self_emb + self.type_embed(torch.tensor(1, device=self_emb.device))
        team_emb = team_emb + self.type_embed(torch.tensor(2, device=team_emb.device))
        enemy_emb = enemy_emb + self.type_embed(torch.tensor(3, device=enemy_emb.device))
        
        # 3. 拼接序列 (Sequence Concatenation)
        # Sequence: [Map, Self, Team_1...Team_N, Enemy_1...Enemy_M]
        # Shape: (B, 1 + 1 + N_team + N_enemy, embed_dim)
        sequence = torch.cat([map_emb, self_emb, team_emb, enemy_emb], dim=1)
        
        # 4. Transformer 处理
        # TransformerEncoder 不需要 mask (除非有 Padding，这里假设 batch 内所有 agent 数量一致，或由 PPO padding 处理)
        # 如果需要处理变长序列（例如死去的队友），可以在这里传入 src_key_padding_mask
        out_seq = self.transformer(sequence)
        
        # 5. 聚合/提取特征 (Pooling / Token Selection)
        # 策略 1: 使用 "Self" Token 的输出 (index 1) 作为决策依据
        # 因为 Self token 经过 attention 已经聚合了 map, team, enemy 的信息
        decision_token = out_seq[:, 1, :] 
        
        # 策略 2 (备选): Global Average Pooling
        # decision_token = out_seq.mean(dim=1)
        
        # 6. Heads 输出
        action_mean = torch.tanh(self.actor_mean(decision_token)) # [-1, 1]
        
        # 广播 log_std 到 batch 大小
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        value = self.critic(decision_token)
        
        return action_mean, action_std, value