import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
import numpy as np
from typing import Dict, Tuple, Any

from utils import CFG


class EntityEncoder(nn.Module):
    """
    Attention-based encoder for variable length entity lists (Team/Enemy).
    """
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        # Batch_first=True ensures input format is (Batch, Seq, Feature)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Max_Entities, Feature_Dim)
        Returns:
            pooled_features: (Batch, Embed_Dim)
        """
        # Handle case with 0 entities (empty sequence)
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], self.embedding.out_features, device=x.device)
            
        # Create mask for padding (assuming zero-padding implies empty entity)
        # Shape: (Batch, Seq)
        mask = (torch.abs(x).sum(dim=-1) == 0)
        
        emb = self.embedding(x)
        
        # Self-Attention
        # key_padding_mask: True means ignore this position
        out, _ = self.attn(emb, emb, emb, key_padding_mask=mask)
        out = self.ln(out)
        
        # Pooling: Max Pool over the sequence dimension
        # We need to mask out the padded values manually before max pooling 
        # to prevent 0s (or other values) from affecting the max if they are padding.
        mask_expanded = mask.unsqueeze(-1).expand_as(out)
        out[mask_expanded] = -1e9 # Set padding to -infinity
        
        pooled = torch.max(out, dim=1)[0]
        
        # If a sample had ALL padding (no entities), max result is -1e9. Reset to 0.
        pooled = torch.where(pooled < -1e8, torch.zeros_like(pooled), pooled)
        
        return pooled


class SpatialEncoder(nn.Module):
    """
    CNN Encoder for Local Grid Map.
    Unflattens the input tensor back to 2D image before processing.
    """
    def __init__(self, spatial_size:int, output_dim: int):
        super().__init__()
        
        # Calculate flatten dimension after CNN layers
        # Layer 1: Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        #   Output size: (spatial_size + 2*2 - 5) / 2 + 1 = (spatial_size - 1) / 2 + 1 = spatial_size // 2
        # Layer 2: Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        #   Output size: (spatial_size//2 + 2*1 - 3) / 2 + 1 = (spatial_size//2 - 1) / 2 + 1 = spatial_size // 4
        # Final: 32 channels * (spatial_size//4) * (spatial_size//4)
        flatten_dim = 32 * (spatial_size // 4) * (spatial_size // 4)
        
        # Input: 1 channel, LxL grid
        # Architecture tuned for small grid sizes (e.g. 20x20 to 100x100)
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Flatten
            nn.Flatten(),
            # Projection
            # Note: The output size of Conv depends on spatial_size. 
            # For spatial_size=20: 20 -> 10 -> 5. 32*5*5 = 800
            # For spatial_size=100: 100 -> 50 -> 25. 32*25*25 = 20000
            nn.Linear(flatten_dim, output_dim), 
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flattened spatial observation (Batch, L*L)
        Returns:
            embedding: (Batch, output_dim)
        """
        # 1. Unflatten: (Batch, L*L) -> (Batch, 1, L, L)
        # Ensure CFG.L matches the data generation
        batch_size = x.shape[0]
        x_img = x.view(batch_size, 1, CFG.L, CFG.L)
        
        return self.net(x_img)


class ActorCriticRNN(nn.Module):
    """
    Hierarchical Policy Network with Embedded PD Controller.
    
    Structure:
    1. Perception (CNN + Attention)
    2. Memory (GRU)
    3. High-Level Policy Head (Target Position/Direction)
    4. Embedded Controller (Converts Target -> Acceleration)
    """
    
    # --- Controller Gains (Tuned for Newtonian Dynamics) ---
    KP_POS: float = 4.0      # Position P-Gain
    KD_POS: float = 2.0      # Velocity D-Gain (Damping)
    KP_ANG: float = 6.0      # Angle P-Gain
    KD_ANG: float = 1.0      # Angular Velocity D-Gain
    
    def __init__(self, obs_shapes: Dict[str, Tuple], state_dim: int):
        super().__init__()
        
        hidden_dim = 256
        self.hidden_dim = hidden_dim
        
        # --- Encoders ---
        self.spatial_enc = SpatialEncoder(spatial_size=CFG.L, output_dim=128)
        self.self_enc = nn.Sequential(
            nn.Linear(obs_shapes['self'][0], 64),
            nn.ReLU()
        )
        self.team_enc = EntityEncoder(input_dim=obs_shapes['team'][-1], embed_dim=64)
        self.enemy_enc = EntityEncoder(input_dim=obs_shapes['enemy'][-1], embed_dim=64)
        
        # --- Recurrent Core ---
        # Fusion: Spatial(128) + Self(64) + Team(64) + Enemy(64)
        fusion_dim = 128 + 64 + 64 + 64
        self.gru = nn.GRU(fusion_dim, hidden_dim, batch_first=True)
        
        # --- High-Level Actor Heads ---
        # Output: [Target_Dx, Target_Dy, Target_Dtheta]
        # We use Tanh to bound the target relative coordinates.
        # This prevents the agent from requesting targets 1000 meters away.
        self.actor_mean = nn.Linear(hidden_dim, 3)
        self.actor_logstd = nn.Parameter(torch.zeros(1, 3) - 0.5) # Initial log_std = -0.5
        
        # Tactics: [Fire_Logit, Reload_Logit]
        self.actor_tactics = nn.Linear(hidden_dim, 2)
        
        # --- Critic Head ---
        # Centralized V(s)
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_actor(self, obs_dict: Dict[str, torch.Tensor], hidden: torch.Tensor) -> Tuple[Tuple[Normal, Bernoulli], torch.Tensor, torch.Tensor]:
        """
        Forward pass to get Action Distributions (High-Level).
        
        Returns:
            (dist_move, dist_tactics): Distributions for PPO
            new_hidden: Updated GRU hidden state
            rnn_out: Features for Critic
        """
        # 1. Encode Features
        s_feat = self.spatial_enc(obs_dict['spatial'])
        self_feat = self.self_enc(obs_dict['self'])
        team_feat = self.team_enc(obs_dict['team'])
        enemy_feat = self.enemy_enc(obs_dict['enemy'])
        
        features = torch.cat([s_feat, self_feat, team_feat, enemy_feat], dim=-1)
        
        # 2. RNN Update
        # Add seq len dim: (Batch, Feat) -> (Batch, 1, Feat)
        out, new_hidden = self.gru(features.unsqueeze(1), hidden)
        rnn_out = out.squeeze(1) # (Batch, Hidden)
        
        # 3. High-Level Movement Distribution
        # Outputs are desired relative displacements
        mu = torch.tanh(self.actor_mean(rnn_out)) 
        
        # Scaling Strategy:
        # Tanh outputs [-1, 1].
        # We want the agent to be able to set targets e.g. +/- 2 meters away.
        # We can perform this scaling here or in the controller.
        # Let's keep the network output normalized [-1, 1] and scale in the controller.
        
        std = torch.exp(self.actor_logstd).expand_as(mu)
        dist_move = Normal(mu, std)
        
        # 4. Tactics Distribution
        logits = self.actor_tactics(rnn_out)
        dist_tactics = Bernoulli(logits=logits)
        
        return (dist_move, dist_tactics), new_hidden, rnn_out

    def evaluate_critic(self, global_state: torch.Tensor, rnn_features: torch.Tensor) -> torch.Tensor:
        """
        Estimate Value V(s) using Global State + Agent Memory.
        """
        cat_input = torch.cat([global_state, rnn_features], dim=-1)
        return self.critic_net(cat_input)

    @staticmethod
    def embedded_controller(obs_self: torch.Tensor, high_level_action: torch.Tensor) -> torch.Tensor:
        """
        Differentiable PD Controller converting High-Level Targets to Acceleration.
        
        Args:
            obs_self: (Batch, Self_Dim) 
                      Indices from simulator: 0:theta, 1:vx, 2:vy, 3:omega
                      (Based on o_self construction in simulator.py)
            high_level_action: (Batch, 3) -> [dx_cmd, dy_cmd, dtheta_cmd]
                               Values are in [-1, 1] (Tanh output).
        
        Returns:
            low_level_action: (Batch, 3) -> [accel_surge_norm, accel_sway_norm, alpha_norm]
                              Values clipped to [-1, 1] for Simulator.
        """
        # --- 1. Tuning & Scaling Constants ---
        # Scale the [-1, 1] network output to physical units
        MAX_TARGET_DIST = 3.0 # meters (Max lookahead distance)
        MAX_TARGET_ANGLE = 1.5 # radians (~90 degrees)
        
        # --- 2. Extract State ---
        # simulator.py: o_self = [theta, vx, vy, omega, hp, c, n, r, sigma]
        theta_curr = obs_self[:, 0]
        vx_global  = obs_self[:, 1]
        vy_global  = obs_self[:, 2]
        omega_curr = obs_self[:, 3]
        
        # --- 3. Extract High-Level Goals ---
        target_dx = high_level_action[:, 0] * MAX_TARGET_DIST
        target_dy = high_level_action[:, 1] * MAX_TARGET_DIST
        target_dtheta = high_level_action[:, 2] * MAX_TARGET_ANGLE
        
        # --- 4. Transform Global Velocity to Local Frame ---
        cos_t = torch.cos(theta_curr)
        sin_t = torch.sin(theta_curr)
        
        # Rotation Matrix:
        # [ v_surge ]   [  cos   sin ] [ vx_global ]
        # [ v_sway  ] = [ -sin   cos ] [ vy_global ]
        v_surge =  vx_global * cos_t + vy_global * sin_t
        v_sway  = -vx_global * sin_t + vy_global * cos_t
        
        # --- 5. PD Control Logic (Acceleration Command) ---
        # Force = Kp * Error - Kd * Velocity
        # Error in local frame is simply (target_dx - 0), (target_dy - 0)
        
        accel_surge = ActorCriticRNN.KP_POS * target_dx - ActorCriticRNN.KD_POS * v_surge
        accel_sway  = ActorCriticRNN.KP_POS * target_dy - ActorCriticRNN.KD_POS * v_sway
        
        # Angular Control
        alpha = ActorCriticRNN.KP_ANG * target_dtheta - ActorCriticRNN.KD_ANG * omega_curr
        
        # --- 6. Normalization for Simulator ---
        # The simulator expects inputs in [-1, 1], which it then multiplies by CFG.ACCEL_MAX.
        # So we normalize our calculated acceleration by ACCEL_MAX.
        
        accel_surge_norm = accel_surge / CFG.ACCEL_MAX
        accel_sway_norm  = accel_sway / CFG.ACCEL_MAX
        alpha_norm       = alpha / CFG.ALPHA_MAX
        
        # Clip to ensure valid range
        accel_surge_norm = torch.clamp(accel_surge_norm, -1.0, 1.0)
        accel_sway_norm  = torch.clamp(accel_sway_norm, -1.0, 1.0)
        alpha_norm       = torch.clamp(alpha_norm, -1.0, 1.0)
        
        return torch.stack([accel_surge_norm, accel_sway_norm, alpha_norm], dim=1)

    def get_action_for_env(self, obs_dict: Dict[str, torch.Tensor], hidden: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full inference pipeline: Obs -> HighLevel -> Controller -> LowLevel(Env).
        
        Returns:
            actions_np: Numpy array of Low-Level actions for Simulator (Batch, 5)
            high_level_action: Tensor of chosen high level goals (Batch, 5)
            log_probs: Tensor (Batch,)
            next_hidden: Tensor
        """
        # 1. Forward Policy
        (dist_move, dist_tactics), next_hidden, _ = self.forward_actor(obs_dict, hidden)
        
        # 2. Sample High-Level Actions
        if deterministic:
            hl_move = dist_move.mean
            # For Bernoulli, mode is 1 if p > 0.5
            hl_tactics = (dist_tactics.probs > 0.5).float()
        else:
            hl_move = dist_move.sample()
            hl_tactics = dist_tactics.sample()
        
        log_probs = dist_move.log_prob(hl_move).sum(-1) + dist_tactics.log_prob(hl_tactics).sum(-1)
        
        # 3. Pass through Controller
        # hl_move contains [dx, dy, dtheta] (Normalized Tanh space)
        # obs_dict['self'] contains the state needed for PD
        ll_move = self.embedded_controller(obs_dict['self'], hl_move)
        
        # 4. Assemble Full Action Vector
        # Low Level: [accel_surge, accel_sway, alpha, fire, reload]
        # hl_tactics: [fire, reload]
        full_action = torch.cat([ll_move, hl_tactics], dim=1)
        
        # 5. Assemble High Level (to store in buffer for PPO)
        # We store the actions that generated the log_probs
        full_high_level = torch.cat([hl_move, hl_tactics], dim=1)
        
        return full_action.cpu().numpy(), full_high_level, log_probs, next_hidden