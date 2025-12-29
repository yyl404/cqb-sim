import torch
import numpy as np
import math
import os
from typing import Dict, Any, Optional

from utils import CFG
from models import ActorCriticRNN


class GeneralCQBPolicy:
    """Base class for CQB policies."""
    
    def __init__(self):
        pass

    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """Computes action based on observation.
        
        Returns:
            np.ndarray: [accel_surge, accel_sway, accel_alpha, fire, reload]
            Values should be suitable for the simulator (mostly normalized or physical).
        """
        raise NotImplementedError


class NaiveCQBPolicy(GeneralCQBPolicy):
    """A heuristic policy that patrols and engages enemies.

    Behavior:
    1. Random Walk: Moves randomly while oscillating heading (scanning).
    2. Engage: If enemy visible, stops, aims, approaches, and fires.
    3. Reload: Immediate reload if empty.
    """

    def __init__(self):
        super().__init__()
        self.scan_timer = 0.0
        self.scan_phase = 0.0
        self.move_timer = 0.0
        self.current_move_cmd = np.zeros(2) # surge, sway

    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        # Action vector: [accel_surge, accel_sway, alpha, fire, reload]
        # Note: Naive policy outputs normalized "velocity-like" commands, 
        # but in the new Dynamics system, we map them to accelerations.
        # To keep naive policy working without rewriting it entirely, 
        # we assume its outputs are "desired directions" and scale them.
        
        action = np.zeros(5, dtype=np.float32)
        
        if obs is None:
            return action

        # Parse Self Obs
        # theta, vx, vy, omega, hp, c, n, r, sigma
        o_self = obs['self']
        theta_global = o_self[0].item()
        ammo = o_self[5].item()
        spare_mags = o_self[6].item()
        
        # Parse Enemy Obs
        o_enemy = obs['enemy']
        has_enemy = o_enemy.numel() > 0
        
        # --- Logic 1: Reload ---
        if ammo <= 0 and spare_mags > 0:
            action[4] = 1.0 # Reload
            return action

        # --- Logic 2: Engage Enemy ---
        if has_enemy:
            # Pick closest enemy
            # Enemy feat: [rel_x, rel_y, theta, vx, vy, omega, hp, r]
            rel_pos = o_enemy[:, 0:2]
            dists = torch.norm(rel_pos, dim=1)
            closest_idx = torch.argmin(dists)
            
            ex, ey = rel_pos[closest_idx, 0].item(), rel_pos[closest_idx, 1].item()
            dist = dists[closest_idx].item()
            
            angle_to_target = math.atan2(ey, ex)
            
            # Angle difference (-pi, pi)
            diff = angle_to_target - theta_global
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            
            # Aim (P-Controller style for angular accel)
            # If diff is large, max alpha.
            action[2] = np.clip(diff * 10.0, -1.0, 1.0) 
            
            # Fire if aiming well
            if abs(diff) < 0.15:
                action[3] = 1.0
            
            # Approach logic (Bang-Bang acceleration)
            desired_dist = 4.0
            if dist > desired_dist:
                action[0] = 1.0 # Surge forward
            elif dist < 2.0:
                action[0] = -1.0 # Surge backward
                
            # Strafe
            action[1] = np.sin(self.scan_phase * 5.0) * 0.8
            
            self.scan_phase += 0.1

        # --- Logic 3: Random Walk & Scan ---
        else:
            self.move_timer -= 0.05
            if self.move_timer <= 0:
                self.move_timer = np.random.uniform(1.0, 3.0)
                self.current_move_cmd = np.random.uniform(-0.5, 0.5, size=2)
                self.current_move_cmd[0] += 0.5 # Bias forward
            
            action[0] = self.current_move_cmd[0]
            action[1] = self.current_move_cmd[1]
            
            # Oscillate heading (Scan)
            self.scan_phase += 0.1
            action[2] = math.sin(self.scan_phase) * 0.5
            
        return action


class RLInferencePolicy(GeneralCQBPolicy):
    """
    Wraps the Hierarchical Actor-Critic Network for inference.
    Manages hidden states (RNN) and tensor conversions.
    """

    def __init__(self, obs_shapes: Dict[str, tuple], state_dim: int, model_path: Optional[str] = None):
        super().__init__()
        self.device = CFG.DEVICE
        
        # Initialize Model Structure
        self.model = ActorCriticRNN(obs_shapes, state_dim).to(self.device)
        self.model.eval()
        
        # Load Weights
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded RL Policy from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}. Using random weights.")
        else:
            print("No model path provided or file not found. Using initialized (random) weights.")

        # RNN Hidden State: (1, 1, Hidden_Dim) for single agent inference
        self.hidden = torch.zeros(1, 1, self.model.hidden_dim, device=self.device)

    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Args:
            obs: Dictionary of tensors for a SINGLE agent.
                 Tensors are on device but lack batch dimension (usually).
        """
        if obs is None:
            # Agent is dead
            return np.zeros(5, dtype=np.float32)

        # 1. Add Batch Dimension (Input is single agent -> Batch 1)
        batched_obs = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                # obs['spatial'] is (L*L), needs to be (1, L*L)
                # obs['self'] is (Dim), needs to be (1, Dim)
                # obs['team'] is (N, Dim), needs to be (1, N, Dim)
                batched_obs[k] = v.unsqueeze(0)
            else:
                batched_obs[k] = v # Should not happen based on simulator

        # 2. Forward Pass
        with torch.no_grad():
            # deterministic=True for evaluation/deployment
            action, _, _, next_hidden = self.model.get_action_for_env(
                batched_obs, 
                self.hidden, 
                deterministic=True
            )
        
        # 3. Update Hidden State
        self.hidden = next_hidden

        # 4. Extract Action (Remove batch dim)
        # action is numpy array (1, 5)
        return action[0]

    def reset(self):
        """Resets the RNN hidden state."""
        self.hidden = torch.zeros(1, 1, self.model.hidden_dim, device=self.device)