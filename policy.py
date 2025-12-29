import torch
import numpy as np
import math
from typing import Dict, Any


class GeneralCQBPolicy:
    """Base class for CQB policies."""
    
    def __init__(self):
        pass

    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """Computes action based on observation.
        
        Returns:
            np.ndarray: [v_surge, v_sway, v_omega, fire, reload]
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
        # Action vector: [surge, sway, omega, fire, reload]
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
            
            # Calculate angle to enemy relative to current heading
            # Since enemy pos is relative (Global Enemy - Global Self),
            # we just need atan2(dy, dx) - current_theta?
            # Wait, o_enemy[0:2] is (Target - Self).
            # The angle of the vector is global angle.
            angle_to_target = math.atan2(ey, ex)
            
            # Angle difference (-pi, pi)
            diff = angle_to_target - theta_global
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            
            # Aim (P-Controller)
            action[2] = np.clip(diff * 5.0, -1.0, 1.0) # Omega
            
            # Fire if aiming well
            if abs(diff) < 0.1:
                action[3] = 1.0
            
            # Approach if too far, backup if too close
            desired_dist = 4.0
            if dist > desired_dist:
                action[0] = 1.0 # Surge forward
            elif dist < 2.0:
                action[0] = -1.0 # Surge backward
                
            # Strafe slightly to be harder to hit
            action[1] = np.sin(self.scan_phase * 5.0) * 0.5
            
            self.scan_phase += 0.1 # Update internal timer

        # --- Logic 3: Random Walk & Scan ---
        else:
            self.move_timer -= 0.05
            if self.move_timer <= 0:
                self.move_timer = np.random.uniform(1.0, 3.0)
                # Random direction
                self.current_move_cmd = np.random.uniform(-0.5, 0.5, size=2)
                self.current_move_cmd[0] += 0.5 # Bias forward
            
            action[0] = self.current_move_cmd[0]
            action[1] = self.current_move_cmd[1]
            
            # Oscillate heading (Scan)
            self.scan_phase += 0.1
            action[2] = math.sin(self.scan_phase) * 0.8
            
        return action