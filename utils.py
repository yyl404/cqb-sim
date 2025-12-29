import math
import torch
from typing import Dict, List, Optional


class CQBConfig:
    """Configuration parameters for the CQB Simulator."""
    
    # --- Environment Dimensions ---
    H: int = 100
    W: int = 100
    L: int = 100           # Local observation crop size
    DT: float = 0.05       # Time step (s)
    
    # Auto-detect device
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Physics Constraints (Dynamics) ---
    V_MAX: float = 3.0      # Max speed limit (m/s) - Soft limit via drag
    OMEGA_MAX: float = 3.0  # Max angular velocity (rad/s)
    
    # Acceleration Limits (The source of Inertia)
    ACCEL_MAX: float = 6.0       # Max linear acceleration (m/s^2)
    ALPHA_MAX: float = 12.0      # Max angular acceleration (rad/s^2)
    
    # Drag/Friction (Simulates air resistance/ground friction)
    # v_new = v_old * (1 - DRAG * DT)
    LIN_DRAG: float = 2.0   
    ANG_DRAG: float = 4.0
    
    RADIUS: float = 0.5     # Agent collision radius

    # --- Shooting & Spread ---
    SIGMA_STABLE: float = 0.05
    SIGMA_MAX: float = 0.3
    SIGMA_UNSTABLE: float = 0.15
    K_DECAY: float = 0.5
    DELTA_SIGMA: float = 0.05
    V_STABLE: float = 0.1
    W_STABLE: float = 0.1
    
    FIRE_RATE: float = 5.0
    DMG_MAX: float = 1.5
    DMG_WIDTH: float = 0.2
    HIT_RADIUS: float = 0.8

    # --- Ammo ---
    MAG_SIZE: int = 10
    MAX_MAGS: int = 3
    RELOAD_TIME: float = 2.0

    # --- Perception ---
    FOV: float = 120 * (math.pi / 180)
    PROXIMITY_RADIUS: float = 2.0  # Omni-directional awareness radius (Unified)


CFG = CQBConfig()


def batch_obs(obs_list: List[Dict[str, torch.Tensor]], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    """Batches a list of observation dictionaries into a single dictionary of tensors."""
    if not obs_list:
        return None
        
    batch = {'self': [], 'spatial': [], 'team': [], 'enemy': []}
    for o in obs_list:
        batch['self'].append(o['self'])
        batch['spatial'].append(o['spatial'])
        batch['team'].append(o['team'])
        batch['enemy'].append(o['enemy'])

    batched_self = torch.stack(batch['self']).to(device)
    batched_spatial = torch.stack(batch['spatial']).to(device)

    def pad_sequence(tensor_list: List[torch.Tensor]) -> torch.Tensor:
        """Pads a list of variable-length tensors to the max length in the batch."""
        max_len = max([t.shape[0] for t in tensor_list])
        if max_len == 0: max_len = 1
        
        feature_dim = tensor_list[0].shape[1] if (tensor_list and tensor_list[0].numel() > 0) else (10 if tensor_list is batch['team'] else 8)
        padded = torch.zeros((len(tensor_list), max_len, feature_dim), device=device)
        for i, t in enumerate(tensor_list):
            if t.numel() > 0:
                padded[i, :t.shape[0], :] = t
        return padded

    return {
        'self': batched_self,
        'spatial': batched_spatial,
        'team': pad_sequence(batch['team']),
        'enemy': pad_sequence(batch['enemy'])
    }