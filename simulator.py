import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any

from utils import CFG


class CQBSimulator:
    """GPU-accelerated 2D Multi-Agent CQB Simulator with Newtonian Dynamics."""

    def __init__(self, n_a: int = 2, n_b: int = 2):
        self.n_a = n_a
        self.n_b = n_b
        self.agents_total = n_a + n_b
        
        # Map Data
        self.map: torch.Tensor = torch.zeros((CFG.H, CFG.W), device=CFG.DEVICE, dtype=torch.float32)
        self.spawn_rect_a: Optional[Tuple[int, int, int, int]] = None
        self.spawn_rect_b: Optional[Tuple[int, int, int, int]] = None
        self.v_walls: Optional[np.ndarray] = None
        self.h_walls: Optional[np.ndarray] = None
        
        self._generate_map()
        
        # State Tensor: (N, 12)
        # 0:x, 1:y, 2:vx(global), 3:vy(global), 4:theta, 5:omega, 6:hp, 7:c, 8:n, 9:r_timer, 10:sigma, 11:team
        self.state: torch.Tensor = torch.zeros((self.agents_total, 12), device=CFG.DEVICE)
        self.last_shot_time: torch.Tensor = torch.zeros(self.agents_total, device=CFG.DEVICE) - 100.0
        
        self.time: float = 0.0
        self.steps: int = 0
        self.event_log: List[Dict[str, Any]] = []

    def reset(self) -> Dict[int, Dict[str, torch.Tensor]]:
        self.time = 0.0
        self.steps = 0
        self.event_log = []
        
        self._generate_map()
        
        padding = 1.5
        
        ax, ay, aw, ah = self.spawn_rect_a
        safe_wa = max(1.0, aw - 2 * padding)
        safe_ha = max(1.0, ah - 2 * padding)
        
        self.state[:self.n_a, 0] = ax + padding + torch.rand(self.n_a, device=CFG.DEVICE) * safe_wa
        self.state[:self.n_a, 1] = ay + padding + torch.rand(self.n_a, device=CFG.DEVICE) * safe_ha
        self.state[:self.n_a, 11] = 0  # Team A
        
        bx, by, bw, bh = self.spawn_rect_b
        safe_wb = max(1.0, bw - 2 * padding)
        safe_hb = max(1.0, bh - 2 * padding)
        
        self.state[self.n_a:, 0] = bx + padding + torch.rand(self.n_b, device=CFG.DEVICE) * safe_wb
        self.state[self.n_a:, 1] = by + padding + torch.rand(self.n_b, device=CFG.DEVICE) * safe_hb
        self.state[self.n_a:, 11] = 1  # Team B

        # Initialize facing center
        cx, cy = CFG.W / 2, CFG.H / 2
        dx = cx - self.state[:, 0]
        dy = cy - self.state[:, 1]
        self.state[:, 4] = torch.atan2(dy, dx)
        
        self.state[:, 2:4] = 0.0      # vx, vy (Initial velocity 0)
        self.state[:, 5] = 0.0        # omega
        self.state[:, 6] = 1.0        # hp
        self.state[:, 7] = CFG.MAG_SIZE
        self.state[:, 8] = CFG.MAX_MAGS
        self.state[:, 9] = 0.0        # reload timer
        self.state[:, 10] = CFG.SIGMA_STABLE

        return self._get_observations()

    def step(self, actions_dict: Dict[int, np.ndarray], reward_fn=None) -> Tuple:
        # Actions input: [accel_surge, accel_sway, alpha, fire, reload]
        # All values generally normalized to [-1, 1] (except logic), we scale them inside.
        actions = torch.zeros((self.agents_total, 5), device=CFG.DEVICE)
        
        if actions_dict:
            indices = list(actions_dict.keys())
            act_values = np.array(list(actions_dict.values()))
            actions[indices] = torch.tensor(act_values, device=CFG.DEVICE, dtype=torch.float32)

        self._update_status(actions)
        self._update_physics()
        hits = self._resolve_combat(actions)
        
        current_frame_data = {
            "time": self.time,
            "states": self.state.cpu().numpy().tolist(),
            "hits": hits
        }
        self.event_log.append(current_frame_data)
        
        self.time += CFG.DT
        self.steps += 1
        
        obs = self._get_observations()
        
        is_dead = self.state[:, 6] <= 0
        vanquished = {i: is_dead[i].item() for i in range(self.agents_total)}
        
        all_dead_a = torch.all(is_dead[:self.n_a]).item()
        all_dead_b = torch.all(is_dead[self.n_a:]).item()
        all_vanquished = all_dead_a or all_dead_b
        
        if reward_fn is not None:
            rewards = reward_fn(current_frame_data, vanquished)
            return obs, current_frame_data, rewards, vanquished, all_vanquished
        
        return obs, current_frame_data, vanquished, all_vanquished

    def _update_status(self, actions: torch.Tensor):
        # Newtonian Dynamics
        # actions: [accel_surge, accel_sway, alpha, fire, reload]
        
        # 1. Unpack Acceleration Commands (normalized [-1, 1] presumed, scale to Physics)
        # Note: In training loop we will scale actions before passing or here. 
        # Let's assume input is raw scaled physics values for simplicity if Controller is removed?
        # No, standard RL outputs [-1, 1]. Let's scale here.
        
        a_surge = actions[:, 0] * CFG.ACCEL_MAX
        a_sway = actions[:, 1] * CFG.ACCEL_MAX
        alpha = actions[:, 2] * CFG.ALPHA_MAX
        
        theta = self.state[:, 4]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # 2. Transform Local Accel to Global Accel
        ax_global = cos_t * a_surge - sin_t * a_sway
        ay_global = sin_t * a_surge + cos_t * a_sway
        
        # 3. Integrate Velocity (Euler): v = v + a * dt
        alive_mask = self.state[:, 6] > 0
        
        # Linear Velocity Update
        self.state[alive_mask, 2] += ax_global[alive_mask] * CFG.DT
        self.state[alive_mask, 3] += ay_global[alive_mask] * CFG.DT
        
        # Angular Velocity Update
        self.state[alive_mask, 5] += alpha[alive_mask] * CFG.DT
        
        # 4. Apply Drag/Friction (Damping)
        # v = v * (1 - decay)
        lin_decay = 1.0 - CFG.LIN_DRAG * CFG.DT
        ang_decay = 1.0 - CFG.ANG_DRAG * CFG.DT
        
        self.state[alive_mask, 2] *= lin_decay
        self.state[alive_mask, 3] *= lin_decay
        self.state[alive_mask, 5] *= ang_decay
        
        # 5. Hard Speed Cap (Safety)
        # Check norm
        v_norm = torch.sqrt(self.state[:, 2]**2 + self.state[:, 3]**2 + 1e-6)
        scale_mask = (v_norm > CFG.V_MAX) & alive_mask
        scale_factor = CFG.V_MAX / v_norm[scale_mask]
        self.state[scale_mask, 2] *= scale_factor
        self.state[scale_mask, 3] *= scale_factor
        
        self.state[alive_mask, 5] = torch.clamp(self.state[alive_mask, 5], -CFG.OMEGA_MAX, CFG.OMEGA_MAX)
        
        v_sq = self.state[:, 2]**2 + self.state[:, 3]**2
        omega_abs = torch.abs(self.state[:, 5])
        
        is_unstable = (v_sq > CFG.V_STABLE**2) | (omega_abs > CFG.W_STABLE)
        target_sigma = torch.where(
            is_unstable, 
            torch.tensor(CFG.SIGMA_UNSTABLE, device=CFG.DEVICE), 
            torch.tensor(CFG.SIGMA_STABLE, device=CFG.DEVICE)
        )
        
        current_sigma = self.state[:, 10]
        recover_mask = current_sigma > target_sigma
        current_sigma[recover_mask] -= CFG.K_DECAY * CFG.DT
        jump_mask = current_sigma < target_sigma
        current_sigma[jump_mask] = target_sigma[jump_mask]
        self.state[:, 10] = torch.clamp(current_sigma, CFG.SIGMA_STABLE, CFG.SIGMA_MAX)

        timer = self.state[:, 9]
        is_reloading = timer > 0
        timer[is_reloading] -= CFG.DT
        
        finished_reload = (timer <= 0) & is_reloading
        if finished_reload.any():
            self.state[finished_reload, 7] = CFG.MAG_SIZE
            self.state[finished_reload, 8] -= 1
            self.state[finished_reload, 9] = 0 
        
        trigger_reload = (actions[:, 4] > 0.5) & (self.state[:, 8] > 0) & (~is_reloading) & alive_mask
        self.state[trigger_reload, 9] = CFG.RELOAD_TIME

    def _update_physics(self):
        # Position update: p = p + v * dt
        alive_mask = self.state[:, 6] > 0
        curr_pos = self.state[:, 0:2].clone()
        velocity = self.state[:, 2:4]
        
        # X Update
        next_x = curr_pos[:, 0] + velocity[:, 0] * CFG.DT
        collision_x = self._check_collision_vectorized(next_x, curr_pos, axis=0)
        
        # Elastic collision response? Or just stop?
        # For simple top-down, stopping is easier. To prevent sticking, set vel to 0.
        self.state[alive_mask & ~collision_x, 0] = next_x[alive_mask & ~collision_x]
        self.state[alive_mask & collision_x, 2] = 0.0 # Stop X velocity on hit
        
        # Y Update
        curr_pos_updated_x = self.state[:, 0:2] 
        next_y = curr_pos[:, 1] + velocity[:, 1] * CFG.DT
        collision_y = self._check_collision_vectorized(next_y, curr_pos_updated_x, axis=1)
        self.state[alive_mask & ~collision_y, 1] = next_y[alive_mask & ~collision_y]
        self.state[alive_mask & collision_y, 3] = 0.0 # Stop Y velocity on hit
        
        # Angle Update
        theta_new = (self.state[:, 4] + self.state[:, 5] * CFG.DT) % (2 * math.pi)
        self.state[alive_mask, 4] = theta_new[alive_mask]

    def _check_collision_vectorized(self, next_pos: torch.Tensor, current_pos: torch.Tensor, axis: int) -> torch.Tensor:
        test_pos_x = next_pos if axis == 0 else current_pos[:, 0]
        test_pos_y = next_pos if axis == 1 else current_pos[:, 1]
        
        min_x = (test_pos_x - CFG.RADIUS).floor().long().clamp(0, CFG.W - 1)
        max_x = (test_pos_x + CFG.RADIUS).floor().long().clamp(0, CFG.W - 1)
        min_y = (test_pos_y - CFG.RADIUS).floor().long().clamp(0, CFG.H - 1)
        max_y = (test_pos_y + CFG.RADIUS).floor().long().clamp(0, CFG.H - 1)
        
        def is_wall(gx, gy):
            return self.map[gy, gx] > 0.5

        c1 = is_wall(min_x, min_y)
        c2 = is_wall(max_x, min_y)
        c3 = is_wall(min_x, max_y)
        c4 = is_wall(max_x, max_y)
        return c1 | c2 | c3 | c4

    def _vectorized_raycast(self, starts: torch.Tensor, ends: torch.Tensor, num_samples: int) -> torch.Tensor:
        N = starts.shape[0]
        if N == 0: return torch.zeros(0, dtype=torch.bool, device=CFG.DEVICE)
        t = torch.linspace(0, 1, num_samples, device=CFG.DEVICE).view(1, -1, 1)
        points = starts.unsqueeze(1) + (ends - starts).unsqueeze(1) * t
        grid_x = points[:, :, 0].long().clamp(0, CFG.W - 1)
        grid_y = points[:, :, 1].long().clamp(0, CFG.H - 1)
        map_vals = self.map[grid_y, grid_x]
        return (map_vals > 0.5).any(dim=1)

    def _resolve_combat(self, actions: torch.Tensor) -> List[Dict]:
        hits_log = []
        fire_cmd = actions[:, 3] > 0.0 # Continuous > 0 threshold
        can_fire = (self.state[:, 6] > 0) & (self.state[:, 7] > 0) & (self.state[:, 9] <= 0) & ((self.time - self.last_shot_time) >= (1.0/CFG.FIRE_RATE))
        
        shooter_indices = torch.nonzero(fire_cmd & can_fire).squeeze(1)
        if shooter_indices.numel() == 0: return hits_log

        self.state[shooter_indices, 7] -= 1
        self.last_shot_time[shooter_indices] = self.time
        
        shooters_sigma = self.state[shooter_indices, 10]
        noise = torch.randn(len(shooter_indices), device=CFG.DEVICE) * shooters_sigma
        shoot_angle = self.state[shooter_indices, 4] + noise
        self.state[shooter_indices, 10] = torch.min(shooters_sigma + CFG.DELTA_SIGMA, torch.tensor(CFG.SIGMA_MAX, device=CFG.DEVICE))
        
        p_s = self.state[shooter_indices, 0:2]
        dir_vec = torch.stack([torch.cos(shoot_angle), torch.sin(shoot_angle)], dim=1)
        p_t = self.state[:, 0:2]
        hp_t = self.state[:, 6]
        
        diff = p_t.unsqueeze(0) - p_s.unsqueeze(1) 
        dist_st = torch.norm(diff, dim=2)
        proj_t = (diff * dir_vec.unsqueeze(1)).sum(dim=2)
        
        valid_mask = (dist_st > 0.01) & (proj_t > 0) & (hp_t.unsqueeze(0) > 0)
        lat_offset = diff - proj_t.unsqueeze(2) * dir_vec.unsqueeze(1)
        lat_dist = torch.norm(lat_offset, dim=2)
        hit_candidate_mask = valid_mask & (lat_dist < CFG.HIT_RADIUS)
        
        if hit_candidate_mask.any():
            s_idx_list, t_idx_list = torch.nonzero(hit_candidate_mask, as_tuple=True)
            if len(s_idx_list) > 0:
                c_ps = p_s[s_idx_list]
                c_pt = p_t[t_idx_list]
                c_dist = dist_st[s_idx_list, t_idx_list]
                
                max_dist = c_dist.max().item()
                num_samples = int(max_dist * 2.0) + 2
                is_blocked = self._vectorized_raycast(c_ps, c_pt, num_samples=num_samples)
                
                final_hit_mask = ~is_blocked
                for i in range(len(final_hit_mask)):
                    if final_hit_mask[i]:
                        s_real = shooter_indices[s_idx_list[i]].item()
                        t_real = t_idx_list[i].item()
                        dmg = CFG.DMG_MAX * math.exp(-(lat_dist[s_idx_list[i], t_idx_list[i]].item()**2) / (2 * CFG.DMG_WIDTH**2))
                        self.state[t_real, 6] = max(0.0, self.state[t_real, 6].item() - dmg)
                        hits_log.append({'shooter': s_real, 'target': t_real, 'damage': dmg, 'loc': self.state[t_real, 0:2].cpu().numpy().tolist()})
        return hits_log

    def _get_observations(self) -> Dict[int, Optional[Dict[str, torch.Tensor]]]:
        obs_dict = {}
        half_L = CFG.L // 2
        alive_mask = self.state[:, 6] > 0
        
        padded_map = torch.nn.functional.pad(self.map, (CFG.W, CFG.W, CFG.H, CFG.H), value=1)
        pos = self.state[:, 0:2]
        rel_pos_matrix = pos.unsqueeze(0) - pos.unsqueeze(1) 
        
        teams = self.state[:, 11].long()
        team_matrix = teams.unsqueeze(0) == teams.unsqueeze(1)
        
        for i in range(self.agents_total):
            if not alive_mask[i]:
                obs_dict[i] = None
                continue
                
            o_self = torch.cat([self.state[i, 4:5], self.state[i, 2:4], self.state[i, 5:6], self.state[i, 6:11]])
            o_self[7] = 1.0 if o_self[7] > 0 else 0.0
            
            cx, cy = int(self.state[i, 0]), int(self.state[i, 1])
            start_x = cx + CFG.W - half_L
            start_y = cy + CFG.H - half_L
            local_grid = padded_map[start_y : start_y + CFG.L, start_x : start_x + CFG.L]
            o_spatial = local_grid.flatten()
            
            idxs = torch.arange(self.agents_total, device=CFG.DEVICE)
            is_teammate = team_matrix[i] & (idxs != i)
            is_enemy = ~team_matrix[i]
            
            if is_teammate.any():
                t_idxs = idxs[is_teammate]
                feat = torch.cat([
                    rel_pos_matrix[i, t_idxs], self.state[t_idxs, 4:5], self.state[t_idxs, 2:4],
                    self.state[t_idxs, 5:6], self.state[t_idxs, 6:10]
                ], dim=1)
                feat[:, 9] = (feat[:, 9] > 0).float()
                o_team = feat
            else:
                o_team = torch.empty(0, 10, device=CFG.DEVICE)
            
            if is_enemy.any():
                e_idxs = idxs[is_enemy]
                viewer_mask = team_matrix[i] & alive_mask
                viewers_pos = pos[viewer_mask]
                viewers_theta = self.state[viewer_mask, 4]
                targets_pos = pos[e_idxs]
                
                ve_vec = targets_pos.unsqueeze(0) - viewers_pos.unsqueeze(1)
                ve_dist = torch.norm(ve_vec, dim=2)
                
                angle_to_target = torch.atan2(ve_vec[:, :, 1], ve_vec[:, :, 0])
                angle_diff = torch.abs(angle_to_target - viewers_theta.unsqueeze(1))
                angle_diff = torch.min(angle_diff, 2*math.pi - angle_diff)
                
                # Visibility Logic: FOV OR Proximity
                in_fov = angle_diff < (CFG.FOV / 2)
                is_near = ve_dist < CFG.PROXIMITY_RADIUS
                
                check_mask = (in_fov | is_near).flatten()
                
                visible_mask = torch.zeros_like(check_mask, dtype=torch.bool)
                
                if check_mask.any():
                    v_mesh, e_mesh = torch.meshgrid(
                        torch.arange(len(viewers_pos), device=CFG.DEVICE),
                        torch.arange(len(targets_pos), device=CFG.DEVICE),
                        indexing='ij'
                    )
                    flat_starts = viewers_pos[v_mesh.flatten()[check_mask]]
                    flat_ends = targets_pos[e_mesh.flatten()[check_mask]]
                    
                    # FIX: Dynamic Sampling to prevent wall skipping
                    # Distance of each pair
                    dists = torch.norm(flat_ends - flat_starts, dim=1)
                    max_d = dists.max().item() if dists.numel() > 0 else 1.0
                    
                    # At least 2 samples per meter
                    obs_num_samples = int(max_d * 2.0) + 2
                    
                    is_blocked = self._vectorized_raycast(flat_starts, flat_ends, num_samples=obs_num_samples)
                    
                    visible_flat_indices = torch.nonzero(check_mask).squeeze(1)
                    valid_indices = visible_flat_indices[~is_blocked]
                    visible_mask[valid_indices] = True
                
                visible_mask = visible_mask.view(len(viewers_pos), len(targets_pos))
                is_visible = visible_mask.any(dim=0)
                visible_e_idxs = e_idxs[is_visible]
                
                if len(visible_e_idxs) > 0:
                    feat = torch.cat([
                        rel_pos_matrix[i, visible_e_idxs], self.state[visible_e_idxs, 4:5],
                        self.state[visible_e_idxs, 2:4], self.state[visible_e_idxs, 5:6],
                        self.state[visible_e_idxs, 6:7], self.state[visible_e_idxs, 9:10]
                    ], dim=1)
                    feat[:, -1] = (feat[:, -1] > 0).float()
                    o_enemy = feat
                else:
                    o_enemy = torch.empty(0, 8, device=CFG.DEVICE)
            else:
                o_enemy = torch.empty(0, 8, device=CFG.DEVICE)

            obs_dict[i] = {'self': o_self, 'spatial': o_spatial, 'team': o_team, 'enemy': o_enemy}
            
        return obs_dict

    def _generate_map(self):
        x_coords = self._generate_grid_lines(CFG.W)
        y_coords = self._generate_grid_lines(CFG.H)
        rows, cols = len(y_coords) - 1, len(x_coords) - 1
        
        self.v_walls = np.ones((rows, cols + 1), dtype=np.int8)
        self.h_walls = np.ones((rows + 1, cols), dtype=np.int8)
        
        # 1. Random Merges
        cell_group_id = np.arange(rows * cols).reshape(rows, cols)
        for _ in range(int(rows * cols * 0.2)):
            self._merge_random_cluster(rows, cols, cell_group_id)
            
        # 2. Spawn Points
        spawn_r_a = np.random.randint(1, rows - 1)
        spawn_c_a = np.random.randint(1, cols - 1)
        spawn_r_b = rows - 1 - spawn_r_a
        spawn_c_b = cols - 1 - spawn_c_a
        
        self.spawn_rect_a = self._get_cell_rect(spawn_r_a, spawn_c_a, x_coords, y_coords)
        self.spawn_rect_b = self._get_cell_rect(spawn_r_b, spawn_c_b, x_coords, y_coords)
        
        # 3. Ensure Connectivity (Cell-based BFS-Prim)
        self._ensure_connectivity_from_spawn(spawn_r_a, spawn_c_a, rows, cols)
        
        # 4. Render
        self.grid_np = np.zeros((CFG.H, CFG.W), dtype=np.uint8)
        self._render_grid(x_coords, y_coords)
        self.grid_np[0, :] = 1; self.grid_np[-1, :] = 1
        self.grid_np[:, 0] = 1; self.grid_np[:, -1] = 1
        
        self.map = torch.tensor(self.grid_np, device=CFG.DEVICE, dtype=torch.float32)

    def _generate_grid_lines(self, length):
        coords = [0]
        while coords[-1] < length:
            step = np.random.randint(10, 21)
            next_pos = coords[-1] + step
            if next_pos >= length - 10:
                coords.append(length)
                break
            coords.append(next_pos)
        return coords

    def _get_cell_rect(self, r, c, x_coords, y_coords):
        x = x_coords[c]
        y = y_coords[r]
        return (x, y, x_coords[c+1] - x, y_coords[r+1] - y)

    def _merge_random_cluster(self, rows, cols, cell_group_id):
        start_r, start_c = np.random.randint(0, rows), np.random.randint(0, cols)
        target_group = cell_group_id[start_r, start_c]
        cluster = [(start_r, start_c)]
        candidates = []
        self._get_cluster_neighbors(start_r, start_c, rows, cols, candidates)
        
        while len(cluster) < np.random.randint(2, 6) and candidates:
            idx = np.random.randint(len(candidates))
            nr, nc, wtype = candidates.pop(idx)
            if cell_group_id[nr, nc] != target_group:
                if wtype == 'h':
                    self.h_walls[max(nr, cluster[-1][0])][nc] = 0
                else:
                    self.v_walls[nr][max(nc, cluster[-1][1])] = 0
                old_id = cell_group_id[nr, nc]
                cell_group_id[cell_group_id == old_id] = target_group
                cluster.append((nr, nc))
                self._get_cluster_neighbors(nr, nc, rows, cols, candidates)

    def _get_cluster_neighbors(self, r, c, rows, cols, lst):
        if r > 0: lst.append((r-1, c, 'h'))
        if r < rows-1: lst.append((r+1, c, 'h'))
        if c > 0: lst.append((r, c-1, 'v'))
        if c < cols-1: lst.append((r, c+1, 'v'))

    def _ensure_connectivity_from_spawn(self, start_r, start_c, rows, cols):
        """
        Guarantees graph connectivity using Cell-based Prim's algorithm.
        Ensures all rooms are reachable from spawn.
        """
        visited = np.zeros((rows, cols), dtype=bool)
        visited[start_r, start_c] = True
        
        # Queue for BFS filling of open areas
        bfs_queue = [(start_r, start_c)]
        
        # Candidate walls to break (connecting visited to unvisited)
        # (r, c, neighbor_r, neighbor_c, type, w_r, w_c)
        candidate_walls = []
        
        # Initial scan
        self._scan_neighbors(start_r, start_c, rows, cols, visited, bfs_queue, candidate_walls)
        
        visited_count = 1
        total_cells = rows * cols
        
        while visited_count < total_cells:
            # 1. Flood fill reachable area (BFS)
            while bfs_queue:
                r, c = bfs_queue.pop(0)
                # For each popped cell, scan its neighbors to add more to queue or candidates
                self._scan_neighbors(r, c, rows, cols, visited, bfs_queue, candidate_walls)
                # Count is updated when adding to queue
            
            # Recalculate count to be sure or track it dynamically
            visited_count = np.sum(visited)
            if visited_count >= total_cells:
                break
                
            # 2. Break a wall (Prim's)
            found_break = False
            while candidate_walls:
                idx = np.random.randint(len(candidate_walls))
                candidate_walls[idx], candidate_walls[-1] = candidate_walls[-1], candidate_walls[idx]
                r, c, nr, nc, wtype, wr, wc = candidate_walls.pop()
                
                if not visited[nr, nc]:
                    # Break wall
                    if wtype == 'h': self.h_walls[wr][wc] = 2
                    else: self.v_walls[wr][wc] = 2
                    
                    # Mark visited and add to BFS
                    visited[nr, nc] = True
                    bfs_queue.append((nr, nc))
                    found_break = True
                    break
            
            if not found_break and visited_count < total_cells:
                # This technically shouldn't happen on a grid unless initialized wrong
                break

    def _scan_neighbors(self, r, c, rows, cols, visited, bfs_queue, candidates):
        # Up
        if r > 0:
            if not visited[r-1, c]:
                if self.h_walls[r][c] == 1: candidates.append((r, c, r-1, c, 'h', r, c))
                else: visited[r-1, c] = True; bfs_queue.append((r-1, c))
        # Down
        if r < rows - 1:
            if not visited[r+1, c]:
                if self.h_walls[r+1][c] == 1: candidates.append((r, c, r+1, c, 'h', r+1, c))
                else: visited[r+1, c] = True; bfs_queue.append((r+1, c))
        # Left
        if c > 0:
            if not visited[r, c-1]:
                if self.v_walls[r][c] == 1: candidates.append((r, c, r, c-1, 'v', r, c))
                else: visited[r, c-1] = True; bfs_queue.append((r, c-1))
        # Right
        if c < cols - 1:
            if not visited[r, c+1]:
                if self.v_walls[r][c+1] == 1: candidates.append((r, c, r, c+1, 'v', r, c+1))
                else: visited[r, c+1] = True; bfs_queue.append((r, c+1))

    def _render_grid(self, x_coords, y_coords):
        rows, cols = len(y_coords) - 1, len(x_coords) - 1
        for r in range(rows + 1):
            y = min(y_coords[r], CFG.H - 1)
            for c in range(cols):
                if self.h_walls[r][c] == 1:
                    self.grid_np[y, x_coords[c]:x_coords[c+1]] = 1
                elif self.h_walls[r][c] == 2:
                    self.grid_np[y, x_coords[c]:x_coords[c+1]] = 1
                    ds = np.random.randint(3, 6)
                    if x_coords[c+1]-x_coords[c] > ds:
                         st = x_coords[c] + np.random.randint(1, x_coords[c+1]-x_coords[c]-ds)
                         self.grid_np[y, st:st+ds] = 0
        
        for r in range(rows):
            yst, yed = y_coords[r], y_coords[r+1]
            for c in range(cols + 1):
                x = min(x_coords[c], CFG.W - 1)
                if self.v_walls[r][c] == 1:
                    self.grid_np[yst:yed, x] = 1
                elif self.v_walls[r][c] == 2:
                    self.grid_np[yst:yed, x] = 1
                    ds = np.random.randint(3, 6)
                    if yed - yst > ds:
                        st = yst + np.random.randint(1, yed - yst - ds)
                        self.grid_np[st:st+ds, x] = 0