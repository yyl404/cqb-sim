import torch
import numpy as np
import cv2
import os
import time
import json
import math
from typing import List, Dict, Tuple, Any
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================

class CQBConfig:
    # --- 环境尺寸 ---
    H, W = 100, 100  # 地图尺寸
    L = 20           # 局部观测裁剪尺寸 (20x20)
    DT = 0.05        # 时间步长 (s)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 物理限制 ---
    V_MAX = 3.0      # 最大速度 (m/s)
    ACC_MAX = 5.0    # 加速度 (m/s^2)
    OMEGA_MAX = 3.0  # 最大角速度 (rad/s)
    FRICTION = 2.0   # 摩擦系数
    RADIUS = 0.5     # 智能体碰撞半径

    # --- 射击与散布 ---
    SIGMA_0 = 0.05       # 静息散布 (rad)
    SIGMA_MAX = 0.3      # 最大散布 (rad)
    SIGMA_1 = 0.15       # 运动惩罚散布 (rad)
    K_DECAY = 0.5        # 散布恢复速率 (rad/s)
    DELTA_SIGMA = 0.05   # 单发增加散布
    V_STABLE = 0.1       # 静息速度阈值
    W_STABLE = 0.1       # 静息转向阈值
    
    FIRE_RATE = 5.0      # 射速 (rounds/s)
    DMG_MAX = 0.4        # 最大单发伤害
    DMG_WIDTH = 0.2      # 伤害衰减宽度
    HIT_RADIUS = 0.8     # 命中判定半径

    # --- 弹药 ---
    MAG_SIZE = 10        # 弹匣容量
    MAX_MAGS = 3         # 备弹数量
    RELOAD_TIME = 2.0    # 换弹时间 (s)

    # --- 感知 ---
    FOV = 120 * (math.pi / 180)  # 视锥角度 (弧度)
    
    def __init__(self):
        pass

CFG = CQBConfig()

# ==========================================
# [新增位置] PPO模型定义 (移到这里以解决循环引用)
# ==========================================
class PPOActorCritic(nn.Module):
    def __init__(self, spatial_dim=CFG.L*CFG.L, self_dim=9):
        super().__init__()
        # --- Shared Features ---
        self.fc_spatial = nn.Linear(spatial_dim, 128)
        self.fc_self = nn.Linear(self_dim, 64)
        self.fc_common = nn.Linear(128 + 64, 256)
        
        # --- Actor (输出动作均值) ---
        # 动作维度: 5 (surge, sway, omega, fire, reload)
        self.actor_mean = nn.Linear(256, 5)
        
        # --- Actor (输出动作标准差 - 可学习的参数) ---
        self.actor_log_std = nn.Parameter(torch.zeros(1, 5))
        
        # --- Critic (输出状态价值) ---
        self.critic = nn.Linear(256, 1)

    def forward(self, o_self, o_spatial):
        # 特征提取
        x1 = torch.relu(self.fc_spatial(o_spatial))
        x2 = torch.relu(self.fc_self(o_self))
        x = torch.cat([x1, x2], dim=-1)
        x = torch.relu(self.fc_common(x))
        
        # Actor: Mean & Std
        action_mean = torch.tanh(self.actor_mean(x)) # 限制在 [-1, 1]
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Critic: Value
        value = self.critic(x)
        
        return action_mean, action_std, value

# ==========================================
# 2. 核心仿真环境 (Physics Engine)
# ==========================================

class CQBSimulator:
    def __init__(self, n_a=2, n_b=2):
        self.n_a = n_a
        self.n_b = n_b
        self.agents_total = n_a + n_b
        
        # 初始化地图 (0: 空地, 1: 墙)
        self.map = torch.zeros((CFG.H, CFG.W), device=CFG.DEVICE)
        self._generate_map()
        
        # 状态张量化存储
        # 0:x, 1:y, 2:vx, 3:vy, 4:theta, 5:omega, 6:hp, 7:c, 8:n, 9:r_timer, 10:sigma, 11:team
        self.state = torch.zeros((self.agents_total, 12), device=CFG.DEVICE)
        self.last_shot_time = torch.zeros(self.agents_total, device=CFG.DEVICE) - 100.0
        
        self.time = 0.0
        self.steps = 0
        self.event_log = [] # 用于回放

    def _generate_map(self):
        # 简单生成一些墙壁
        cx, cy = CFG.W // 2, CFG.H // 2
        self.map[cy-5:cy+5, cx-20:cx+20] = 1
        self.map[cy-20:cy+20, cx-5:cx+5] = 1
        self.map[0, :] = 1
        self.map[-1, :] = 1
        self.map[:, 0] = 1
        self.map[:, -1] = 1

    def reset(self):
        self.time = 0.0
        self.steps = 0
        self.event_log = []
        
        for i in range(self.agents_total):
            is_a = i < self.n_a
            x = np.random.uniform(5, 30) if is_a else np.random.uniform(CFG.W-30, CFG.W-5)
            y = np.random.uniform(5, CFG.H-5)
            
            self.state[i, 0] = x
            self.state[i, 1] = y
            self.state[i, 2] = 0 
            self.state[i, 3] = 0
            self.state[i, 4] = 0 if is_a else np.pi 
            self.state[i, 5] = 0 
            self.state[i, 6] = 1.0 
            self.state[i, 7] = CFG.MAG_SIZE 
            self.state[i, 8] = CFG.MAX_MAGS 
            self.state[i, 9] = 0 
            self.state[i, 10] = CFG.SIGMA_0 
            self.state[i, 11] = 0 if is_a else 1

        return self._get_observations()

    def step(self, actions_dict):
        # 将动作转为Tensor
        actions = torch.zeros((self.agents_total, 5), device=CFG.DEVICE)
        for i in range(self.agents_total):
            if i in actions_dict:
                actions[i] = torch.tensor(actions_dict[i], device=CFG.DEVICE)

        self._update_physics(actions)
        self._update_status(actions)
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
        rewards = {} 
        dones = {}
        for i in range(self.agents_total):
            r = 0
            if self.state[i, 6] > 0: r += 0.01 
            rewards[i] = r
            dones[i] = self.state[i, 6] <= 0 

        for hit in hits:
            shooter_idx = hit['shooter']
            dmg = hit['damage']
            rewards[shooter_idx] += dmg * 10

        done_all = all([dones[i] for i in range(self.n_a)]) or all([dones[i] for i in range(self.n_a, self.agents_total)])
        
        return obs, rewards, dones, done_all

    def _update_physics(self, actions):
        theta = self.state[:, 4]
        
        a_surge = actions[:, 0]
        a_sway = actions[:, 1]
        a_omega = actions[:, 2]
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        ax_global = CFG.ACC_MAX * (cos_t * a_surge - sin_t * a_sway)
        ay_global = CFG.ACC_MAX * (sin_t * a_surge + cos_t * a_sway)
        
        vx = self.state[:, 2]
        vy = self.state[:, 3]
        
        vx_new = vx + ax_global * CFG.DT - CFG.FRICTION * vx * CFG.DT
        vy_new = vy + ay_global * CFG.DT - CFG.FRICTION * vy * CFG.DT
        
        v_norm = torch.sqrt(vx_new**2 + vy_new**2 + 1e-6)
        scale = torch.clamp(CFG.V_MAX / v_norm, max=1.0)
        vx_new *= scale
        vy_new *= scale
        
        omega = self.state[:, 5]
        omega_new = omega + a_omega * CFG.DT 
        omega_new = torch.clamp(omega_new, -CFG.OMEGA_MAX, CFG.OMEGA_MAX)
        
        x_new = self.state[:, 0] + vx_new * CFG.DT
        y_new = self.state[:, 1] + vy_new * CFG.DT
        theta_new = (theta + omega_new * CFG.DT) % (2 * np.pi)
        
        gx = x_new.long().clamp(0, CFG.W-1)
        gy = y_new.long().clamp(0, CFG.H-1)
        
        collided = self.map[gy, gx] > 0.5
        
        alive = self.state[:, 6] > 0
        update_mask = alive & (~collided)
        
        self.state[update_mask, 0] = x_new[update_mask]
        self.state[update_mask, 1] = y_new[update_mask]
        self.state[alive, 2] = vx_new[alive]
        self.state[alive, 3] = vy_new[alive]
        self.state[alive, 4] = theta_new[alive]
        self.state[alive, 5] = omega_new[alive]
        
        self.state[collided & alive, 2] = 0
        self.state[collided & alive, 3] = 0

    def _update_status(self, actions):
        v_sq = self.state[:, 2]**2 + self.state[:, 3]**2
        omega_abs = torch.abs(self.state[:, 5])
        
        is_moving = (v_sq > CFG.V_STABLE**2) | (omega_abs > CFG.W_STABLE)
        target_sigma = torch.where(is_moving, torch.tensor(CFG.SIGMA_1, device=CFG.DEVICE), torch.tensor(CFG.SIGMA_0, device=CFG.DEVICE))
        
        current_sigma = self.state[:, 10]
        
        recover_mask = current_sigma > target_sigma
        current_sigma[recover_mask] -= CFG.K_DECAY * CFG.DT
        
        jump_mask = current_sigma < target_sigma
        current_sigma[jump_mask] = target_sigma[jump_mask]
        
        self.state[:, 10] = torch.clamp(current_sigma, CFG.SIGMA_0, CFG.SIGMA_MAX)

        timer = self.state[:, 9]
        is_reloading = timer > 0
        
        timer[is_reloading] -= CFG.DT
        finished_reload = (timer <= 0) & is_reloading
        
        idx_finished = torch.nonzero(finished_reload).squeeze()
        if idx_finished.numel() > 0:
            if idx_finished.dim() == 0: idx_finished = idx_finished.unsqueeze(0)
            for idx in idx_finished:
                self.state[idx, 7] = CFG.MAG_SIZE
                self.state[idx, 8] -= 1
                self.state[idx, 9] = 0 
        
        trigger_reload = (actions[:, 4] > 0.5) & (self.state[:, 8] > 0) & (~is_reloading) & (self.state[:, 6] > 0)
        self.state[trigger_reload, 9] = CFG.RELOAD_TIME
        
    def _resolve_combat(self, actions):
        hits_log = []
        fire_cmd = actions[:, 3] > 0.5
        can_fire = (self.state[:, 6] > 0) & \
                   (self.state[:, 7] > 0) & \
                   (self.state[:, 9] <= 0) & \
                   ((self.time - self.last_shot_time) >= (1.0/CFG.FIRE_RATE))
        
        shooters = torch.nonzero(fire_cmd & can_fire).squeeze()
        if shooters.numel() == 0:
            return hits_log
        if shooters.dim() == 0: shooters = shooters.unsqueeze(0)

        for s_idx in shooters:
            self.state[s_idx, 7] -= 1
            self.last_shot_time[s_idx] = self.time
            self.state[s_idx, 10] = min(self.state[s_idx, 10] + CFG.DELTA_SIGMA, CFG.SIGMA_MAX)
            
            noise = np.random.normal(0, self.state[s_idx, 10].item())
            shoot_angle = self.state[s_idx, 4].item() + noise
            
            p0 = self.state[s_idx, 0:2]
            direction = torch.tensor([math.cos(shoot_angle), math.sin(shoot_angle)], device=CFG.DEVICE)
            
            targets = torch.arange(self.agents_total, device=CFG.DEVICE)
            targets = targets[targets != s_idx]
            
            min_dist = 9999
            hit_target = -1
            hit_perp_dist = 0
            
            for t_idx in targets:
                if self.state[t_idx, 6] <= 0: continue
                
                pt = self.state[t_idx, 0:2]
                v_st = pt - p0
                proj_t = torch.dot(v_st, direction)
                
                if proj_t < 0: continue
                
                perp_vec = v_st - proj_t * direction
                d_perp = torch.norm(perp_vec)
                
                if d_perp < CFG.HIT_RADIUS:
                    if proj_t < min_dist:
                        min_dist = proj_t
                        hit_target = t_idx
                        hit_perp_dist = d_perp
            
            if hit_target != -1:
                damage = CFG.DMG_MAX * math.exp(-(hit_perp_dist.item()**2) / (2 * CFG.DMG_WIDTH**2))
                self.state[hit_target, 6] = max(0, self.state[hit_target, 6] - damage)
                hits_log.append({
                    'shooter': s_idx.item(),
                    'target': hit_target.item(),
                    'damage': damage,
                    'loc': self.state[hit_target, 0:2].cpu().numpy().tolist()
                })
                
        return hits_log

    def _get_observations(self):
        obs_dict = {}
        for i in range(self.agents_total):
            if self.state[i, 6] <= 0:
                obs_dict[i] = None
                continue
                
            o_self = torch.cat([
                self.state[i, 4:5], 
                self.state[i, 2:4], 
                self.state[i, 5:6], 
                self.state[i, 6:11] 
            ])
            o_self[7] = 1.0 if o_self[7] > 0 else 0.0 
            
            cx, cy = int(self.state[i, 0]), int(self.state[i, 1])
            half_L = CFG.L // 2
            padded_map = torch.nn.functional.pad(self.map, (half_L, half_L, half_L, half_L), value=1)
            x_start = cx
            y_start = cy
            local_grid = padded_map[y_start:y_start+CFG.L, x_start:x_start+CFG.L]
            o_spatial = local_grid.flatten()
            
            team_obs = []
            enemy_obs = []
            
            my_team = self.state[i, 11]
            my_pos = self.state[i, 0:2]
            my_theta = self.state[i, 4]
            
            for j in range(self.agents_total):
                if i == j: continue
                
                rel_pos = self.state[j, 0:2] - my_pos 
                target_team = self.state[j, 11]
                
                if target_team == my_team:
                    feat = torch.cat([rel_pos, self.state[j, 4:5], self.state[j, 2:4], self.state[j, 6:10]])
                    feat[8] = 1.0 if feat[8] > 0 else 0.0
                    team_obs.append(feat)
                else:
                    vec_to_enemy = rel_pos
                    dist = torch.norm(vec_to_enemy)
                    angle_to_enemy = torch.atan2(vec_to_enemy[1], vec_to_enemy[0])
                    angle_diff = torch.abs(angle_to_enemy - my_theta)
                    angle_diff = torch.min(angle_diff, 2*np.pi - angle_diff)
                    
                    is_visible = False
                    if dist < 0.1: is_visible = True
                    elif angle_diff < (CFG.FOV / 2):
                        is_visible = not self._check_line_of_sight(
                            int(my_pos[0]), int(my_pos[1]), 
                            int(self.state[j, 0]), int(self.state[j, 1])
                        )
                    
                    if is_visible:
                        feat = torch.cat([
                            rel_pos, 
                            self.state[j, 4:5], 
                            self.state[j, 2:4], 
                            self.state[j, 5:6],
                            self.state[j, 6:7],
                            self.state[j, 9:10]
                        ])
                        feat[-1] = 1.0 if feat[-1] > 0 else 0.0
                        enemy_obs.append(feat)
                    else:
                        enemy_obs.append(torch.zeros(8, device=CFG.DEVICE)) 

            o_team = torch.stack(team_obs) if team_obs else torch.empty(0, device=CFG.DEVICE)
            o_enemy = torch.stack(enemy_obs) if enemy_obs else torch.empty(0, device=CFG.DEVICE)
            
            obs_dict[i] = {
                'self': o_self,
                'spatial': o_spatial,
                'team': o_team,
                'enemy': o_enemy
            }
            
        return obs_dict

    def _check_line_of_sight(self, x0, y0, x1, y1):
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0: return False
        
        xs = torch.linspace(x0, x1, steps=steps+1, device=CFG.DEVICE)
        ys = torch.linspace(y0, y1, steps=steps+1, device=CFG.DEVICE)
        
        for k in range(1, steps):
            ix, iy = int(xs[k]), int(ys[k])
            if 0 <= ix < CFG.W and 0 <= iy < CFG.H:
                if self.map[iy, ix] > 0.5:
                    return True 
        return False
    
    def save_replay(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.event_log, f)
        print(f"Replay saved to {filepath}")

# ==========================================
# 3. 渲染器 (Visualizer)
# ==========================================

class CQBRenderer:
    def __init__(self, map_data):
        self.map = map_data.cpu().numpy()
        self.H, self.W = self.map.shape
        self.scale = 8 # 像素放大倍数
        
        # 1. 先在原始尺寸 (H, W) 上绘制颜色
        small_bg = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        small_bg[self.map == 0] = (240, 240, 240) 
        small_bg[self.map == 1] = (100, 100, 100) 

        # 2. 使用最近邻插值放大到目标尺寸
        self.bg = cv2.resize(
            small_bg, 
            (self.W * self.scale, self.H * self.scale), 
            interpolation=cv2.INTER_NEAREST
        )

    def render_frame(self, state_list, hits, save_path=None):
        canvas = self.bg.copy()
        
        for i, s in enumerate(state_list):
            x, y = s[0], s[1]
            theta = s[4]
            hp = s[6]
            team = int(s[11])
            is_reloading = s[9] > 0
            
            if hp <= 0: continue 
            
            cx, cy = int(x * self.scale), int(y * self.scale)
            color = (0, 0, 200) if team == 0 else (200, 0, 0)
            
            # 身体
            cv2.circle(canvas, (cx, cy), int(CFG.RADIUS * self.scale), color, -1)
            # 朝向
            ex = int(cx + math.cos(theta) * 10)
            ey = int(cy + math.sin(theta) * 10)
            cv2.line(canvas, (cx, cy), (ex, ey), (0, 0, 0), 2)
            # 血条
            bar_len = 20
            cv2.rectangle(canvas, (cx-10, cy-15), (cx-10+int(bar_len*hp), cy-12), (0, 255, 0), -1)
            
            if is_reloading:
                cv2.putText(canvas, "R", (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return canvas

    def render_replay(self, log_path, output_mp4):
        with open(log_path, 'r') as f:
            logs = json.load(f)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_mp4, fourcc, 20.0, (self.W*self.scale, self.H*self.scale))
        
        print(f"Rendering {len(logs)} frames to {output_mp4}...")
        for frame_data in tqdm(logs):
            states = frame_data['states']
            hits = frame_data['hits']
            img = self.render_frame(states, hits)
            out.write(img)
            
        out.release()
        print("Done.")

# ==========================================
# 4. 策略接口
# ==========================================

class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, obs_dict) -> np.array:
        pass

class RandomPolicy(BasePolicy):
    def get_action(self, obs_dict):
        act = np.random.uniform(-1, 1, 5)
        act[3] = 1.0 if np.random.rand() > 0.9 else 0.0 
        act[4] = 1.0 if np.random.rand() > 0.99 else 0.0
        return act

# ==========================================
# 5. 主程序入口 (Main)
# ==========================================

def run_simulation():
    print(f"Running on device: {CFG.DEVICE}")
    
    env = CQBSimulator(n_a=2, n_b=2)
    obs = env.reset()
    
    # 使用本文件中定义的 PPOActorCritic
    trained_net = PPOActorCritic().to(CFG.DEVICE)

    try:
        trained_net.load_state_dict(torch.load("ppo_cqb.pth"))
        print("Loaded trained model!")
    except FileNotFoundError:
        print("No trained model found, using random weights.")

    class TrainedPolicy(BasePolicy):
        def __init__(self, model):
            self.model = model
        
        def get_action(self, obs):
            self.model.eval()
            with torch.no_grad():
                o_s = obs['self'].unsqueeze(0)
                o_m = obs['spatial'].unsqueeze(0)
                mean, _, _ = self.model(o_s, o_m)
                return mean.squeeze(0).cpu().numpy()

    policy_A = TrainedPolicy(trained_net)
    policy_B = TrainedPolicy(trained_net) # 双方都用训练好的策略

    renderer = CQBRenderer(env.map)
    window_name = "CQB Real-time Sim"

    MAX_STEPS = 1000
    print("Simulating... (Press 'q' in the window to stop)")
    
    for t in range(MAX_STEPS):
        actions = {}
        for i in range(env.agents_total):
            if i not in obs: continue 
            
            agent_obs = obs[i]
            if i < env.n_a:
                act = policy_A.get_action(agent_obs)
            else:
                act = policy_B.get_action(agent_obs)
            actions[i] = act
            
        obs, rewards, dones, done_all = env.step(actions)
        
        current_frame_data = env.event_log[-1]
        img = renderer.render_frame(current_frame_data['states'], current_frame_data['hits'])
        cv2.putText(img, f"Step: {t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        try:
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Simulation stopped by user.")
                break
        except cv2.error:
            pass
        except Exception as e:
            pass
        
        if done_all:
            print(f"Game Over at step {t}")
            break
            
    try:
        cv2.destroyAllWindows()
    except:
        pass

    log_file = "cqb_replay.json"
    env.save_replay(log_file)
    
    video_file = "cqb_match.mp4"
    print(f"Rendering final video to {video_file}...")
    renderer.render_replay(log_file, video_file)

if __name__ == "__main__":
    run_simulation()