import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import argparse
import os
import time
from collections import deque, defaultdict

from simulator import CQBSimulator, CFG
from net import CQBTransformerPolicy

# ==========================================
# 1. 辅助函数：数据整理与 Padding
# ==========================================

def batch_obs(obs_list, device):
    """
    将多个智能体的观测字典列表整理为一个 Batch 字典。
    自动对 team 和 enemy 的变长序列进行 Padding。
    """
    if not obs_list:
        return None

    batch = {
        'self': [],
        'spatial': [],
        'team': [],
        'enemy': []
    }

    # 1. 收集数据
    for o in obs_list:
        batch['self'].append(o['self'])
        batch['spatial'].append(o['spatial'])
        batch['team'].append(o['team'])
        batch['enemy'].append(o['enemy'])

    # 2. 堆叠定长数据
    batch['self'] = torch.stack(batch['self']).to(device)
    batch['spatial'] = torch.stack(batch['spatial']).to(device)

    # 3. 处理变长序列 (Team & Enemy) -> Padding
    def pad_sequence(tensor_list):
        max_len = max([t.shape[0] for t in tensor_list])
        if max_len == 0:
            max_len = 1
        
        feature_dim = tensor_list[0].shape[1] if tensor_list[0].numel() > 0 else (10 if tensor_list is batch['team'] else 8)
        
        padded_batch = torch.zeros((len(tensor_list), max_len, feature_dim), device=device)
        
        for i, t in enumerate(tensor_list):
            if t.numel() > 0:
                length = t.shape[0]
                padded_batch[i, :length, :] = t
        
        return padded_batch

    batch['team'] = pad_sequence(batch['team'])
    batch['enemy'] = pad_sequence(batch['enemy'])

    return batch

# ==========================================
# 2. 奖励函数设计 (修改版)
# ==========================================

class CombatReward:
    def __init__(self):
        # --- 战斗奖励 ---
        self.rew_hit_enemy = 2.0    # 造成 1.0 伤害的奖励
        self.rew_kill = 5.0         # 击杀奖励
        self.rew_be_hit = -1.0      # 受伤惩罚
        self.rew_die = -10.0        # 死亡惩罚
        
        # --- 全局奖励 ---
        self.rew_win = 20.0         # 胜利奖励
        self.rew_lose = -5.0        # 失败惩罚
        self.rew_step = -0.01       # 步数惩罚 (稍微加大，鼓励快速行动)
        
        # --- 动作约束 (解决原地转圈问题) ---
        self.rew_waste_ammo = -0.1  # 射击惩罚 (加大，防止乱开枪)
        self.rew_spin = -0.05       # 【新增】旋转惩罚：惩罚角速度的绝对值
        self.rew_static = -0.02     # 【新增】静止惩罚：如果不移动，给予小惩罚

    def compute(self, frame_data, vanquished, agents_prev_hp, actions_dict, simulator):
        """
        计算当前步的奖励
        返回: 
            rewards: Dict[int, float] -> 每个智能体的总奖励
            stats: Dict[str, float] -> 用于日志记录的各项奖励总和
        """
        rewards = {}
        # 初始化统计字典
        stats = defaultdict(float)
        
        # 只为存活的智能体计算基础奖励
        active_ids = [i for i in range(simulator.agents_total) if not vanquished.get(i, True)]
        
        for i in active_ids:
            rewards[i] = self.rew_step
            stats['step'] += self.rew_step

        # 1. 命中与伤害奖励
        hits = frame_data['hits']
        current_states = frame_data['states']
        
        for hit in hits:
            shooter = hit['shooter']
            target = hit['target']
            dmg = hit['damage']
            
            # 攻击者奖励
            if shooter in rewards:
                r = dmg * self.rew_hit_enemy
                rewards[shooter] += r
                stats['hit_dmg'] += r
            
            # 受害者惩罚
            if target in rewards:
                r = dmg * self.rew_be_hit
                rewards[target] += r
                stats['be_hit'] += r
                
        # 2. 击杀与死亡判定
        for i in range(simulator.agents_total):
            curr_hp = current_states[i][6]
            prev_hp = agents_prev_hp.get(i, 1.0)
            
            # 死亡判定
            if curr_hp <= 0 and prev_hp > 0:
                if i in rewards:
                    rewards[i] += self.rew_die
                    stats['die'] += self.rew_die
        
        # 击杀判定 (补丁逻辑)
        for hit in hits:
            target = hit['target']
            shooter = hit['shooter']
            if current_states[target][6] <= 0 and agents_prev_hp.get(target, 0) > 0:
                if shooter in rewards:
                    rewards[shooter] += self.rew_kill
                    stats['kill'] += self.rew_kill

        # 3. 动作惩罚 (Action Penalty)
        for i, act in actions_dict.items():
            if i not in rewards: continue
            
            # A. 弹药惩罚 (action[3] > 0.5)
            if act[3] > 0.5: 
                rewards[i] += self.rew_waste_ammo
                stats['ammo'] += self.rew_waste_ammo
            
            # B. 【关键】旋转惩罚 (action[2] is omega in [-1, 1])
            # 惩罚大幅度转向，鼓励平滑瞄准或仅在必要时转向
            spin_mag = abs(act[2])
            r_spin = spin_mag * self.rew_spin
            rewards[i] += r_spin
            stats['spin'] += r_spin
            
            # C. 【关键】静止惩罚 (鼓励移动)
            # action[0] is surge, action[1] is sway
            move_speed = np.sqrt(act[0]**2 + act[1]**2)
            if move_speed < 0.1: # 几乎静止
                rewards[i] += self.rew_static
                stats['static'] += self.rew_static

        return rewards, stats

# ==========================================
# 3. PPO Agent
# ==========================================

class PPOAgent:
    def __init__(self, args):
        self.device = CFG.DEVICE
        self.lr = args.lr
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_eps = args.clip_eps
        self.ent_coef = args.ent_coef
        self.max_grad_norm = 0.5
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        self.policy = CQBTransformerPolicy(
            action_dim=5, 
            embed_dim=256,
            nhead=4,
            num_layers=2
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.buffer = []

    def select_action(self, obs_batch_dict):
        self.policy.eval()
        with torch.no_grad():
            mean, std, value = self.policy(obs_batch_dict)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action_clipped = torch.clamp(action, -1.0, 1.0)
            
        return action_clipped.cpu().numpy(), action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def store(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        self.policy.train()
        
        obs_list = [t[0] for t in self.buffer]
        raw_actions = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(np.array([t[2] for t in self.buffer]), dtype=torch.float32, device=self.device)
        rewards = [t[3] for t in self.buffer]
        values = torch.tensor(np.array([t[4] for t in self.buffer]), dtype=torch.float32, device=self.device).squeeze()
        dones = [t[5] for t in self.buffer]

        advantages = []
        gae = 0
        next_value = 0
        
        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[i])
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[i]
            
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        buffer_size = len(self.buffer)
        indices = np.arange(buffer_size)
        avg_loss = 0
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, buffer_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                mb_obs_list = [obs_list[i] for i in idx]
                mb_obs = batch_obs(mb_obs_list, self.device)
                
                mb_actions = raw_actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                mean, std, curr_values = self.policy(mb_obs)
                dist = Normal(mean, std)
                curr_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                curr_values = curr_values.squeeze()
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(curr_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * nn.MSELoss()(curr_values, mb_returns)
                loss = actor_loss + critic_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                avg_loss += loss.item()

        self.buffer = []
        return avg_loss / (self.n_epochs * (buffer_size // self.batch_size + 1))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            try:
                self.policy.load_state_dict(torch.load(path, map_location=self.device))
                print(f"Loaded model from {path}")
            except Exception as e:
                print(f"Failed to load model: {e}, starting scratch.")
        else:
            print(f"No model found at {path}, starting from scratch.")

# ==========================================
# 4. 主训练循环
# ==========================================

def run_training(args):
    env = CQBSimulator(n_a=2, n_b=2)
    agent = PPOAgent(args)
    reward_calculator = CombatReward()
    
    agent.load(args.model_path)
    
    # 统计滑动窗口
    running_reward = deque(maxlen=20)
    
    print(f"Start training on {CFG.DEVICE}...")
    
    for episode in range(args.max_episodes):
        obs_dict = env.reset()
        agents_prev_hp = {i: env.state[i, 6].item() for i in range(env.agents_total)}
        
        ep_reward = 0
        ep_steps = 0
        
        # 记录本局各分项奖励总和
        ep_stats = defaultdict(float)
        
        while True:
            # 过滤掉观测为 None 的已死亡智能体
            active_agents = [i for i, o in obs_dict.items() if o is not None]
            if not active_agents:
                break
            
            obs_list = [obs_dict[i] for i in active_agents]
            batch_input = batch_obs(obs_list, agent.device)
            
            actions_clipped, raw_actions, log_probs, values = agent.select_action(batch_input)
            
            step_actions = {}
            for idx, agent_id in enumerate(active_agents):
                step_actions[agent_id] = actions_clipped[idx]
            
            next_obs_dict, frame_data, vanquished, all_vanquished = env.step(step_actions)
            
            # 计算奖励 (获取 rewards 和 stats)
            rewards_dict, step_stats_dict = reward_calculator.compute(frame_data, vanquished, agents_prev_hp, step_actions, env)
            
            # 累加本步的统计信息到本局总览
            for k, v in step_stats_dict.items():
                ep_stats[k] += v

            # 胜负奖励逻辑
            if all_vanquished:
                a_alive = any([not vanquished[i] for i in range(env.n_a)])
                b_alive = any([not vanquished[i] for i in range(env.n_a, env.agents_total)])
                win_team = -1
                if a_alive and not b_alive: win_team = 0
                elif b_alive and not a_alive: win_team = 1
                
                for i in range(env.agents_total):
                    if i in rewards_dict:
                         team = env.state[i, 11].item()
                         if team == win_team:
                             r = reward_calculator.rew_win
                             rewards_dict[i] += r
                             ep_stats['win'] += r
                         else:
                             r = reward_calculator.rew_lose
                             rewards_dict[i] += r
                             ep_stats['lose'] += r

            for idx, agent_id in enumerate(active_agents):
                o = obs_dict[agent_id]
                ra = raw_actions[idx]
                lp = log_probs[idx]
                val = values[idx]
                r = rewards_dict.get(agent_id, 0.0)
                done = vanquished.get(agent_id, False) or all_vanquished
                
                agent.store((o, ra, lp, r, val, done))
                ep_reward += r
            
            obs_dict = next_obs_dict
            for i in range(env.agents_total):
                agents_prev_hp[i] = env.state[i, 6].item()
            
            ep_steps += 1
            
            if all_vanquished or ep_steps >= args.max_steps_per_ep:
                break
        
        # 更新网络
        loss = 0
        if len(agent.buffer) >= args.update_timesteps:
            loss = agent.update()
            agent.save(args.model_path)
            
        running_reward.append(ep_reward)
        avg_reward = sum(running_reward) / len(running_reward)
        
        # 详细日志输出
        if episode % 1 == 0:
            # 格式化 stats 字符串
            stats_str = " | ".join([f"{k}:{v:.1f}" for k, v in ep_stats.items() if abs(v) > 0.1])
            print(f"Ep {episode:<4} | R: {ep_reward:>6.1f} | Avg: {avg_reward:>6.1f} | Loss: {loss:.3f} || {stats_str}")

    print("Training finished.")
    agent.save(args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for CQB Simulator")
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128) # 稍微调大 batch size
    parser.add_argument("--n_epochs", type=int, default=4)
    
    # 循环参数
    parser.add_argument("--max_episodes", type=int, default=5000)
    parser.add_argument("--max_steps_per_ep", type=int, default=500)
    parser.add_argument("--update_timesteps", type=int, default=2000)
    
    # 路径
    parser.add_argument("--model_path", type=str, default="cqb_model.pth")
    
    args = parser.parse_args()
    run_training(args)