import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# 从 cqb.py 导入所有需要的类，包括刚刚移动过去的 PPOActorCritic
from cqb import CQBSimulator, CQBConfig, CQBRenderer, PPOActorCritic

# ==========================================
# 2. PPO 算法核心类
# ==========================================
class PPOAgent:
    def __init__(self, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        
        self.policy = PPOActorCritic().to(CQBConfig.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 存储轨迹的 Buffer
        self.buffer = []

    def select_action(self, obs_dict):
        """在与环境交互时调用，不计算梯度"""
        self.policy.eval()
        with torch.no_grad():
            o_self = obs_dict['self'].unsqueeze(0)    # [1, Dim]
            o_spatial = obs_dict['spatial'].unsqueeze(0) # [1, Dim]
            
            mean, std, value = self.policy(o_self, o_spatial)
            
            # 构建正态分布进行采样
            dist = Normal(mean, std)
            action = dist.sample()
            
            # 计算当前动作的对数概率 (用于后续计算 Loss)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # 动作截断到合法范围 [-1, 1] (虽然 mean 已经是 tanh，但 sample 可能会越界)
            action_clipped = torch.clamp(action, -1.0, 1.0)
            
        return action_clipped.cpu().numpy()[0], log_prob, value

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        """使用收集到的轨迹更新网络"""
        if len(self.buffer) == 0: return

        self.policy.train()
        
        # 1. 整理数据
        # 这里的 obs 需要特殊处理，因为是字典
        o_self_batch = torch.stack([t[0]['self'] for t in self.buffer])
        o_spatial_batch = torch.stack([t[0]['spatial'] for t in self.buffer])
        actions = torch.tensor(np.array([t[1] for t in self.buffer]), device=CQBConfig.DEVICE)
        old_log_probs = torch.stack([t[2] for t in self.buffer]).detach()
        rewards = [t[3] for t in self.buffer]
        values = torch.stack([t[4] for t in self.buffer]).squeeze().detach()
        # 修复：强制转换为 float (0.0 或 1.0)，避免布尔运算报错
        dones = [float(t[5]) for t in self.buffer]

        # 2. 计算 GAE (Generalized Advantage Estimation)
        advantages = []
        gae = 0
        # 倒序计算
        next_value = 0 # 简化处理：假设最后一个状态后 value=0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                delta = rewards[i] - values[i] # 简化
            else:
                delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, device=CQBConfig.DEVICE)
        returns = advantages + values # Returns = Advantage + Value

        # 归一化 Advantage (稳定训练的关键)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO Update (多轮 Epoch 更新)
        for _ in range(4): # Epochs = 4
            # 重新评估当前策略下的 log_prob 和 value
            mean, std, curr_values = self.policy(o_self_batch, o_spatial_batch)
            dist = Normal(mean, std)
            curr_log_probs = dist.log_prob(actions).sum(dim=-1)
            curr_values = curr_values.squeeze()
            
            # Ratio
            ratio = torch.exp(curr_log_probs - old_log_probs)
            
            # Surrogate Loss (PPO Clip)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss (MSE)
            critic_loss = nn.MSELoss()(curr_values, returns)
            
            # Entropy Loss (鼓励探索)
            entropy_loss = -dist.entropy().mean() * 0.01
            
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # 梯度裁剪
            self.optimizer.step()
            
        # 清空 Buffer
        self.buffer = []
        return loss.item()

# ==========================================
# 3. 训练主循环
# ==========================================
def train():
    print(f"Training on: {CQBConfig.DEVICE}")
    env = CQBSimulator(n_a=2, n_b=2)
    
    # 我们只训练 A 队的 Agent，B 队使用随机策略作为陪练
    # 也可以实现 Self-Play，让 Agent 自己打自己
    agent = PPOAgent()
    
    MAX_EPISODES = 100
    MAX_STEPS = 200
    UPDATE_TIMESTEP = 500 # 每收集多少步数据更新一次网络
    
    global_step = 0
    
    for episode in range(MAX_EPISODES):
        obs = env.reset()
        episode_reward = 0
        
        for t in range(MAX_STEPS):
            actions_dict = {}
            
            # --- 1. 获取动作 ---
            # 存活的 Agent A 使用 PPO 策略
            agent_transitions = {} # 暂存 (obs, action, log_prob, val)
            
            for i in range(env.agents_total):
                if i not in obs: continue # 死人跳过
                
                if i < env.n_a:
                    # 训练对象：A 队
                    action, log_prob, value = agent.select_action(obs[i])
                    actions_dict[i] = action
                    agent_transitions[i] = (obs[i], action, log_prob, value)
                else:
                    # 陪练对象：B 队 (随机)
                    actions_dict[i] = np.random.uniform(-1, 1, 5)
                    # 降低随机开火频率
                    actions_dict[i][3] = 1.0 if np.random.rand() > 0.95 else 0.0
            
            # --- 2. 环境步进 ---
            next_obs, rewards, dones, done_all = env.step(actions_dict)
            
            # --- 3. 收集数据 (只收集 A 队的数据) ---
            for i in range(env.n_a):
                if i in agent_transitions:
                    # 取出刚才决策时的信息
                    o, a, lp, v = agent_transitions[i]
                    r = rewards.get(i, 0)
                    d = dones.get(i, False)
                    
                    # 存入 buffer: (obs, action, log_prob, reward, value, done)
                    agent.store_transition((o, a, lp, r, v, d))
                    episode_reward += r
            
            obs = next_obs
            global_step += 1
            
            # --- 4. 更新网络 ---
            if global_step % UPDATE_TIMESTEP == 0:
                loss = agent.update()
                print(f"Update at step {global_step}, Loss: {loss:.4f}")
            
            if done_all:
                break
                
        print(f"Episode {episode+1} Reward: {episode_reward:.2f}")

    # 保存模型
    torch.save(agent.policy.state_dict(), "ppo_cqb.pth")
    print("Training Finished. Model saved.")

if __name__ == "__main__":
    train()