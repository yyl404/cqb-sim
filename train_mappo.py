import torch
import torch.optim as optim
import numpy as np
from collections import deque
from simulator import CQBSimulator
from utils import CFG, batch_obs
from models import ActorCriticRNN

class TrainConfig:
    LR: float = 3e-4
    GAMMA: float = 0.99
    CLIP_EPS: float = 0.2
    NUM_STEPS: int = 128
    BATCH_SIZE: int = 64
    TOTAL_UPDATES: int = 1000
    SEED: int = 42

def main():
    torch.manual_seed(TrainConfig.SEED)
    env = CQBSimulator(n_a=2, n_b=2)
    obs_dict = env.reset()
    
    # Dummy shape check
    dummy = batch_obs([obs_dict[0]], CFG.DEVICE)
    obs_shapes = {k: v.shape[1:] for k, v in dummy.items()}
    state_dim = env.state.shape[1] * env.agents_total
    
    agent = ActorCriticRNN(obs_shapes, state_dim).to(CFG.DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=TrainConfig.LR)
    
    print("Starting Physics-Based RL Training (Acceleration Control)...")
    
    h_state = torch.zeros(1, env.agents_total, agent.hidden_dim, device=CFG.DEVICE)
    
    for update in range(1, TrainConfig.TOTAL_UPDATES + 1):
        buffer = {'obs':[], 'act_m':[], 'act_t':[], 'rew':[], 'don':[], 'log':[]}
        
        for step in range(TrainConfig.NUM_STEPS):
            obs_list = [obs_dict[i] if obs_dict[i] else obs_dict[0] for i in range(env.agents_total)]
            batched_obs = batch_obs(obs_list, CFG.DEVICE)
            
            with torch.no_grad():
                (dist_m, dist_t), next_h, _ = agent.forward_actor(batched_obs, h_state)
                a_move = dist_m.sample()
                a_tac = dist_t.sample()
                log_prob = dist_m.log_prob(a_move).sum(-1) + dist_t.log_prob(a_tac).sum(-1)

            # Store
            buffer['obs'].append(batched_obs)
            buffer['act_m'].append(a_move)
            buffer['act_t'].append(a_tac)
            buffer['log'].append(log_prob)
            
            # Execute: Direct mapping Network [-1,1] -> Sim (Sim scales to ACCEL_MAX)
            actions_dict = {}
            for i in range(env.agents_total):
                if env.state[i, 6] > 0:
                    # Concatenate [ax, ay, alpha] + [fire, reload]
                    full_act = np.concatenate([a_move[i].cpu().numpy(), a_tac[i].cpu().numpy()])
                    actions_dict[i] = full_act
            
            obs_dict, _, _, _, all_done = env.step(actions_dict)
            
            # Simple Sparse Reward
            rewards = torch.zeros(env.agents_total, device=CFG.DEVICE)
            # ... (Add your reward logic here) ...
            
            buffer['rew'].append(rewards)
            buffer['don'].append(torch.tensor([all_done]*env.agents_total))
            
            h_state = next_h
            if all_done:
                obs_dict = env.reset()
                h_state = torch.zeros(1, env.agents_total, agent.hidden_dim, device=CFG.DEVICE)
        
        # --- PPO Update (Simplified) ---
        # Calculation of returns/advantages and gradient steps omitted for brevity 
        # but logic is identical to previous phase.
        print(f"Update {update} complete.")

if __name__ == "__main__":
    main()