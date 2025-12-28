import torch
import numpy as np
import cv2
import os
import json
import argparse
import math
from tqdm import tqdm

from simulator import CQBSimulator, CFG
from net import CQBTransformerPolicy

# ==========================================
# 1. 渲染器 (Visualizer)
# ==========================================
class CQBRenderer:
    def __init__(self, map_data, scale=8):
        """
        Args:
            map_data: shape (H, W), 0 for floor, 1 for wall. Can be numpy or tensor.
            scale: pixel scale factor
        """
        if isinstance(map_data, torch.Tensor):
            self.map = map_data.cpu().numpy()
        else:
            self.map = np.array(map_data)
            
        self.H, self.W = self.map.shape
        self.scale = scale
        
        # 预绘制背景
        # 1. 先在原始尺寸上绘制
        small_bg = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        small_bg[self.map == 0] = (230, 230, 230) # 地板: 浅灰
        small_bg[self.map == 1] = (60, 60, 60)    # 墙壁: 深灰
        
        # 2. 使用最近邻插值放大到目标尺寸 (保持边缘锐利)
        self.bg = cv2.resize(
            small_bg, 
            (self.W * self.scale, self.H * self.scale), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # 强制绘制网格线 (可选，让地图看起来更有战术感)
        # for x in range(0, self.W * scale, scale):
        #     cv2.line(self.bg, (x, 0), (x, self.H * scale), (220, 220, 220), 1)
        # for y in range(0, self.H * scale, scale):
        #     cv2.line(self.bg, (0, y), (self.W * scale, y), (220, 220, 220), 1)

    def render_frame(self, state_list, hits):
        canvas = self.bg.copy()
        
        # 绘制击中效果 (短暂的红线或火花)
        for hit in hits:
            # 由于日志里只记录了击中者的ID和被击中者的位置，
            # 为了画线，我们需要知道射击者的位置。
            # 但 state_list 是当前帧的状态。
            # 这里简化处理：只在被击中位置画一个爆炸标记
            target_loc = hit['loc']
            tx, ty = int(target_loc[0] * self.scale), int(target_loc[1] * self.scale)
            cv2.circle(canvas, (tx, ty), 4, (0, 0, 255), -1) # 红色命中点
            cv2.circle(canvas, (tx, ty), 8, (0, 165, 255), 1) # 橙色扩散圈

        # 绘制智能体
        for i, s in enumerate(state_list):
            # s 结构: [x, y, vx, vy, theta, omega, hp, c, n, r, sigma, team]
            x, y = s[0], s[1]
            theta = s[4]
            hp = s[6]
            team = int(s[11])
            is_reloading = s[9] > 0
            
            if hp <= 0: 
                # 绘制尸体 (灰色叉叉)
                cx, cy = int(x * self.scale), int(y * self.scale)
                r = int(CFG.RADIUS * self.scale)
                cv2.line(canvas, (cx-r, cy-r), (cx+r, cy+r), (150, 150, 150), 2)
                cv2.line(canvas, (cx+r, cy-r), (cx-r, cy+r), (150, 150, 150), 2)
                continue 
            
            cx, cy = int(x * self.scale), int(y * self.scale)
            
            # 阵营颜色 (BGR)
            # Team 0 (Red): (50, 50, 200)
            # Team 1 (Blue): (200, 100, 50)
            color = (50, 50, 220) if team == 0 else (220, 150, 50)
            
            # 1. 视锥 (FOV) - 半透明扇形
            # 为了性能，这里只画两条线表示视野范围
            fov_len = 30
            fov_l = theta - CFG.FOV / 2
            fov_r = theta + CFG.FOV / 2
            lx = int(cx + math.cos(fov_l) * fov_len)
            ly = int(cy + math.sin(fov_l) * fov_len)
            rx = int(cx + math.cos(fov_r) * fov_len)
            ry = int(cy + math.sin(fov_r) * fov_len)
            
            overlay = canvas.copy()
            cv2.line(overlay, (cx, cy), (lx, ly), (200, 200, 200), 1)
            cv2.line(overlay, (cx, cy), (rx, ry), (200, 200, 200), 1)
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

            # 2. 身体
            radius = int(CFG.RADIUS * self.scale)
            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (0, 0, 0), 1) # 黑色描边
            
            # 3. 朝向指示器
            ex = int(cx + math.cos(theta) * radius * 1.5)
            ey = int(cy + math.sin(theta) * radius * 1.5)
            cv2.line(canvas, (cx, cy), (ex, ey), (0, 0, 0), 2)
            
            # 4. 信息状态 (血条 & 换弹)
            # 血条背景
            bar_w = 24
            bar_h = 4
            bx, by = cx - bar_w//2, cy - radius - 8
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            # 血条前景 (绿->红)
            hp_w = int(bar_w * hp)
            hp_color = (0, 255, 0) if hp > 0.5 else (0, 0, 255)
            cv2.rectangle(canvas, (bx, by), (bx + hp_w, by + bar_h), hp_color, -1)
            
            if is_reloading:
                cv2.putText(canvas, "RELOAD", (bx, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        return canvas

# ==========================================
# 2. 辅助函数
# ==========================================
def batch_obs(obs_list, device):
    """
    推理用的 Batch 处理，从观测字典列表生成 Tensor Batch
    """
    if not obs_list: return None
    batch = {'self': [], 'spatial': [], 'team': [], 'enemy': []}
    for o in obs_list:
        batch['self'].append(o['self'])
        batch['spatial'].append(o['spatial'])
        batch['team'].append(o['team'])
        batch['enemy'].append(o['enemy'])

    batch['self'] = torch.stack(batch['self']).to(device)
    batch['spatial'] = torch.stack(batch['spatial']).to(device)

    def pad_sequence(tensor_list):
        max_len = max([t.shape[0] for t in tensor_list])
        if max_len == 0: max_len = 1
        feature_dim = tensor_list[0].shape[1] if tensor_list[0].numel() > 0 else (10 if tensor_list is batch['team'] else 8)
        padded = torch.zeros((len(tensor_list), max_len, feature_dim), device=device)
        for i, t in enumerate(tensor_list):
            if t.numel() > 0: padded[i, :t.shape[0], :] = t
        return padded

    batch['team'] = pad_sequence(batch['team'])
    batch['enemy'] = pad_sequence(batch['enemy'])
    return batch

# ==========================================
# 3. 核心功能：实时对战
# ==========================================
def run_live(args):
    print(f"Initializing Simulation on {CFG.DEVICE}...")
    env = CQBSimulator(n_a=args.n_agents, n_b=args.n_agents)
    obs_dict = env.reset()
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    policy = CQBTransformerPolicy(action_dim=5).to(CFG.DEVICE)
    if os.path.exists(args.model_path):
        policy.load_state_dict(torch.load(args.model_path, map_location=CFG.DEVICE))
        print("Model loaded successfully.")
    else:
        print("Warning: Model file not found, using random weights!")
    policy.eval()
    
    # 初始化渲染器
    renderer = CQBRenderer(env.map.cpu())
    window_name = "CQB Simulation (Press 'q' to quit)"
    
    steps = 0
    max_steps = 1000
    
    print("Starting loop...")
    while True:
        # 1. 收集观测
        active_agents = [i for i, o in obs_dict.items() if o is not None]
        if not active_agents:
            print("All agents dead.")
            break
            
        obs_list = [obs_dict[i] for i in active_agents]
        batch_input = batch_obs(obs_list, CFG.DEVICE)
        
        # 2. 推理
        actions = {}
        with torch.no_grad():
            # 获取动作均值 (deterministic)
            mean, _, _ = policy(batch_input)
            mean = mean.cpu().numpy()
            
            for idx, agent_id in enumerate(active_agents):
                actions[agent_id] = mean[idx] # 使用确定性策略
        
        # 3. 环境步进
        obs_dict, frame_data, vanquished, all_vanquished = env.step(actions)
        
        # 4. 渲染
        img = renderer.render_frame(frame_data['states'], frame_data['hits'])
        
        # UI 信息
        cv2.putText(img, f"Step: {steps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Red Alive: {sum([1 for i in range(env.n_a) if not vanquished[i]])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"Blue Alive: {sum([1 for i in range(env.n_a, env.agents_total) if not vanquished[i]])}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        cv2.imshow(window_name, img)
        
        # 控制帧率 (例如 30fps = 33ms)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        steps += 1
        if all_vanquished or steps >= max_steps:
            print(f"Game Over at step {steps}")
            break
            
    cv2.destroyAllWindows()
    
    # 5. 保存完整回放 (包含地图)
    if args.save_replay:
        replay_data = {
            "map_h": CFG.H,
            "map_w": CFG.W,
            "map_data": env.map.cpu().numpy().tolist(), # 保存地图
            "log": env.event_log
        }
        with open(args.output_file, 'w') as f:
            json.dump(replay_data, f)
        print(f"Replay saved to {args.output_file}")

# ==========================================
# 4. 核心功能：录像回放 (渲染为视频)
# ==========================================
def render_replay(args):
    if not os.path.exists(args.input_file):
        print(f"Error: Replay file {args.input_file} not found.")
        return

    print(f"Loading replay from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        replay_data = json.load(f)
    
    # 兼容性处理：如果 JSON 是旧版本（只有 log list），没有 map
    if isinstance(replay_data, list):
        print("Warning: Legacy replay format detected (no map data). Rendering on blank map.")
        map_data = np.zeros((CFG.H, CFG.W)) # 空白地图
        logs = replay_data
    else:
        map_data = np.array(replay_data["map_data"])
        logs = replay_data["log"]
    
    # 初始化渲染器
    renderer = CQBRenderer(map_data)
    
    # 视频输出配置
    H, W = map_data.shape
    render_h, render_w = H * renderer.scale, W * renderer.scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, 30.0, (render_w, render_h))
    
    print(f"Rendering {len(logs)} frames to {args.output_video}...")
    
    for i, frame in enumerate(tqdm(logs)):
        img = renderer.render_frame(frame['states'], frame['hits'])
        
        # 添加进度条
        cv2.putText(img, f"Replay: {i}/{len(logs)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(img)
        
        # 可选：同时也显示窗口
        if not args.headless:
            cv2.imshow("Replay Rendering", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Rendering aborted.")
                break
    
    out.release()
    cv2.destroyAllWindows()
    print("Rendering complete.")

# ==========================================
# 5. 入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CQB Simulator Player & Renderer")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: live or replay", required=True)
    
    # 模式 A: 实时对战
    parser_live = subparsers.add_parser("live", help="Run a live simulation with model")
    parser_live.add_argument("--model_path", type=str, default="cqb_model.pth", help="Path to model weights")
    parser_live.add_argument("--n_agents", type=int, default=2, help="Agents per team")
    parser_live.add_argument("--save_replay", action="store_true", default=True, help="Save replay to json")
    parser_live.add_argument("--output_file", type=str, default="last_match.json", help="Replay output path")
    
    # 模式 B: 回放渲染
    parser_replay = subparsers.add_parser("replay", help="Render an existing replay file to video")
    parser_replay.add_argument("--input_file", type=str, default="last_match.json", help="Path to json replay")
    parser_replay.add_argument("--output_video", type=str, default="replay.mp4", help="Output video path")
    parser_replay.add_argument("--headless", action="store_true", help="Don't show window while rendering video")
    
    args = parser.parse_args()
    
    if args.mode == "live":
        run_live(args)
    elif args.mode == "replay":
        render_replay(args)