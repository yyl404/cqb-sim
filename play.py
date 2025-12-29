import torch
import numpy as np
import cv2
import os
import json
import argparse
import math
from tqdm import tqdm
import mediapy

# 引入simulator中的类和配置
from simulator import CQBSimulator, CFG

# ==========================================
# 1. 渲染器 (Visualizer) - 修改版
# ==========================================
class CQBRenderer:
    # --- 特效配置 ---
    SHOT_DURATION = 0.5       # 线条持续时间 (秒)
    SHOT_VIS_LENGTH = 30.0    # 最大视觉长度 (米)
    SHOT_FADE_DISTANCE = 1.5  # 靠近射手的淡出/透明遮罩半径 (米)
    SHOT_COLOR = (0, 255, 255) # 黄色 (BGR)

    def __init__(self, map_data, scale=8):
        # 处理地图数据
        if isinstance(map_data, torch.Tensor):
            self.map = map_data.cpu().numpy()
        else:
            self.map = np.array(map_data)
            
        self.H, self.W = self.map.shape
        self.scale = scale
        
        # 1. 预绘制背景
        small_bg = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        small_bg[self.map == 0] = (230, 230, 230) # 地板: 浅灰
        small_bg[self.map == 1] = (60, 60, 60)    # 墙壁: 深灰
        
        # 2. 放大到渲染尺寸
        self.bg = cv2.resize(
            small_bg, 
            (self.W * self.scale, self.H * self.scale), 
            interpolation=cv2.INTER_NEAREST
        )

        # 3. 状态追踪变量
        self.last_ammo = {}    # {agent_id: ammo_count}
        self.active_shots = [] # [{'start': (x,y), 'end': (x,y), 'time': t}, ...]

    def _cast_ray_to_wall(self, start_pos, theta):
        """
        视觉层面的 DDA 射线检测，用于计算击中墙壁的坐标
        """
        x1, y1 = start_pos
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # 最大射程终点
        x2 = x1 + cos_t * self.SHOT_VIS_LENGTH
        y2 = y1 + sin_t * self.SHOT_VIS_LENGTH

        # --- DDA 初始化 ---
        map_x = int(math.floor(x1))
        map_y = int(math.floor(y1))
        
        ray_dir_x = cos_t
        ray_dir_y = sin_t
        
        delta_dist_x = abs(1.0 / (ray_dir_x + 1e-9))
        delta_dist_y = abs(1.0 / (ray_dir_y + 1e-9))

        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (x1 - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - x1) * delta_dist_x

        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (y1 - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - y1) * delta_dist_y

        # --- DDA 步进 ---
        # 我们需要记录总距离，以便计算精确撞击点
        hit = False
        side = 0 # 0 for x-hit, 1 for y-hit
        
        # 限制最大步数，避免死循环
        max_steps = int(self.SHOT_VIS_LENGTH * 2) 
        
        for _ in range(max_steps):
            # 碰撞检查
            if 0 <= map_x < self.W and 0 <= map_y < self.H:
                if self.map[map_y, map_x] > 0.5:
                    hit = True
                    break
            else:
                # 出界视为撞墙
                hit = True
                break
                
            # 步进
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1

        # --- 计算精确终点 ---
        if hit:
            # 计算撞墙时的距离
            # 公式原理：(map_x - x1 + (1-stepX)/2) / ray_dir_x
            if side == 0:
                perp_wall_dist = (map_x - x1 + (1 - step_x) / 2) / (ray_dir_x + 1e-9)
            else:
                perp_wall_dist = (map_y - y1 + (1 - step_y) / 2) / (ray_dir_y + 1e-9)
            
            # 最终坐标
            hit_x = x1 + ray_dir_x * perp_wall_dist
            hit_y = y1 + ray_dir_y * perp_wall_dist
            return (hit_x, hit_y)
        else:
            # 没撞墙，返回最大射程处
            return (x2, y2)

    def render_frame(self, state_list, hits, current_time):
        canvas = self.bg.copy()
        
        # --- A. 射击特效逻辑 ---
        
        # 1. 检测新的开火事件
        for i, s in enumerate(state_list):
            current_ammo = s[7]
            is_reloading = s[9] > 0
            
            # 判定开火
            if i in self.last_ammo:
                if current_ammo < self.last_ammo[i] and not is_reloading:
                    start_pos = (s[0], s[1])
                    theta = s[4]
                    
                    # --- 核心修改：确定线条终点 ---
                    end_pos = None
                    
                    # 优先检查：是否命中了智能体？
                    # 遍历 hits 列表，看当前 shooter 是否造成了伤害
                    for h in hits:
                        if h['shooter'] == i:
                            end_pos = tuple(h['loc']) # 截断在命中点
                            break
                    
                    # 如果没命中人，则检测墙壁 (截断在墙壁)
                    if end_pos is None:
                        end_pos = self._cast_ray_to_wall(start_pos, theta)
                    
                    self.active_shots.append({
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'theta': theta,
                        'start_time': current_time
                    })
            
            self.last_ammo[i] = current_ammo

        # 2. 清理过期特效
        self.active_shots = [
            shot for shot in self.active_shots 
            if (current_time - shot['start_time']) < self.SHOT_DURATION
        ]

        # 3. 绘制光束
        if self.active_shots:
            overlay = canvas.copy()
            floor_color = (230, 230, 230)
            
            for shot in self.active_shots:
                sx, sy = shot['start_pos']
                ex, ey = shot['end_pos'] # 使用已计算好的终点
                
                # 转为像素坐标
                cx, cy = int(sx * self.scale), int(sy * self.scale)
                end_x, end_y = int(ex * self.scale), int(ey * self.scale)
                
                # a. 绘制黄色实线 (使用截断后的终点)
                cv2.line(overlay, (cx, cy), (end_x, end_y), self.SHOT_COLOR, 2)
                
                # b. 绘制靠近射手的遮罩
                mask_radius = int(self.SHOT_FADE_DISTANCE * self.scale)
                cv2.circle(overlay, (cx, cy), mask_radius, floor_color, -1)

            # c. 混合图层
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # --- B. 绘制击中点 ---
        for hit in hits:
            target_loc = hit['loc']
            tx, ty = int(target_loc[0] * self.scale), int(target_loc[1] * self.scale)
            cv2.circle(canvas, (tx, ty), 4, (0, 0, 255), -1) 
            cv2.circle(canvas, (tx, ty), 8, (0, 165, 255), 1)

        # --- C. 绘制智能体 ---
        for i, s in enumerate(state_list):
            x, y = s[0], s[1]
            theta = s[4]
            hp = s[6]
            team = int(s[11])
            is_reloading = s[9] > 0
            
            cx, cy = int(x * self.scale), int(y * self.scale)
            
            # 1. 尸体
            if hp <= 0: 
                r = int(CFG.RADIUS * self.scale)
                cv2.line(canvas, (cx-r, cy-r), (cx+r, cy+r), (150, 150, 150), 2)
                cv2.line(canvas, (cx+r, cy-r), (cx-r, cy+r), (150, 150, 150), 2)
                continue 
            
            color = (50, 50, 220) if team == 0 else (220, 150, 50)
            
            # 2. 视锥
            fov_len = 30
            fov_l = theta - CFG.FOV / 2
            fov_r = theta + CFG.FOV / 2
            lx = int(cx + math.cos(fov_l) * fov_len)
            ly = int(cy + math.sin(fov_l) * fov_len)
            rx = int(cx + math.cos(fov_r) * fov_len)
            ry = int(cy + math.sin(fov_r) * fov_len)
            
            cv2.line(canvas, (cx, cy), (lx, ly), (200, 200, 200), 1)
            cv2.line(canvas, (cx, cy), (rx, ry), (200, 200, 200), 1)

            # 3. 身体
            radius = int(CFG.RADIUS * self.scale)
            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (0, 0, 0), 1)
            
            # 4. 朝向
            ex = int(cx + math.cos(theta) * radius * 1.5)
            ey = int(cy + math.sin(theta) * radius * 1.5)
            cv2.line(canvas, (cx, cy), (ex, ey), (0, 0, 0), 2)
            
            # 5. 血条与状态
            bar_w = 24
            bar_h = 4
            bx, by = cx - bar_w//2, cy - radius - 8
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            hp_w = int(bar_w * hp)
            hp_color = (0, 255, 0) if hp > 0.5 else (0, 0, 255)
            cv2.rectangle(canvas, (bx, by), (bx + hp_w, by + bar_h), hp_color, -1)
            
            if is_reloading:
                cv2.putText(canvas, "R", (bx, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        return canvas

# ==========================================
# 2. 辅助函数 (Obs处理)
# ==========================================
def batch_obs(obs_list, device):
    """
    将观测字典列表转换为Tensor Batch (如果需要接入神经网络)
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
# 3. 运行逻辑 (Live & Replay)
# ==========================================

def run_live(args):
    """
    实时仿真模式
    """
    print(f"Initializing Simulation on {CFG.DEVICE}...")
    env = CQBSimulator(n_a=args.n_agents, n_b=args.n_agents)
    obs_dict = env.reset()
    
    # 初始化渲染器
    renderer = None
    window_name = "CQB Simulation"
    if not args.headless:
        renderer = CQBRenderer(env.map.cpu())
    else:
        print("Headless mode enabled. No GUI window will be shown.")
    
    steps = 0
    max_steps = 1000
    
    print("Starting simulation loop...")
    pbar = tqdm(total=max_steps)
    
    while True:
        # 1. 收集存活智能体观测
        active_agents = [i for i, o in obs_dict.items() if o is not None]
        if not active_agents:
            print("\nAll agents dead.")
            break
            
        obs_list = [obs_dict[i] for i in active_agents]
        
        # 2. 策略推理 (此处为演示，使用简单的随机/测试策略)
        # 如果有神经网络，可以在此处调用 batch_obs 和 model.forward
        actions = {}
        # mean = torch.zeros([len(obs_list), 5]) 
        # mean[:, 3] = 1.0 # 取消注释此行可以让所有单位持续开火以测试光束特效
        
        # 简单的随机游走 + 随机开火策略用于演示
        for idx, agent_id in enumerate(active_agents):
            # [surge, sway, omega, fire, reload]
            act = np.zeros(5)
            act[0] = np.random.uniform(0.5, 1.0) # 前进
            act[2] = np.random.uniform(-0.5, 0.5) # 转向
            if np.random.rand() > 0.8: # 20%概率开火
                act[3] = 1.0
            actions[agent_id] = act
        
        # 3. 环境步进
        obs_dict, frame_data, vanquished, all_vanquished = env.step(actions)
        
        # 4. 渲染
        if not args.headless:
            # 传入 env.time 以计算特效持续时间
            img = renderer.render_frame(frame_data['states'], frame_data['hits'], env.time)
            
            # UI 信息
            cv2.putText(img, f"Step: {steps} Time: {env.time:.2f}s", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow(window_name, img)
            
            # 按Q退出
            if cv2.waitKey(33) & 0xFF == ord('q'):
                print("\nStopped by user.")
                break
        
        steps += 1
        pbar.update(1)
        
        if all_vanquished or steps >= max_steps:
            print(f"\nGame Over at step {steps}")
            break
            
    pbar.close()
    if not args.headless:
        cv2.destroyAllWindows()
    
    # 保存回放
    if args.save_replay:
        replay_data = {
            "map_h": CFG.H,
            "map_w": CFG.W,
            "map_data": env.map.cpu().numpy().tolist(),
            "log": env.event_log
        }
        with open(args.output_file, 'w') as f:
            json.dump(replay_data, f)
        print(f"Replay successfully saved to {args.output_file}")


def render_replay(args):
    """
    回放渲染模式 (生成视频)
    """
    if not os.path.exists(args.input_file):
        print(f"Error: Replay file {args.input_file} not found.")
        return

    print(f"Loading replay from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        replay_data = json.load(f)
    
    # 兼容旧格式
    if isinstance(replay_data, list):
        print("Warning: Legacy format. Using blank map.")
        map_data = np.zeros((CFG.H, CFG.W))
        logs = replay_data
    else:
        map_data = np.array(replay_data["map_data"])
        logs = replay_data["log"]
    
    renderer = CQBRenderer(map_data)
    
    print(f"Rendering {len(logs)} frames to {args.output_video}...")
    frames = []
    
    for i, frame in enumerate(tqdm(logs)):
        # 获取当前帧时间，如果没有则按步长计算
        current_time = frame.get('time', i * CFG.DT)
        
        img = renderer.render_frame(frame['states'], frame['hits'], current_time)
        
        cv2.putText(img, f"Replay: {i}/{len(logs)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # OpenCV (BGR) -> Mediapy (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)
    
    # 保存视频
    mediapy.write_video(args.output_video, frames, fps=30)
    print("Rendering complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CQB Simulator Player")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # 模式 A: 实时运行 (Live)
    parser_live = subparsers.add_parser("live")
    parser_live.add_argument("--model_path", type=str, default="cqb_model.pth")
    parser_live.add_argument("--n_agents", type=int, default=4)
    parser_live.add_argument("--save_replay", action="store_true", default=True)
    parser_live.add_argument("--output_file", type=str, default="last_match.json")
    parser_live.add_argument("--headless", action="store_true", help="Run without GUI")
    
    # 模式 B: 回放 (Replay)
    parser_replay = subparsers.add_parser("replay")
    parser_replay.add_argument("--input_file", type=str, default="last_match.json")
    parser_replay.add_argument("--output_video", type=str, default="replay.mp4")
    
    args = parser.parse_args()
    
    if args.mode == "live":
        run_live(args)
    elif args.mode == "replay":
        render_replay(args)