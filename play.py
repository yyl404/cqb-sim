import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapy
import numpy as np
from tqdm import tqdm

from policy import NaiveCQBPolicy, RLInferencePolicy
from simulator import CQBSimulator
from utils import CFG


class CQBRenderer:
    SHOT_DURATION: float = 0.5
    SHOT_VIS_LENGTH: float = 30.0
    SHOT_FADE_DISTANCE: float = 1.5
    SHOT_COLOR: Tuple[int, int, int] = (0, 255, 255)
    
    FOW_RAY_STEP: float = 2.0
    FOW_DIM_FACTOR: float = 0.3

    def __init__(self, map_data: np.ndarray, scale: int = 8) -> None:
        self.map = map_data
        self.H, self.W = self.map.shape
        self.scale = scale

        small_bg = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        small_bg[self.map == 0] = (230, 230, 230)
        small_bg[self.map > 0.5] = (60, 60, 60)

        self.bg = cv2.resize(
            small_bg,
            (self.W * self.scale, self.H * self.scale),
            interpolation=cv2.INTER_NEAREST,
        )

        self.last_ammo: Dict[int, int] = {}
        self.active_shots: List[Dict[str, Any]] = []

    def _cast_ray_to_wall(self, start_pos: Tuple[float, float], theta: float, max_dist: float = 30.0) -> Tuple[float, float]:
        x1, y1 = start_pos
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x2, y2 = x1 + cos_t * max_dist, y1 + sin_t * max_dist

        map_x, map_y = int(math.floor(x1)), int(math.floor(y1))
        
        delta_dist_x = abs(1.0 / (cos_t + 1e-9))
        delta_dist_y = abs(1.0 / (sin_t + 1e-9))

        step_x = 1 if cos_t >= 0 else -1
        side_dist_x = (map_x + 1.0 - x1) * delta_dist_x if cos_t >= 0 else (x1 - map_x) * delta_dist_x

        step_y = 1 if sin_t >= 0 else -1
        side_dist_y = (map_y + 1.0 - y1) * delta_dist_y if sin_t >= 0 else (y1 - map_y) * delta_dist_y

        hit = False
        side = 0
        
        for _ in range(int(max_dist * 2)):
            if 0 <= map_x < self.W and 0 <= map_y < self.H:
                if self.map[map_y, map_x] > 0.5:
                    hit = True
                    break
            else:
                hit = True
                break

            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1

        if hit:
            if side == 0:
                perp = (map_x - x1 + (1 - step_x) / 2) / (cos_t + 1e-9)
            else:
                perp = (map_y - y1 + (1 - step_y) / 2) / (sin_t + 1e-9)
            return (x1 + cos_t * perp, y1 + sin_t * perp)

        return (x2, y2)

    def _compute_vision_mask(self, state_list: List[List[float]], spectate_id: int) -> np.ndarray:
        my_team = int(state_list[spectate_id][11])
        team_indices = [i for i, s in enumerate(state_list) if int(s[11]) == my_team and s[6] > 0]
        
        mask = np.zeros((self.H * self.scale, self.W * self.scale), dtype=np.uint8)
        
        for i in team_indices:
            s = state_list[i]
            cx, cy = s[0], s[1]
            theta = s[4]
            
            start_angle = np.degrees(theta - CFG.FOV / 2)
            end_angle = np.degrees(theta + CFG.FOV / 2)
            angles = np.arange(start_angle, end_angle + self.FOW_RAY_STEP, self.FOW_RAY_STEP)
            
            points = [[int(cx * self.scale), int(cy * self.scale)]]
            for ang_deg in angles:
                end_pt = self._cast_ray_to_wall((cx, cy), np.radians(ang_deg), max_dist=CFG.L/2)
                points.append([int(end_pt[0] * self.scale), int(end_pt[1] * self.scale)])
                
            pts = np.array([points], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
            
            # Unified Proximity Circle
            prox_radius = int(CFG.PROXIMITY_RADIUS * self.scale) 
            cv2.circle(mask, (int(cx*self.scale), int(cy*self.scale)), prox_radius, 255, -1)

        return mask

    def render_frame(self, state_list: List[List[float]], hits: List[Dict[str, Any]], current_time: float, spectate_id: Optional[int] = None) -> np.ndarray:
        canvas = self.bg.copy()
        
        vision_mask = None
        if spectate_id is not None and 0 <= spectate_id < len(state_list):
            vision_mask = self._compute_vision_mask(state_list, spectate_id)
            shadowed = (canvas.astype(np.float32) * self.FOW_DIM_FACTOR).astype(np.uint8)
            mask_3c = cv2.cvtColor(vision_mask, cv2.COLOR_GRAY2BGR)
            mask_norm = mask_3c.astype(np.float32) / 255.0
            canvas = (canvas.astype(np.float32) * mask_norm + shadowed.astype(np.float32) * (1.0 - mask_norm)).astype(np.uint8)

        for i, s in enumerate(state_list):
            current_ammo = int(s[7])
            if i in self.last_ammo:
                if current_ammo < self.last_ammo[i] and s[9] <= 0:
                    start_pos = (s[0], s[1])
                    end_pos = None
                    for h in hits:
                        if h["shooter"] == i:
                            end_pos = tuple(h["loc"])
                            break
                    if end_pos is None:
                        end_pos = self._cast_ray_to_wall(start_pos, s[4])
                    self.active_shots.append({"start_pos": start_pos, "end_pos": end_pos, "start_time": current_time})
            self.last_ammo[i] = current_ammo

        self.active_shots = [s for s in self.active_shots if current_time - s["start_time"] < self.SHOT_DURATION]

        if self.active_shots:
            overlay = canvas.copy()
            for shot in self.active_shots:
                cx, cy = int(shot["start_pos"][0] * self.scale), int(shot["start_pos"][1] * self.scale)
                ex, ey = int(shot["end_pos"][0] * self.scale), int(shot["end_pos"][1] * self.scale)
                cv2.line(overlay, (cx, cy), (ex, ey), self.SHOT_COLOR, 2)
                cv2.circle(overlay, (cx, cy), int(self.SHOT_FADE_DISTANCE * self.scale), (230, 230, 230), -1)
            cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

        for hit in hits:
            tx, ty = int(hit["loc"][0] * self.scale), int(hit["loc"][1] * self.scale)
            is_visible_hit = True
            if vision_mask is not None:
                if 0 <= tx < self.W * self.scale and 0 <= ty < self.H * self.scale:
                    if vision_mask[ty, tx] == 0: is_visible_hit = False
            if is_visible_hit:
                cv2.circle(canvas, (tx, ty), 4, (0, 0, 255), -1)
                cv2.circle(canvas, (tx, ty), 8, (0, 165, 255), 1)

        for i, s in enumerate(state_list):
            if s[6] <= 0:
                # Always draw dead bodies? Or hide them? Let's hide if not visible.
                cx, cy = int(s[0] * self.scale), int(s[1] * self.scale)
                if vision_mask is not None:
                    if 0 <= cx < self.W*self.scale and 0 <= cy < self.H*self.scale:
                        if vision_mask[cy, cx] == 0: continue
                r = int(CFG.RADIUS * self.scale)
                cv2.line(canvas, (cx-r, cy-r), (cx+r, cy+r), (150, 150, 150), 2)
                cv2.line(canvas, (cx+r, cy-r), (cx-r, cy+r), (150, 150, 150), 2)
                continue

            cx, cy = int(s[0] * self.scale), int(s[1] * self.scale)
            if vision_mask is not None and i != spectate_id:
                if 0 <= cx < self.W*self.scale and 0 <= cy < self.H*self.scale:
                    if vision_mask[cy, cx] == 0: continue

            color = (50, 50, 220) if int(s[11]) == 0 else (220, 150, 50)
            radius = int(CFG.RADIUS * self.scale)
            
            should_draw_fov = True
            if spectate_id is not None and int(s[11]) != int(state_list[spectate_id][11]):
                should_draw_fov = False
            
            if should_draw_fov:
                for ang in [s[4] - CFG.FOV / 2, s[4] + CFG.FOV / 2]:
                    lx = int(cx + math.cos(ang) * 30)
                    ly = int(cy + math.sin(ang) * 30)
                    cv2.line(canvas, (cx, cy), (lx, ly), (200, 200, 200), 1)

            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (0, 0, 0), 1)
            ex = int(cx + math.cos(s[4]) * radius * 1.5)
            ey = int(cy + math.sin(s[4]) * radius * 1.5)
            cv2.line(canvas, (cx, cy), (ex, ey), (0, 0, 0), 2)
            
            if i == spectate_id:
                cv2.circle(canvas, (cx, cy), radius + 4, (0, 255, 0), 2)
            if s[9] > 0:
                cv2.putText(canvas, "R", (cx, cy - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return canvas


g_clicked_point = None
def mouse_callback(event, x, y, flags, param):
    global g_clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        g_clicked_point = (x, y)


def run_live(args: argparse.Namespace) -> None:
    global g_clicked_point
    print(f"Initializing Simulation on {CFG.DEVICE}...")
    
    # 1. Initialize Environment
    env = CQBSimulator(n_a=args.n_agents, n_b=args.n_agents)
    obs_dict = env.reset()
    
    # 2. Detect Shapes for RL Policy Init (if needed)
    # Get shapes from the first alive agent
    first_agent_id = list(obs_dict.keys())[0]
    dummy_obs = obs_dict[first_agent_id]
    obs_shapes = {k: v.shape for k, v in dummy_obs.items()}
    # Calculate global state dim: Agent Count * 12
    state_dim = env.agents_total * 12 

    # 3. Initialize Policies based on arguments
    policies = {}
    print(f"Equipping agents with policy: {args.policy}")
    
    for i in obs_dict.keys():
        if args.policy == "rl":
            # Use the model path from arguments, or None if file doesn't exist
            model_path = args.model_path if os.path.exists(args.model_path) else None
            if model_path is None:
                print(f"Warning: Model file '{args.model_path}' not found. Using random weights.")
            policies[i] = RLInferencePolicy(obs_shapes, state_dim, model_path=model_path)
        else:
            policies[i] = NaiveCQBPolicy()
    
    renderer = None
    spectate_id = None
    if not args.headless:
        renderer = CQBRenderer(env.map.cpu().numpy())
        cv2.namedWindow("CQB Sim")
        cv2.setMouseCallback("CQB Sim", mouse_callback)

    print("Starting simulation loop... Press Q to quit.")
    pbar = tqdm(total=args.max_steps)
    steps = 0
    
    while steps < args.max_steps:
        if not args.headless and g_clicked_point is not None:
            cx, cy = g_clicked_point
            clicked_id = None
            min_d = 9999
            for i in range(env.agents_total):
                if env.state[i, 6] <= 0: continue
                ax, ay = env.state[i, 0].item(), env.state[i, 1].item()
                sx, sy = int(ax * renderer.scale), int(ay * renderer.scale)
                d = math.sqrt((cx-sx)**2 + (cy-sy)**2)
                if d < CFG.RADIUS * renderer.scale * 2.0 and d < min_d:
                    min_d = d
                    clicked_id = i
            spectate_id = clicked_id
            g_clicked_point = None

        # Get Actions
        actions = {}
        active_ids = [i for i, o in obs_dict.items() if o is not None]
        if not active_ids: break
        
        for i in active_ids: 
            # Policy inference
            actions[i] = policies[i].get_action(obs_dict[i])

        # Step Environment
        obs_dict, frame_data, _, all_dead = env.step(actions)

        if not args.headless and renderer:
            img = renderer.render_frame(frame_data["states"], frame_data["hits"], env.time, spectate_id=spectate_id)
            mode_str = f"Spectating: Agent {spectate_id}" if spectate_id is not None else "God View"
            cv2.putText(img, f"Step: {steps} | {mode_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("CQB Sim", img)
            if cv2.waitKey(33) & 0xFF == ord("q"): break

        if all_dead: break
        steps += 1
        pbar.update(1)

    pbar.close()
    if not args.headless: cv2.destroyAllWindows()
    if args.save_replay:
        with open(args.output_file, "w") as f:
            json.dump({"map_h": CFG.H, "map_w": CFG.W, "map_data": env.map.cpu().numpy().tolist(), "log": env.event_log}, f)
        print(f"Replay saved to {args.output_file}")

def render_replay(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found."); sys.exit(1)
    with open(args.input_file, "r") as f: replay_data = json.load(f)
    
    if isinstance(replay_data, list): map_data = np.zeros((CFG.H, CFG.W)); logs = replay_data
    else: map_data = np.array(replay_data["map_data"]); logs = replay_data["log"]
    
    renderer = CQBRenderer(map_data)
    spectate_id = args.spectate_id if args.spectate_id >= 0 else None
    
    print(f"Rendering {len(logs)} frames... View: {'God' if spectate_id is None else f'Agent {spectate_id}'}")
    frames = []
    for i, frame in enumerate(tqdm(logs)):
        current_time = frame.get("time", i * CFG.DT)
        img = renderer.render_frame(frame["states"], frame["hits"], current_time, spectate_id=spectate_id)
        cv2.putText(img, f"Replay | {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    mediapy.write_video(args.output_video, frames, fps=int(1.0/CFG.DT))
    print(f"Saved to {args.output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    pl = subparsers.add_parser("live")
    pl.add_argument("--n_agents", type=int, default=2) # Default reduced for clearer testing
    pl.add_argument("--max_steps", type=int, default=1000)
    pl.add_argument("--headless", action="store_true")
    pl.add_argument("--save_replay", action="store_true", default=True)
    pl.add_argument("--output_file", type=str, default="last_match.json")
    pl.add_argument("--policy", type=str, default="naive", choices=["naive", "rl"], help="Choose agent policy: 'naive' (Script) or 'rl' (Neural Network)")
    pl.add_argument("--model_path", type=str, default="cqb_policy_latest.pth", help="Path to RL policy model weights file (only used when --policy=rl)")
    
    pr = subparsers.add_parser("replay")
    pr.add_argument("--input_file", type=str, default="last_match.json")
    pr.add_argument("--output_video", type=str, default="replay.mp4")
    pr.add_argument("--spectate_id", type=int, default=-1)
    
    args = parser.parse_args()
    if args.mode == "live": run_live(args)
    elif args.mode == "replay": render_replay(args)