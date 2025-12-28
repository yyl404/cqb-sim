import torch
import numpy as np
import json
import math
import cv2


class CQBConfig:
    # --- 环境尺寸 ---
    H, W = 100, 100  # 地图尺寸
    L = 20           # 局部观测裁剪尺寸 (20x20)
    DT = 0.05        # 时间步长 (s)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 物理限制 ---
    V_MAX = 3.0      # 最大速度 (m/s)
    OMEGA_MAX = 3.0  # 最大角速度 (rad/s)
    RADIUS = 0.5     # 智能体碰撞半径

    # --- 射击与散布 ---
    SIGMA_STABLE = 0.05         # 静息散布 (rad)
    SIGMA_MAX = 0.3             # 最大散布 (rad)
    SIGMA_UNSTABLE = 0.15       # 运动惩罚散布 (rad)
    K_DECAY = 0.5               # 散布恢复速率 (rad/s)
    DELTA_SIGMA = 0.05          # 单发增加散布
    V_STABLE = 0.1              # 静息速度阈值
    W_STABLE = 0.1              # 静息转向阈值
    
    FIRE_RATE = 5.0      # 射速 (rounds/s)
    DMG_MAX = 1.5        # 最大单发伤害
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


class CQBSimulator:
    def __init__(self, n_a=2, n_b=2):
        """
        初始化 CQB 仿真环境
        
        创建室内近距离作战仿真环境，包含两个对抗阵营的智能体。环境会初始化地图、
        智能体状态和仿真参数。
        
        Args:
            n_a (int, optional): 红方阵营智能体数量，默认为 2
            n_b (int, optional): 蓝方阵营智能体数量，默认为 2
        
        初始化内容：
        1. 阵营配置：设置红方和蓝方的智能体数量
        2. 地图生成：创建二值地图（0: 空地, 1: 障碍物/墙）
        3. 状态初始化：为所有智能体分配状态张量（12维状态向量）
        4. 仿真参数：初始化时间、步数和回放日志
        
        状态向量字段（12维）：
        - [0] x: 全局X坐标
        - [1] y: 全局Y坐标
        - [2] vx: 全局X方向速度
        - [3] vy: 全局Y方向速度
        - [4] theta: 朝向角度
        - [5] omega: 角速度
        - [6] hp: 生命值
        - [7] c: 当前弹匣剩余子弹
        - [8] n: 剩余备用弹匣数量
        - [9] r_timer: 换弹倒计时
        - [10] sigma: 射击散布值
        - [11] team: 阵营标识（0: 红方, 1: 蓝方）
        """
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
        """
        基于二叉空间分割 (BSP) 生成类室内结构地图 (CQB风格)
        满足：水平垂直墙体、连通性、对称出生点
        """
        # 尝试生成直到满足连通性（BSP通常是连通的，防止意外）
        max_retries = 10
        for _ in range(max_retries):
            # 1. 初始化全墙壁
            self.grid_np = np.ones((CFG.H, CFG.W), dtype=np.uint8)
            
            # 2. BSP 分割参数
            min_node_size = 15 # 分割区域的最小尺寸
            padding = 2        # 墙壁保留厚度
            
            # 根节点
            root = {'x': 0, 'y': 0, 'w': CFG.W, 'h': CFG.H}
            nodes = [root]
            
            # 递归分割
            for _ in range(4): # 分割层数，4层大约生成 16 个区域
                new_nodes = []
                for node in nodes:
                    # 尝试分割
                    split_success = False
                    # 随机决定横切还是竖切
                    if np.random.rand() > 0.5: # 尝试竖切
                        # --- 修改点 1: 将 >= 改为 >，防止 randint(low, low) 报错 ---
                        if node['w'] > min_node_size * 2:
                            split_x = np.random.randint(min_node_size, node['w'] - min_node_size)
                            child1 = {'x': node['x'], 'y': node['y'], 'w': split_x, 'h': node['h']}
                            child2 = {'x': node['x'] + split_x, 'y': node['y'], 'w': node['w'] - split_x, 'h': node['h']}
                            # 记录走廊连接点 (中心)
                            self._create_corridor(child1, child2, 'vertical')
                            new_nodes.extend([child1, child2])
                            split_success = True
                    
                    if not split_success: # 尝试横切
                        # --- 修改点 2: 将 >= 改为 >，防止 randint(low, low) 报错 ---
                        if node['h'] > min_node_size * 2:
                            split_y = np.random.randint(min_node_size, node['h'] - min_node_size)
                            child1 = {'x': node['x'], 'y': node['y'], 'w': node['w'], 'h': split_y}
                            child2 = {'x': node['x'], 'y': node['y'] + split_y, 'w': node['w'], 'h': node['h'] - split_y}
                            self._create_corridor(child1, child2, 'horizontal')
                            new_nodes.extend([child1, child2])
                            split_success = True
                    
                    if not split_success:
                        new_nodes.append(node) # 无法分割，保留
                nodes = new_nodes

            # 3. 在每个叶子节点生成房间
            for node in nodes:
                # 随机缩小房间尺寸，形成墙壁厚度
                # 确保房间不贴边，留出最外层围墙
                # 即使 node['w'] 很小，也要保证 randint 合法
                max_padding_x = max(padding + 1, node['w'] // 4)
                # 再次防护：如果计算出的 high <= low，强制设为 low+1
                if max_padding_x <= padding: max_padding_x = padding + 1
                
                room_x = node['x'] + np.random.randint(padding, max_padding_x)
                
                max_padding_y = max(padding + 1, node['h'] // 4)
                if max_padding_y <= padding: max_padding_y = padding + 1
                
                room_y = node['y'] + np.random.randint(padding, max_padding_y)
                
                room_w = max(4, node['w'] - (room_x - node['x']) - padding)
                room_h = max(4, node['h'] - (room_y - node['y']) - padding)
                
                # 边界检查
                if room_x + room_w >= CFG.W: room_w = CFG.W - 1 - room_x
                if room_y + room_h >= CFG.H: room_h = CFG.H - 1 - room_y
                
                # 挖空房间 (0)
                self.grid_np[room_y:room_y+room_h, room_x:room_x+room_w] = 0
                
                # 4. 在房间内添加点缀障碍物 (Table/Pillar)
                if room_w > 6 and room_h > 6 and np.random.rand() > 0.3:
                    self._add_room_obstacle(room_x, room_y, room_w, room_h)

            # 5. 强制边界围墙
            self.grid_np[0:2, :] = 1
            self.grid_np[-2:, :] = 1
            self.grid_np[:, 0:2] = 1
            self.grid_np[:, -2:] = 1
            
            # 6. 连通性检查
            if self._check_connectivity():
                break
        
        # 转为 Tensor
        self.map = torch.tensor(self.grid_np, device=CFG.DEVICE, dtype=torch.float32)
        
        # 确定出生点模式
        self._determine_spawn_points()

    def _create_corridor(self, node1, node2, direction):
        """连接两个区域的走廊"""
        # 计算两个区域的中心
        c1_x, c1_y = node1['x'] + node1['w']//2, node1['y'] + node1['h']//2
        c2_x, c2_y = node2['x'] + node2['w']//2, node2['y'] + node2['h']//2
        
        # 走廊宽度随机
        thickness = np.random.randint(2, 4)
        
        x_min, x_max = min(c1_x, c2_x), max(c1_x, c2_x)
        y_min, y_max = min(c1_y, c2_y), max(c1_y, c2_y)
        
        # 在 grid 上挖出走廊 (0)
        # 注意：走廊可能会被后来的墙壁覆盖，所以我们在生成房间前先记录连接关系，
        # 或者简单的：我们在这一步直接挖空。为了保证连通，我们在生成房间后，如果走廊被堵，BSP通常能保证
        # 但这里简单的做法是：先挖走廊。
        
        # 修正：为了保证走廊不被随机缩小的房间完全切断，我们通常在最后画走廊，或者
        # 在这里画，但要保证后续房间生成时会覆盖到走廊的端点。
        # 简化版：直接画直线，连接中心。
        
        if direction == 'vertical': # 左右分割，画横向走廊
            # 稍微随机一点 y 位置，不一定要正中心
            y = c1_y
            self.grid_np[y-thickness//2 : y+thickness//2+1, x_min:x_max] = 0
        else: # 上下分割，画纵向走廊
            x = c1_x
            self.grid_np[y_min:y_max, x-thickness//2 : x+thickness//2+1] = 0

    def _add_room_obstacle(self, rx, ry, rw, rh):
        """在房间内添加障碍物"""
        obs_type = np.random.choice(['rect', 'circle'])
        
        cx, cy = rx + rw//2, ry + rh//2
        
        if obs_type == 'rect':
            # 矩形障碍物 (如桌子)
            w = np.random.randint(2, max(3, rw//3))
            h = np.random.randint(2, max(3, rh//3))
            # 随机位置
            ox = np.random.randint(rx+1, rx+rw-w)
            oy = np.random.randint(ry+1, ry+rh-h)
            self.grid_np[oy:oy+h, ox:ox+w] = 1
            
        elif obs_type == 'circle':
            # 圆形障碍物 (如柱子)
            radius = np.random.randint(1, max(2, min(rw, rh)//5))
            # 简单的圆形填充
            y_indices, x_indices = np.ogrid[:CFG.H, :CFG.W]
            mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= radius**2
            self.grid_np[mask] = 1

    def _check_connectivity(self):
        """检查是否有隔绝区域，并剔除小的死区"""
        empty_space = (self.grid_np == 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(empty_space, connectivity=4)
        
        if num_labels <= 1: return False # 全是墙
        
        # 找到最大的空地
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # 如果最大区域太小（比如生成失败），重试
        if stats[largest_label, cv2.CC_STAT_AREA] < (CFG.H * CFG.W * 0.2):
            return False

        # 填平所有非最大区域
        new_grid = np.ones_like(self.grid_np)
        new_grid[labels == largest_label] = 0
        self.grid_np = new_grid
        return True

    def _determine_spawn_points(self):
        """确定对称的出生点"""
        modes = ['diagonal_tl_br', 'diagonal_tr_bl', 'horizontal', 'vertical']
        mode = np.random.choice(modes)
        
        padding = 5
        cx, cy = CFG.W // 2, CFG.H // 2
        
        # 预设几个可能的出生中心点
        pts = {
            'tl': (padding + 5, padding + 5),          # 左上
            'tr': (CFG.W - padding - 5, padding + 5),  # 右上
            'bl': (padding + 5, CFG.H - padding - 5),  # 左下
            'br': (CFG.W - padding - 5, CFG.H - padding - 5), # 右下
            'lc': (padding + 5, cy),                   # 左中
            'rc': (CFG.W - padding - 5, cy),           # 右中
            'tc': (cx, padding + 5),                   # 上中
            'bc': (cx, CFG.H - padding - 5)            # 下中
        }
        
        if mode == 'diagonal_tl_br':
            self.spawn_center_a = pts['tl']
            self.spawn_center_b = pts['br']
        elif mode == 'diagonal_tr_bl':
            self.spawn_center_a = pts['tr']
            self.spawn_center_b = pts['bl']
        elif mode == 'horizontal':
            self.spawn_center_a = pts['lc']
            self.spawn_center_b = pts['rc']
        elif mode == 'vertical':
            self.spawn_center_a = pts['tc']
            self.spawn_center_b = pts['bc']
            
        # 确保出生点及其周围的一小块区域是空的
        self._clear_spawn_area(self.spawn_center_a)
        self._clear_spawn_area(self.spawn_center_b)

    def _clear_spawn_area(self, center, radius=3):
        cx, cy = int(center[0]), int(center[1])
        x_min = max(1, cx - radius)
        x_max = min(CFG.W - 1, cx + radius)
        y_min = max(1, cy - radius)
        y_max = min(CFG.H - 1, cy + radius)
        
        # 强制挖空
        self.grid_np[y_min:y_max, x_min:x_max] = 0
        # 更新 map tensor
        self.map = torch.tensor(self.grid_np, device=CFG.DEVICE, dtype=torch.float32)

    def reset(self):
        """重置仿真器"""
        self.time = 0.0
        self.steps = 0
        self.event_log = []
        
        # 重新生成地图 (每次都随机)
        self._generate_map()
        
        # 在固定出生点周围随机撒点
        # A 队
        for i in range(self.n_a):
            # 简单的拒绝采样，确保不生在墙里
            for _ in range(100):
                rx = np.random.uniform(-3, 3)
                ry = np.random.uniform(-3, 3)
                x = self.spawn_center_a[0] + rx
                y = self.spawn_center_a[1] + ry
                # 检查边界和墙
                ix, iy = int(x), int(y)
                if 0 < ix < CFG.W and 0 < iy < CFG.H and self.grid_np[iy, ix] == 0:
                    self.state[i, 0] = x
                    self.state[i, 1] = y
                    break
        
        # B 队 (索引从 n_a 开始)
        for i in range(self.n_a, self.agents_total):
            for _ in range(100):
                rx = np.random.uniform(-3, 3)
                ry = np.random.uniform(-3, 3)
                x = self.spawn_center_b[0] + rx
                y = self.spawn_center_b[1] + ry
                ix, iy = int(x), int(y)
                if 0 < ix < CFG.W and 0 < iy < CFG.H and self.grid_np[iy, ix] == 0:
                    self.state[i, 0] = x
                    self.state[i, 1] = y
                    break

        # 初始化其他状态 (theta, hp 等)
        for i in range(self.agents_total):
            is_a = i < self.n_a
            self.state[i, 2] = 0 # vx
            self.state[i, 3] = 0 # vy
            # 让 A 队朝向 B 队大致方向，B 队反之，或者简单地 A 朝右(0) B 朝左(pi)
            self.state[i, 4] = 0 if is_a else np.pi 
            self.state[i, 5] = 0 
            self.state[i, 6] = 1.0 
            self.state[i, 7] = CFG.MAG_SIZE 
            self.state[i, 8] = CFG.MAX_MAGS 
            self.state[i, 9] = 0 
            self.state[i, 10] = CFG.SIGMA_STABLE 
            self.state[i, 11] = 0 if is_a else 1 

        return self._get_observations()

    def step(self, actions_dict, reward_fn=None):
        """
        执行一个仿真时间步
        
        这是强化学习环境的标准 step 函数，接收所有智能体的动作，执行物理更新、
        战斗解析和状态更新，然后返回观测、奖励和终止标志。
        
        Args:
            actions_dict (Dict[int, np.ndarray]): 智能体动作字典
                - 键：智能体索引
                - 值：动作数组，形状为 (5,)，包含 [v_surge, v_sway, v_omega, fire, reload]
                    - v_surge: 纵向速度 [-1, 1]
                    - v_sway: 横向速度 [-1, 1]
                    - v_omega: 角速度 [-1, 1]
                    - fire: 射击指令 [0, 1]，> 0.5 表示开火
                    - reload: 换弹指令 [0, 1]，> 0.5 表示换弹
            reward_fn (Callable, optional): 可选的奖励计算函数
                - 如果提供，函数将接收 (current_frame_data, vanquished) 作为参数
                - 应返回 Dict[int, float]，键为智能体索引，值为奖励值
                - 如果提供，返回值中将包含计算得到的奖励
        
        Returns:
            Tuple: 返回值取决于是否提供 reward_fn
                - 如果 reward_fn 为 None:
                    (obs, current_frame_data, vanquished, all_vanquished)
                - 如果 reward_fn 不为 None:
                    (obs, current_frame_data, rewards, vanquished, all_vanquished)
                其中：
                - obs: 观测字典，键为智能体索引，值为包含 'self', 'spatial', 'team', 'enemy' 的字典
                - current_frame_data: 当前帧数据字典，包含 'time', 'states', 'hits'，用于奖励计算
                - rewards: 奖励字典（仅当 reward_fn 提供时返回），键为智能体索引，值为奖励值
                - vanquished: 阵亡标志字典，键为智能体索引，值为是否阵亡
                - all_vanquished: 全局终止标志，True 表示一方阵营全部阵亡
        
        执行流程：
        1. 将动作字典转换为 Tensor 格式
        2. 更新状态（将动作转换为速度并更新散布、换弹状态）
        3. 物理更新（根据速度更新位置和角度）
        4. 战斗解析（处理射击和伤害计算）
        5. 记录当前帧数据到回放日志
        6. 更新时间步和步数
        7. 生成观测和终止标志
        8. 如果提供 reward_fn，计算奖励并返回
        """
        # 将动作转为Tensor
        actions = torch.zeros((self.agents_total, 5), device=CFG.DEVICE)
        for i in range(self.agents_total):
            if i in actions_dict:
                actions[i] = torch.tensor(actions_dict[i], device=CFG.DEVICE)

        self._update_status(actions)  # 先更新状态（将动作转换为速度）
        self._update_physics()        # 再使用 state 中的速度进行物理更新
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
        vanquished = {}
        for i in range(self.agents_total):
            vanquished[i] = self.state[i, 6] <= 0

        all_vanquished = all([vanquished[i] for i in range(self.n_a)]) or all([vanquished[i] for i in range(self.n_a, self.agents_total)])
        
        # 如果提供了奖励函数，计算奖励并返回
        if reward_fn is not None:
            rewards = reward_fn(current_frame_data, vanquished)
            return obs, current_frame_data, rewards, vanquished, all_vanquished
        else:
            return obs, current_frame_data, vanquished, all_vanquished

    def _update_physics(self):
        """
        物理更新：根据当前状态中的速度值更新智能体的位置和角度
        
        该函数使用 state 中已更新的速度值（vx, vy, omega）进行物理模拟，
        包括位置更新、角度更新和碰撞检测。速度值应在调用此函数前通过
        _update_status 函数从动作中计算并更新到 state 中。
        
        物理更新流程：
        1. 根据当前速度更新位置和角度
        2. 检测与地图障碍物的碰撞
        3. 更新存活智能体的位置和角度（碰撞的智能体位置不变）
        """
        # 使用 state 中的速度值进行物理更新
        vx = self.state[:, 2]
        vy = self.state[:, 3]
        omega = self.state[:, 5]
        theta = self.state[:, 4]
        
        # 位置更新
        x_new = self.state[:, 0] + vx * CFG.DT
        y_new = self.state[:, 1] + vy * CFG.DT
        theta_new = (theta + omega * CFG.DT) % (2 * np.pi)
        
        # 碰撞检测：考虑智能体的碰撞半径
        # 检查智能体中心周围半径范围内的网格点是否有障碍物
        collided = torch.zeros(self.agents_total, dtype=torch.bool, device=CFG.DEVICE)
        radius = CFG.RADIUS
        
        # 对每个智能体进行碰撞检测
        for i in range(self.agents_total):
            x, y = x_new[i].item(), y_new[i].item()
            
            # 计算需要检查的网格范围（中心 ± 半径）
            x_min = max(0, int(x - radius))
            x_max = min(CFG.W - 1, int(x + radius) + 1)
            y_min = max(0, int(y - radius))
            y_max = min(CFG.H - 1, int(y + radius) + 1)
            
            # 检查范围内是否有障碍物，且距离中心小于等于半径
            for gx in range(x_min, x_max):
                for gy in range(y_min, y_max):
                    # 计算网格中心到智能体中心的距离
                    grid_center_x = gx + 0.5
                    grid_center_y = gy + 0.5
                    dist = math.sqrt((grid_center_x - x)**2 + (grid_center_y - y)**2)
                    
                    # 如果网格是障碍物且距离小于等于半径，则发生碰撞
                    if self.map[gy, gx] > 0.5 and dist <= radius:
                        collided[i] = True
                        break
                if collided[i]:
                    break
        
        alive = self.state[:, 6] > 0
        update_mask = alive & (~collided)
        
        self.state[update_mask, 0] = x_new[update_mask]
        self.state[update_mask, 1] = y_new[update_mask]
        self.state[alive, 4] = theta_new[alive]

    def _update_status(self, actions):
        """
        状态更新：将动作转换为物理量并更新智能体的状态
        
        该函数处理智能体的动作输入，将其转换为物理速度值并更新到 state 中，
        同时更新散布值和换弹状态。这是物理更新的前置步骤，应在 _update_physics
        之前调用。
        
        Args:
            actions (torch.Tensor): 所有智能体的动作张量，形状为 (n_agents, 5)
                - actions[:, 0]: 纵向速度指令 [-1, 1]
                - actions[:, 1]: 横向速度指令 [-1, 1]
                - actions[:, 2]: 角速度指令 [-1, 1]
                - actions[:, 3]: 射击指令 [0, 1]
                - actions[:, 4]: 换弹指令 [0, 1]
        
        更新内容：
        1. 动作转换：将机体坐标系下的速度指令转换为全局坐标系速度并更新到 state
        2. 散布更新：根据移动状态更新射击散布值
        3. 换弹状态：处理换弹逻辑和倒计时
        """
        # 1. 处理动作，将动作转换为速度并更新到 state
        v_surge = actions[:, 0] # 所有智能体的纵向速度 [-1, 1]
        v_sway = actions[:, 1] # 所有智能体的横向速度 [-1, 1]
        v_omega = actions[:, 2] # 所有智能体的角速度 [-1, 1]
        
        theta = self.state[:, 4] # 所有智能体的朝向角度
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # 将智能体坐标系下的速度分量投影到全局坐标系
        vx_new = cos_t * v_surge - sin_t * v_sway
        vy_new = sin_t * v_surge + cos_t * v_sway
        
        # 裁剪模长到 [0, 1] 范围
        v_norm = torch.sqrt(vx_new**2 + vy_new**2 + 1e-6)
        v_norm_clamped = torch.clamp(v_norm, max=1.0)
        
        # 归一化方向并乘以裁剪后的模长，映射到 [0, V_MAX]
        vx_new = CFG.V_MAX * vx_new * v_norm_clamped / (v_norm + 1e-6)
        vy_new = CFG.V_MAX * vy_new * v_norm_clamped / (v_norm + 1e-6)
        
        # 将角速度动作从 [-1, 1] 映射到 [-OMEGA_MAX, OMEGA_MAX] 并裁剪
        omega_new = v_omega * CFG.OMEGA_MAX
        omega_new = torch.clamp(omega_new, -CFG.OMEGA_MAX, CFG.OMEGA_MAX)
        
        # 更新 state 中的速度值
        alive = self.state[:, 6] > 0
        self.state[alive, 2] = vx_new[alive]
        self.state[alive, 3] = vy_new[alive]
        self.state[alive, 5] = omega_new[alive]
        
        # 2. 更新散布值
        v_sq = self.state[:, 2]**2 + self.state[:, 3]**2
        omega_abs = torch.abs(self.state[:, 5])
        
        is_unstable = (v_sq > CFG.V_STABLE**2) | (omega_abs > CFG.W_STABLE)
        target_sigma = torch.where(is_unstable, torch.tensor(CFG.SIGMA_UNSTABLE, device=CFG.DEVICE), torch.tensor(CFG.SIGMA_STABLE, device=CFG.DEVICE))
        
        current_sigma = self.state[:, 10]
        
        recover_mask = current_sigma > target_sigma
        current_sigma[recover_mask] -= CFG.K_DECAY * CFG.DT
        
        jump_mask = current_sigma < target_sigma
        current_sigma[jump_mask] = target_sigma[jump_mask]
        
        self.state[:, 10] = torch.clamp(current_sigma, CFG.SIGMA_STABLE, CFG.SIGMA_MAX)

        # 3. 更新换弹状态
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
        
    def _detect_ray_collision(self, ps, pt):
        """
        射线检测：检查从起点到终点的射线是否被障碍物阻挡
        
        使用 Bresenham 直线算法精确遍历射线经过的所有网格单元，比采样方法更精确且更快。
        该算法直接遍历射线经过的每个网格单元，避免浮点数采样，确保不遗漏任何障碍物。
        
        Args:
            ps (torch.Tensor): 射线起点坐标，形状为 (2,)，表示 (x, y) 位置
            pt (torch.Tensor): 射线终点坐标，形状为 (2,)，表示 (x, y) 位置
        
        Returns:
            bool: 如果射线被障碍物阻挡或超出边界返回 True，否则返回 False
        """
        # 转换为整数网格坐标
        x0 = int(ps[0].item())
        y0 = int(ps[1].item())
        x1 = int(pt[0].item())
        y1 = int(pt[1].item())
        
        # 距离为0，视为同一点
        if x0 == x1 and y0 == y1:
            return False
        
        # 使用 Bresenham 直线算法遍历所有经过的网格单元
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        first_step = True  # 标记是否为第一步（起点）
        
        # 遍历从起点到终点的所有网格单元
        while True:
            # 跳过起点（避免检查射击者自身位置）
            if not first_step:
                # 边界检查
                if x < 0 or x >= CFG.W or y < 0 or y >= CFG.H:
                    return True  # 超出边界视为阻挡
                
                # 检查是否为障碍物
                if self.map[y, x] > 0.5:
                    return True  # 被障碍物阻挡
            
            # 到达终点
            if x == x1 and y == y1:
                break
            
            first_step = False
            
            # Bresenham 算法的下一步
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return False  # 路径畅通
    
    def _resolve_combat(self, actions):
        """
        战斗解析：处理所有智能体的射击动作，计算命中判定和伤害
        
        该函数处理每个智能体的射击指令，包括射击条件检查、弹道模拟、命中判定和伤害计算。
        对于每个射击者，系统会：
        1. 检查射击条件（存活、有弹药、不在换弹、满足射速限制）
        2. 根据当前散布值生成带随机误差的射击方向
        3. 检测所有可能命中的目标（考虑横向距离、障碍物阻挡）
        4. 选择最近的命中目标并计算伤害
        5. 更新射击者的弹药、散布值和目标的生命值
        
        Args:
            actions (torch.Tensor): 所有智能体的动作张量，形状为 (n_agents, 5)
                - actions[:, 3]: 射击指令，> 0.5 表示开火
                - actions[:, 4]: 换弹指令，> 0.5 表示换弹
        
        Returns:
            List[Dict]: 命中事件日志列表，每个元素包含：
                - 'shooter': 射击者索引
                - 'target': 被命中目标索引
                - 'damage': 造成的伤害值
                - 'loc': 命中位置坐标 [x, y]
        """
        hits_log = []
        fire_cmd = actions[:, 3] > 0.5
        # 射击条件检查：
        # - state[:, 6] (h_i): 当前生命值，> 0 表示存活
        # - state[:, 7] (c_i): 当前弹匣剩余子弹，> 0 表示有弹药
        # - state[:, 9] (r_timer): 换弹倒计时，<= 0 表示不在换弹状态
        # - 射速限制：距离上次射击时间 >= 1/FIRE_RATE
        can_fire = (self.state[:, 6] > 0) & \
                   (self.state[:, 7] > 0) & \
                   (self.state[:, 9] <= 0) & \
                   ((self.time - self.last_shot_time) >= (1.0/CFG.FIRE_RATE))
        
        shooters = torch.nonzero(fire_cmd & can_fire).squeeze()
        if shooters.numel() == 0:
            return hits_log
        if shooters.dim() == 0: shooters = shooters.unsqueeze(0)

        for s_idx in shooters:
            self.state[s_idx, 7] -= 1 # 余弹量减少1
            self.last_shot_time[s_idx] = self.time
            
            noise = np.random.normal(0, self.state[s_idx, 10].item()) # 随机生成射击散步误差
            shoot_angle = self.state[s_idx, 4].item() + noise # 射击散步误差叠加射手的朝向为最终射击方向

            self.state[s_idx, 10] = min(self.state[s_idx, 10] + CFG.DELTA_SIGMA, CFG.SIGMA_MAX) # 增加下一次开火的射击散步方差
            
            ps = self.state[s_idx, 0:2] # 射手位置
            direction = torch.tensor([math.cos(shoot_angle), math.sin(shoot_angle)], device=CFG.DEVICE) # 将射击方向角转换为方向向量
            
            targets = torch.arange(self.agents_total, device=CFG.DEVICE)
            targets = targets[targets != s_idx] # 排除射手本身
            
            min_dist = 9999
            hit_target = -1
            hit_lateral_distance = 0
            
            for t_idx in targets:
                if self.state[t_idx, 6] <= 0: continue # 跳过已经阵亡的目标
                
                pt = self.state[t_idx, 0:2] # 目标的位置
                v_st = pt - ps # 目标与射手的相对位置
                proj_t = torch.dot(v_st, direction) # 目标在射手的射击方向上的投影距离
                
                if proj_t < 0: continue # 跳过位于射手后方的目标
                
                lateral_offset = v_st - proj_t * direction  # 垂直于射击方向的横向偏移向量
                lateral_distance = torch.norm(lateral_offset)  # 目标到射击路径的横向距离
                
                if lateral_distance < CFG.HIT_RADIUS: # 如果横向距离小于命中半径，则继续判断其余命中判定条件
                    # 检查是否有障碍物阻挡
                    if self._detect_ray_collision(ps, pt):
                        continue  # 被障碍物阻挡，跳过此目标
                    
                    if proj_t < min_dist: # 记录下最靠近射手的命中对象
                        min_dist = proj_t
                        hit_target = t_idx
                        hit_lateral_distance = lateral_distance
            
            if hit_target != -1:
                damage = CFG.DMG_MAX * math.exp(-(hit_lateral_distance.item()**2) / (2 * CFG.DMG_WIDTH**2)) # 根据命中位置与目标中心的距离计算伤害值
                self.state[hit_target, 6] = max(0, self.state[hit_target, 6] - damage) # 扣除目标的血量
                hits_log.append({ # 记录下此次事件
                    'shooter': s_idx.item(),
                    'target': hit_target.item(),
                    'damage': damage,
                    'loc': self.state[hit_target, 0:2].cpu().numpy().tolist()
                })
                
        return hits_log

    def _get_observations(self):
        """
        获取所有智能体的观测信息
        
        为每个存活的智能体生成观测向量，包括本体感知、空间感知、队友信息和敌方信息。
        观测采用全局轴对齐参考系，便于神经网络处理。敌方信息的获取受可视性约束，
        采用共享视野机制：如果任意一个同阵营友军能看见敌人，则所有友军都能看见。
        
        Returns:
            Dict[int, Dict[str, torch.Tensor]]: 观测字典，键为智能体索引，值为包含以下字段的字典：
                - 'self' (torch.Tensor): 本体感知，形状为 (9,)，包含 (theta, vx, vy, omega, hp, c, n, r, sigma)
                - 'spatial' (torch.Tensor): 空间感知，形状为 (L*L,)，局部地图的展平向量
                - 'team' (torch.Tensor): 队友信息，形状为 (n_teammates, 10)，每行包含一个队友的状态
                - 'enemy' (torch.Tensor): 敌方信息，形状为 (n_enemies, 8)，每行包含一个敌人的状态（不可见时为全零向量）
                如果智能体已阵亡，则对应的值为 None
        """
        obs_dict = {}
        for i in range(self.agents_total):
            if self.state[i, 6] <= 0: # 已经阵亡的目标是不可见的
                obs_dict[i] = None
                continue
                
            o_self = torch.cat([
                self.state[i, 4:5],   # theta: 朝向角
                self.state[i, 2:4],    # vx, vy: 线速度
                self.state[i, 5:6],    # omega: 角速度
                self.state[i, 6:11]    # hp, c, n, r_timer, sigma: 生命值、弹药、备弹、换弹倒计时、散布
            ])
            # 将换弹倒计时 r_timer 转换为二值标志 r (1.0 表示正在换弹，0.0 表示未换弹)
            o_self[7] = 1.0 if o_self[7] > 0 else 0.0 
            
            cx, cy = int(self.state[i, 0]), int(self.state[i, 1])
            half_L = CFG.L // 2
            padded_map = torch.nn.functional.pad(self.map, (half_L, half_L, half_L, half_L), value=1)
            x_start = cx
            y_start = cy
            local_grid = padded_map[y_start:y_start+CFG.L, x_start:x_start+CFG.L]
            o_spatial = local_grid.flatten() # 局部坐标系下的地图
            
            # 对其它智能体进行观测（敌方和友方）
            team_obs = []
            enemy_obs = []
            my_team = self.state[i, 11]
            my_pos = self.state[i, 0:2]
            my_theta = self.state[i, 4]
            
            for j in range(self.agents_total):
                if i == j: continue # 跳过本身
                
                rel_pos = self.state[j, 0:2] - my_pos 
                target_team = self.state[j, 11]
                
                if target_team == my_team:
                    feat = torch.cat([
                        rel_pos,            # rel_x, rel_y: 相对位置
                        self.state[j, 4:5], # theta: 朝向角
                        self.state[j, 2:4], # vx, vy: 线速度
                        self.state[j, 5:6], # omega: 角速度
                        self.state[j, 6:10]  # hp, c, n, r_timer: 生命值、弹药、备弹、换弹倒计时
                    ])
                    feat[9] = 1.0 if feat[9] > 0 else 0.0
                    team_obs.append(feat)
                else:
                    # 敌方：检查是否可见（共享视野机制）
                    enemy_pos = self.state[j, 0:2]
                    
                    # 1. 检查所有同阵营友军是否能看见该敌人（共享视野）
                    is_visible_by_any_teammate = False
                    half_L = CFG.L // 2
                    
                    # 遍历所有同阵营友军
                    for teammate_idx in range(self.agents_total):
                        if teammate_idx == j: continue  # 跳过敌人本身
                        if self.state[teammate_idx, 11] != my_team: continue  # 只检查同阵营
                        if self.state[teammate_idx, 6] <= 0: continue  # 跳过已阵亡的友军
                        
                        teammate_pos = self.state[teammate_idx, 0:2]
                        teammate_theta = self.state[teammate_idx, 4]
                        vec_to_enemy = enemy_pos - teammate_pos
                        
                        # 检查敌人是否在友军的局部地图范围内（以友军为中心，边长为 L 的正方形区域）
                        rel_x = enemy_pos[0] - teammate_pos[0]
                        rel_y = enemy_pos[1] - teammate_pos[1]
                        in_local_map = (abs(rel_x) <= half_L) & (abs(rel_y) <= half_L)
                        
                        if not in_local_map:
                            continue
                        
                        # 检查是否在视锥范围内
                        # atan2 值域: [-π, π]，teammate_theta 值域: [0, 2π)
                        angle_to_enemy = torch.atan2(vec_to_enemy[1], vec_to_enemy[0])
                        angle_diff = torch.abs(angle_to_enemy - teammate_theta)
                        # 由于角度的周期性，需要计算最小角度差（考虑圆周上的最短路径）
                        # 例如：10° 和 350° 的差是 20° 而不是 340°
                        angle_diff = torch.min(angle_diff, 2*np.pi - angle_diff)
                        
                        if angle_diff < (CFG.FOV / 2):
                            # 检查是否有障碍物阻挡
                            if not self._detect_ray_collision(teammate_pos, enemy_pos):
                                is_visible_by_any_teammate = True
                                break
                    
                    # 3. 如果任意友军可见，则当前智能体也能看见（共享视野）
                    if is_visible_by_any_teammate:
                        feat = torch.cat([
                            rel_pos,            # rel_x, rel_y: 相对位置
                            self.state[j, 4:5], # theta: 朝向角
                            self.state[j, 2:4], # vx, vy: 线速度
                            self.state[j, 5:6], # omega: 角速度
                            self.state[j, 6:7], # hp: 生命值
                            self.state[j, 9:10] # r_timer: 换弹倒计时
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
    
    def save_replay(self, filepath):
        """
        保存回放日志到文件
        
        将仿真过程中记录的所有事件日志（包括每帧的状态、命中信息等）
        保存为 JSON 格式文件，用于后续的回放和可视化。
        
        Args:
            filepath (str): 保存文件的路径，应为 .json 格式
        
        注意：
            event_log 在 step 函数中持续累积，包含每帧的完整状态信息。
        """
        with open(filepath, 'w') as f:
            json.dump(self.event_log, f)
        print(f"Replay saved to {filepath}")