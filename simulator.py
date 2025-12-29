import torch
import numpy as np
import json
import math
import cv2


class CQBConfig:
    # --- 环境尺寸 ---
    H, W = 100, 100  # 地图尺寸
    L = 200          # 局部观测裁剪尺寸
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
        全新逻辑：网格化 -> 区域合并 -> 连通性生长 -> 渲染
        """
        # 1. 构造不均匀网格 (Lattice Generation)
        # x_coords, y_coords 存储的是网格线的像素坐标
        x_coords = self._generate_grid_lines(CFG.W)
        y_coords = self._generate_grid_lines(CFG.H)
        
        rows = len(y_coords) - 1
        cols = len(x_coords) - 1
        
        # 2. 初始化墙壁数据结构
        # v_walls[r][c] 代表格子(r,c)左侧的墙 (最后一列多一个边界墙)
        # h_walls[r][c] 代表格子(r,c)上侧的墙 (最后一行多一个边界墙)
        # 状态: 1=Wall, 0=Merged(No Wall), 2=Door
        self.v_walls = np.ones((rows, cols + 1), dtype=np.int8)
        self.h_walls = np.ones((rows + 1, cols), dtype=np.int8)
        
        # 辅助：记录每个格子属于哪个“房间ID” (用于合并逻辑)
        # 初始时，每个格子都是独立的房间
        self.cell_group_id = np.arange(rows * cols).reshape(rows, cols)
        
        # 3. 随机合并房间 (Merge Step)
        # 尝试多次合并，生成多个不规则大厅
        num_merges = int(rows * cols * 0.2) # 尝试合并的次数
        for _ in range(num_merges):
            self._merge_random_cluster(rows, cols)
            
        # 4. 确定出生点 (大致对称)
        # 随机选一个非边缘的格子作为A
        spawn_r_a = np.random.randint(1, rows - 1)
        spawn_c_a = np.random.randint(1, cols - 1)
        # 对称点作为B
        spawn_r_b = rows - 1 - spawn_r_a
        spawn_c_b = cols - 1 - spawn_c_a
        
        # 记录出生点矩形范围 (用于后续逻辑，虽然题目说忽略要求，但我们需要坐标来放人)
        # 获取格子实际像素坐标
        self.spawn_rect_a = self._get_cell_rect(spawn_r_a, spawn_c_a, x_coords, y_coords)
        self.spawn_rect_b = self._get_cell_rect(spawn_r_b, spawn_c_b, x_coords, y_coords)

        # 5. 连通性生长 (Connectivity Step)
        # 从 A 出发，打通去往所有房间的路
        self._ensure_connectivity_from_spawn(spawn_r_a, spawn_c_a, rows, cols)
        
        # 6. 渲染到 Grid (Rendering)
        self.grid_np = np.zeros((CFG.H, CFG.W), dtype=np.uint8)
        self._render_grid(x_coords, y_coords)
        
        # 7. 强制外墙封闭 (Boundary)
        self.grid_np[0, :] = 1
        self.grid_np[-1, :] = 1
        self.grid_np[:, 0] = 1
        self.grid_np[:, -1] = 1

        # 转为 Tensor
        self.map = torch.tensor(self.grid_np, device=CFG.DEVICE, dtype=torch.float32)

    def _generate_grid_lines(self, length):
        """在纵向或横向生成随机间隔的分割线"""
        coords = [0]
        while coords[-1] < length:
            # 间隔 10 - 20
            step = np.random.randint(10, 21)
            next_pos = coords[-1] + step
            if next_pos >= length - 10: # 如果剩余空间太小，直接吸附到边界
                coords.append(length)
                break
            coords.append(next_pos)
        return coords

    def _get_cell_rect(self, r, c, x_coords, y_coords):
        """获取网格单元的像素范围 (x, y, w, h)"""
        x = x_coords[c]
        y = y_coords[r]
        w = x_coords[c+1] - x
        h = y_coords[r+1] - y
        return (x, y, w, h)

    def _merge_random_cluster(self, rows, cols):
        """随机选择一个点，向外合并最多5个格子形成不规则房间"""
        # 随机种子点
        start_r = np.random.randint(0, rows)
        start_c = np.random.randint(0, cols)
        target_group = self.cell_group_id[start_r, start_c]
        
        current_cluster = [(start_r, start_c)]
        max_size = np.random.randint(2, 6) # 大小 2-5
        
        # 简单的广度/随机优先搜索来吸纳邻居
        candidates = []
        self._add_neighbors(start_r, start_c, rows, cols, candidates)
        
        while len(current_cluster) < max_size and candidates:
            # 随机选一个邻居
            idx = np.random.randint(len(candidates))
            nr, nc, wall_type = candidates.pop(idx)
            
            # 如果这个邻居还没被归类到当前组 (避免重复合并)
            # 注意：这里我们允许合并不同的Group，从而让小房间变成大房间
            if self.cell_group_id[nr, nc] != target_group:
                # 1. 拆墙
                if wall_type == 'h': # (nr, nc) 在 (r, c) 下方或上方，看坐标
                    # 确定墙的坐标。墙的索引通常取较大的那个（下侧墙或右侧墙）
                    wr = max(nr, current_cluster[-1][0]) # 简化逻辑，重新判断方向
                    if nr > current_cluster[-1][0]: self.h_walls[nr][nc] = 0 # 下邻居，墙在nr
                    else: self.h_walls[nr+1][nc] = 0 # 上邻居，墙在nr+1 (即current的上方)
                    # 修正逻辑：更严谨的判断
                    r_old, c_old = self._find_adj_cell(nr, nc, current_cluster)
                    if r_old < nr: self.h_walls[nr][nc] = 0 # Down
                    else: self.h_walls[r_old][nc] = 0 # Up
                else: # Vertical
                    r_old, c_old = self._find_adj_cell(nr, nc, current_cluster)
                    if c_old < nc: self.v_walls[nr][nc] = 0 # Right
                    else: self.v_walls[nr][c_old] = 0 # Left

                # 2. 统一 Group ID
                # 将该邻居原来的 group id 全部替换为 target_group
                old_id = self.cell_group_id[nr, nc]
                self.cell_group_id[self.cell_group_id == old_id] = target_group
                
                current_cluster.append((nr, nc))
                self._add_neighbors(nr, nc, rows, cols, candidates)

    def _find_adj_cell(self, r, c, cluster):
        """在cluster中找到与(r,c)相邻的那个格子用于确定墙的位置"""
        for pr, pc in cluster:
            if abs(pr - r) + abs(pc - c) == 1:
                return pr, pc
        return r, c # Should not happen

    def _add_neighbors(self, r, c, rows, cols, list_ref):
        # 上
        if r > 0: list_ref.append((r-1, c, 'h'))
        # 下
        if r < rows - 1: list_ref.append((r+1, c, 'h'))
        # 左
        if c > 0: list_ref.append((r, c-1, 'v'))
        # 右
        if c < cols - 1: list_ref.append((r, c+1, 'v'))

    def _ensure_connectivity_from_spawn(self, start_r, start_c, rows, cols):
        """
        从出生点所在的 Group 开始，使用 Prim 算法思想向外扩展，
        直到所有格子都被访问（连通）。
        """
        # 1. 找到出生点所在的 Group ID
        start_group = self.cell_group_id[start_r, start_c]
        
        # 2. 初始化已访问的 Group 集合
        visited_groups = {start_group}
        
        # 3. 前线墙壁列表：(r_from, c_from, r_to, c_to, wall_type, wall_r, wall_c)
        # 存储所有连接 "已访问区域" 和 "未访问区域" 的实体墙壁
        frontier_walls = []
        
        # 将初始 Group 的所有对外墙壁加入前线
        # 遍历全图太慢，我们先找到属于 start_group 的所有格子
        # (简单起见，这里做一个全图扫描初始化，对于100x100来说很快)
        # 或者更优：直接用 BFS 收集
        self._add_group_walls_to_frontier(start_group, rows, cols, frontier_walls, visited_groups)
        
        total_groups = len(np.unique(self.cell_group_id))
        
        while len(visited_groups) < total_groups and frontier_walls:
            # 随机选一面墙
            idx = np.random.randint(len(frontier_walls))
            w_info = frontier_walls.pop(idx) # 移除
            rf, cf, rt, ct, wtype, wr, wc = w_info
            
            target_group = self.cell_group_id[rt, ct]
            
            if target_group in visited_groups:
                continue # 墙对面已经被连通了，跳过
            
            # --- 打通这面墙 (生成门) ---
            if wtype == 'h':
                self.h_walls[wr][wc] = 2 # 2 代表 Door
            else:
                self.v_walls[wr][wc] = 2
                
            # --- 标记新 Group 为已访问 ---
            visited_groups.add(target_group)
            
            # --- 将新 Group 的对外墙壁加入前线 ---
            self._add_group_walls_to_frontier(target_group, rows, cols, frontier_walls, visited_groups)

    def _add_group_walls_to_frontier(self, group_id, rows, cols, frontier, visited_set):
        """找到属于 group_id 的所有格子的对外墙壁"""
        # 找到所有属于该组的格子坐标
        # mask = (self.cell_group_id == group_id)
        # rs, cs = np.where(mask)
        # 上面方法在循环里可能慢，这里用遍历优化：
        # 由于我们是逐步扩展，只需要扫描 target_group 包含的格子。
        # 这里为了代码简洁，直接遍历整个 group_id 的格子列表
        
        rs, cs = np.where(self.cell_group_id == group_id)
        for i in range(len(rs)):
            r, c = rs[i], cs[i]
            
            # 检查四个方向
            # 上
            if r > 0:
                neighbor_group = self.cell_group_id[r-1, c]
                if neighbor_group not in visited_set:
                    # 只有当这里是实墙时才添加 (如果是0说明内部合并了)
                    if self.h_walls[r][c] == 1:
                        frontier.append((r, c, r-1, c, 'h', r, c))
            # 下
            if r < rows - 1:
                neighbor_group = self.cell_group_id[r+1, c]
                if neighbor_group not in visited_set:
                    if self.h_walls[r+1][c] == 1:
                        frontier.append((r, c, r+1, c, 'h', r+1, c))
            # 左
            if c > 0:
                neighbor_group = self.cell_group_id[r, c-1]
                if neighbor_group not in visited_set:
                    if self.v_walls[r][c] == 1:
                        frontier.append((r, c, r, c-1, 'v', r, c))
            # 右
            if c < cols - 1:
                neighbor_group = self.cell_group_id[r, c+1]
                if neighbor_group not in visited_set:
                    if self.v_walls[r][c+1] == 1:
                        frontier.append((r, c, r, c+1, 'v', r, c+1))

    def _render_grid(self, x_coords, y_coords):
        """将逻辑墙壁渲染到 grid_np"""
        rows = len(y_coords) - 1
        cols = len(x_coords) - 1
        
        # 渲染横向墙 (h_walls)
        # h_walls[r][c] 对应 y_coords[r] 这条线，从 x_coords[c] 到 x_coords[c+1]
        for r in range(rows + 1):
            y = y_coords[r]
            # 防止越界
            if y >= CFG.H: y = CFG.H - 1
            
            for c in range(cols):
                status = self.h_walls[r][c]
                x_start = x_coords[c]
                x_end = x_coords[c+1]
                
                if status == 1: # Wall
                    self.grid_np[y, x_start:x_end] = 1
                elif status == 2: # Door
                    self.grid_np[y, x_start:x_end] = 1 # 先画墙
                    # 挖门
                    door_size = np.random.randint(3, 6) # 3-5
                    segment_len = x_end - x_start
                    if segment_len > door_size:
                        door_start = x_start + np.random.randint(1, segment_len - door_size)
                        self.grid_np[y, door_start : door_start+door_size] = 0
                    else:
                        # 墙太短直接全挖了
                        self.grid_np[y, x_start:x_end] = 0

        # 渲染纵向墙 (v_walls)
        for r in range(rows):
            y_start = y_coords[r]
            y_end = y_coords[r+1]
            
            for c in range(cols + 1):
                x = x_coords[c]
                if x >= CFG.W: x = CFG.W - 1
                
                status = self.v_walls[r][c]
                
                if status == 1: # Wall
                    self.grid_np[y_start:y_end, x] = 1
                elif status == 2: # Door
                    self.grid_np[y_start:y_end, x] = 1
                    door_size = np.random.randint(3, 6)
                    segment_len = y_end - y_start
                    if segment_len > door_size:
                        door_start = y_start + np.random.randint(1, segment_len - door_size)
                        self.grid_np[door_start : door_start+door_size, x] = 0
                    else:
                        self.grid_np[y_start:y_end, x] = 0

    def reset(self):
        """
        重置仿真器 (适配网格迷宫生成)
        """
        self.time = 0.0
        self.steps = 0
        self.event_log = []
        
        # 1. 生成新地图
        # 这一步会更新 self.map, self.v_walls, self.h_walls, self.spawn_rect_a/b
        self._generate_map()
        
        # 2. 在网格房间内生成智能体
        # 此时 self.spawn_rect_a 格式为 (x, y, w, h)
        # 我们需要在矩形内部保留一点边距(padding)，避免贴墙出生
        padding = 1.5 
        
        # --- 生成 A 队 ---
        ax, ay, aw, ah = self.spawn_rect_a
        for i in range(self.n_a):
            # 在矩形范围内随机 (x ~ x+w, y ~ y+h)
            # 确保房间够大，如果房间极小(例如10x10)，padding后还有空间
            safe_w = max(1.0, aw - 2 * padding)
            safe_h = max(1.0, ah - 2 * padding)
            
            self.state[i, 0] = ax + padding + np.random.uniform(0, safe_w)
            self.state[i, 1] = ay + padding + np.random.uniform(0, safe_h)
        
        # --- 生成 B 队 ---
        bx, by, bw, bh = self.spawn_rect_b
        for i in range(self.n_a, self.agents_total):
            safe_w = max(1.0, bw - 2 * padding)
            safe_h = max(1.0, bh - 2 * padding)
            
            self.state[i, 0] = bx + padding + np.random.uniform(0, safe_w)
            self.state[i, 1] = by + padding + np.random.uniform(0, safe_h)

        # 3. 初始化物理和战斗状态
        cx, cy = CFG.W / 2, CFG.H / 2
        for i in range(self.agents_total):
            is_a = i < self.n_a
            
            # 计算初始朝向：让他们面向地图中心，这样双方容易相遇
            mx, my = self.state[i, 0], self.state[i, 1]
            target_angle = math.atan2(cy - my, cx - mx)
            
            self.state[i, 2] = 0 # vx
            self.state[i, 3] = 0 # vy
            self.state[i, 4] = target_angle
            self.state[i, 5] = 0 # omega
            self.state[i, 6] = 1.0 # hp
            self.state[i, 7] = CFG.MAG_SIZE # ammo
            self.state[i, 8] = CFG.MAX_MAGS # spare mags
            self.state[i, 9] = 0 # reload timer
            self.state[i, 10] = CFG.SIGMA_STABLE # recoil/spread
            self.state[i, 11] = 0 if is_a else 1 # team

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
        物理更新：
        1. 采用 AABB (轴对齐包围盒) 碰撞检测，矩形不随朝向旋转。
        2. 实现 Axis-Independent Movement (分轴移动) 以支持贴墙滑动。
        """
        # 获取当前状态
        # 注意：这里我们获取的是值的拷贝或引用，后续直接更新 self.state
        vx = self.state[:, 2]
        vy = self.state[:, 3]
        omega = self.state[:, 5]
        theta = self.state[:, 4]
        radius = CFG.RADIUS
        
        # 旋转更新 (旋转不受阻挡)
        theta_new = (theta + omega * CFG.DT) % (2 * np.pi)
        
        # 定义内部函数：检查特定位置是否发生 AABB 碰撞
        # box_center: (x, y)
        # return: True if collision detected
        def check_collision(cx, cy, agent_idx):
            # 智能体的 AABB 边界
            # 由于是外接矩形，半宽和半高都等于 Radius
            min_ax = cx - radius
            max_ax = cx + radius
            min_ay = cy - radius
            max_ay = cy + radius
            
            # 计算需要检查的网格索引范围
            # 向下取整获取左/上边界所在的格子，向上取整获取右/下边界
            start_gx = max(0, int(math.floor(min_ax)))
            end_gx = min(CFG.W - 1, int(math.floor(max_ax)))
            start_gy = max(0, int(math.floor(min_ay)))
            end_gy = min(CFG.H - 1, int(math.floor(max_ay)))
            
            # 遍历覆盖到的所有网格
            for gy in range(start_gy, end_gy + 1):
                for gx in range(start_gx, end_gx + 1):
                    # 如果该网格是墙壁 (map值 > 0.5)
                    if self.map[gy, gx] > 0.5:
                        # 只要碰到了任何一个墙壁格子，就是发生了碰撞
                        # 因为我们在用 AABB vs Grid，且 Grid 也是 AABB
                        # 只要 Grid 索引在 Agent AABB 范围内，且 Grid 是墙，即为重叠
                        return True
            return False

        # 对每个智能体分别进行物理更新
        for i in range(self.agents_total):
            if self.state[i, 6] <= 0: continue # 跳过死者
            
            curr_x = self.state[i, 0].item()
            curr_y = self.state[i, 1].item()
            
            # --- X 轴尝试移动 ---
            next_x = curr_x + vx[i].item() * CFG.DT
            # 检查：如果在 (next_x, curr_y) 位置是否会撞墙？
            if not check_collision(next_x, curr_y, i):
                # 没撞：应用 X 轴位移
                self.state[i, 0] = next_x
            else:
                # 撞了：X 轴保持不变 (curr_x)，实现“挡住”
                # 可选：将撞墙方向的速度清零，防止物理动量累积，但在简单移动中非必须
                pass 
                
            # --- Y 轴尝试移动 (滑动逻辑) ---
            # 注意：这里使用 update 过的 X (如果X移动成功) 或者旧的 X (如果X撞墙)
            # 这就是“滑动”的关键：X被挡住了，但Y还能动。
            current_x_after_step1 = self.state[i, 0].item()
            
            next_y = curr_y + vy[i].item() * CFG.DT
            # 检查：如果在 (current_x, next_y) 位置是否会撞墙？
            if not check_collision(current_x_after_step1, next_y, i):
                # 没撞：应用 Y 轴位移
                self.state[i, 1] = next_y
            else:
                # 撞了：Y 轴保持不变
                pass
            
            # --- 更新朝向 ---
            self.state[i, 4] = theta_new[i]

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
        射线检测：使用 DDA (Digital Differential Analyzer) 算法
        该算法能精确遍历射线经过的每一个网格，彻底解决穿墙问题。
        """
        # 1. 提取起点和终点的连续坐标
        x1, y1 = ps[0].item(), ps[1].item()
        x2, y2 = pt[0].item(), pt[1].item()

        # 2. 转换为网格坐标 (整数)
        map_x = int(math.floor(x1))
        map_y = int(math.floor(y1))
        end_map_x = int(math.floor(x2))
        end_map_y = int(math.floor(y2))

        # 3. 计算射线方向和距离
        ray_dir_x = x2 - x1
        ray_dir_y = y2 - y1
        dist = math.sqrt(ray_dir_x**2 + ray_dir_y**2) + 1e-6 # 防止除零
        
        # 归一化方向向量
        ray_dir_x /= dist
        ray_dir_y /= dist

        # 4. DDA 初始化
        # delta_dist: 射线移动一个网格单位在 X 或 Y 方向上所需的实际距离
        delta_dist_x = abs(1.0 / (ray_dir_x + 1e-9))
        delta_dist_y = abs(1.0 / (ray_dir_y + 1e-9))

        # step: 步进方向 (+1 或 -1)
        # side_dist: 从当前位置到下一个网格边界的距离
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

        # 5. DDA 步进循环
        # 最大步数设为长宽之和即可覆盖全图
        max_steps = CFG.W + CFG.H
        
        for _ in range(max_steps):
            # --- 核心检查 ---
            # 如果当前网格是墙壁，返回 True (发生碰撞)
            # 先做边界检查防止索引越界
            if 0 <= map_x < CFG.W and 0 <= map_y < CFG.H:
                if self.map[map_y, map_x] > 0.5:
                    return True 
            
            # 如果到达了目标所在的网格，说明中间没有障碍物，循环结束
            if map_x == end_map_x and map_y == end_map_y:
                break
            
            # --- 步进 ---
            # 总是向距离最近的那个轴移动一步
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                
        return False # 路径畅通
    
    def _resolve_combat(self, actions):
        """战斗判定与结算 (修正贴墙无法命中bug版)"""
        hits_log = []
        fire_cmd = actions[:, 3] > 0.5
        can_fire = (self.state[:, 6] > 0) & \
                   (self.state[:, 7] > 0) & \
                   (self.state[:, 9] <= 0) & \
                   ((self.time - self.last_shot_time) >= (1.0/CFG.FIRE_RATE))
        
        shooters = torch.nonzero(fire_cmd & can_fire).squeeze()
        if shooters.numel() == 0: return hits_log
        if shooters.dim() == 0: shooters = shooters.unsqueeze(0)

        for s_idx in shooters:
            self.state[s_idx, 7] -= 1
            self.last_shot_time[s_idx] = self.time
            
            noise = np.random.normal(0, self.state[s_idx, 10].item())
            shoot_angle = self.state[s_idx, 4].item() + noise
            self.state[s_idx, 10] = min(self.state[s_idx, 10] + CFG.DELTA_SIGMA, CFG.SIGMA_MAX)
            
            ps = self.state[s_idx, 0:2]
            direction = torch.tensor([math.cos(shoot_angle), math.sin(shoot_angle)], device=CFG.DEVICE)
            
            targets = torch.arange(self.agents_total, device=CFG.DEVICE)
            targets = targets[targets != s_idx]
            
            min_dist = 9999
            hit_target = -1
            hit_lateral_distance = 0
            
            for t_idx in targets:
                if self.state[t_idx, 6] <= 0: continue
                
                pt = self.state[t_idx, 0:2]
                v_st = pt - ps
                
                # 目标距离
                dist_st = torch.norm(v_st)
                
                # 投影距离
                proj_t = torch.dot(v_st, direction)
                
                if proj_t < 0: continue
                
                lateral_offset = v_st - proj_t * direction 
                lateral_distance = torch.norm(lateral_offset) 
                
                if lateral_distance < CFG.HIT_RADIUS: 
                    # --- 核心修复开始 ---
                    # 问题：如果直接检测 ps 到 pt，当 pt 紧贴墙壁时，DDA可能会检测到墙壁而判定阻挡。
                    # 解决：将检测终点从“目标中心”向“射手”回缩一段距离（身体半径的80%）。
                    # 这样射线只需到达目标“体表”即可，忽略目标背后或身下的墙。
                    
                    # 计算回缩比例。如果距离很近（贴脸），则不回缩（比例为0）
                    pullback_dist = CFG.HIT_RADIUS * 0.8
                    if dist_st > pullback_dist:
                        check_ratio = 1.0 - (pullback_dist / dist_st)
                        # 使用线性插值计算新的检测终点
                        pt_check = ps + v_st * check_ratio
                    else:
                        # 距离极近，直接检测到中心（此时基本不会有墙阻挡）
                        pt_check = pt
                        
                    if self._detect_ray_collision(ps, pt_check): continue
                    # --- 核心修复结束 ---
                    
                    if proj_t < min_dist:
                        min_dist = proj_t
                        hit_target = t_idx
                        hit_lateral_distance = lateral_distance
            
            if hit_target != -1:
                damage = CFG.DMG_MAX * math.exp(-(hit_lateral_distance.item()**2) / (2 * CFG.DMG_WIDTH**2))
                self.state[hit_target, 6] = max(0, self.state[hit_target, 6] - damage)
                hits_log.append({
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
        half_L = CFG.L // 2
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
            # 计算以智能体为中心需要截取的区域
            # 左上角坐标：中心坐标减去 L//2
            x_start = cx - half_L
            y_start = cy - half_L
            
            # 计算需要的 padding 大小（确保不会超出边界）
            pad_left = max(0, -x_start)
            pad_right = max(0, x_start + CFG.L - CFG.W)
            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_start + CFG.L - CFG.H)
            
            # 对地图进行 padding，超出部分用 1（墙）填充
            padded_map = torch.nn.functional.pad(
                self.map, 
                (pad_left, pad_right, pad_top, pad_bottom), 
                value=1
            )
            
            # 调整截取坐标（加上 padding 的偏移）
            x_start_padded = x_start + pad_left
            y_start_padded = y_start + pad_top
            
            # 截取 LxL 的局部地图
            local_grid = padded_map[y_start_padded:y_start_padded+CFG.L, x_start_padded:x_start_padded+CFG.L]
            o_spatial = local_grid.flatten() # 局部坐标系下的地图
            
            # 对其它智能体进行观测（敌方和友方）
            team_obs = []
            enemy_obs = []
            my_team = self.state[i, 11]
            my_pos = self.state[i, 0:2]
            
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