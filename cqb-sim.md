# 室内近距离作战（CQB）多智能体强化学习环境技术规范

## 1. 环境全局定义

本环境模拟了一个二维平面的室内战术对抗场景，包含两个对抗阵营。

* **地图空间**：环境地图表示为矩阵 $\bm M \in \{0,1\}^{H \times W}$。
    * $0$：表示可通行区域。
    * $1$：表示障碍物（墙体）。
    * 坐标系：采用全局二维笛卡尔坐标系，原点位于地图左下角，X轴向右，Y轴向上。
* **智能体阵营**：
    * 红方阵营 $A = \{e_1^A, \dots, e_{n_A}^A\}$。
    * 蓝方阵营 $B = \{e_1^B, \dots, e_{n_B}^B\}$。
* **仿真参数**：
    * 时间步长：$\Delta t$。
    * 距离单位：米 (m)。

## 2. 状态空间 (State Space)

环境维护一个全局状态集合 $\mathcal{S}$，包含所有智能体的真实物理状态。对于任意阵营的第 $i$ 个智能体，其状态向量 $\bm s_i$ 定义如下：

$$
\bm s_i = (\bm p_i, \bm v_i, \theta_i, \omega_i, h_i, c_i, n_i, r_i, \sigma_i)
$$

具体含义及取值范围：

| 符号 | 描述 | 定义/值域 |
| :--- | :--- | :--- |
| $\bm p_i$ | 全局位置坐标 | $(x_i, y_i) \in [0, W] \times [0, H]$ |
| $\bm v_i$ | 全局线速度向量 | $(v_{x,i}, v_{y,i}) \in \mathbb{R}^2$，且 $\|\bm v_i\| \le v_{\max}$ |
| $\theta_i$ | **绝对朝向** | $\theta_i \in [0, 2\pi)$，表示与全局X轴正方向的夹角 |
| $\omega_i$ | 旋转角速度 | $\omega_i \in [-\omega_{\max}, \omega_{\max}]$ |
| $h_i$ | 当前生命值 | $h_i \in [0, 1]$，0为阵亡 |
| $c_i$ | 当前弹匣剩余子弹 | $c_i \in \{0, 1, \dots, s_{\text{mag}}\}$ |
| $n_i$ | 剩余备用弹匣数量 | $n_i \in \{0, 1, \dots, n_{\text{max}}\}$ |
| $r_i$ | **换弹状态标志** | $r_i \in \{0, 1\}$，1表示正在执行换弹动作 |
| $\sigma_i$ | 当前射击散布值 | $\sigma_i \in [\sigma_0, \sigma_{\text{max}}]$ |

---

## 3. 观测空间 (Observation Space)

为了保证空间特征的一致性，便于卷积神经网络（CNN）处理，所有空间相关的观测均采用**全局轴对齐（Global Axis-Aligned）** 参考系。

每个智能体 $i$ 的观测向量由四部分组成：
$$
\bm o_i = (\bm o_{\text{self}}, \bm o_{\text{spatial}}, \bm o_{\text{team}}, \bm o_{\text{enemy}})
$$

### 3.1 本体感知 ($\bm o_{\text{self}}$)
智能体感知自身的运动状态和生存状态：
$$
\bm o_{\text{self}} = (\theta_i, v_{x,i}, v_{y,i}, \omega_i, h_i, c_i, n_i, r_i, \sigma_i)
$$
* **注意**：$\theta_i$ 和 $(v_{x,i}, v_{y,i})$ 均为全局坐标系下的数值。

### 3.2 空间感知 ($\bm o_{\text{spatial}}$)
采用**轴对齐局部裁剪**方式获取环境几何信息：
* 以智能体当前位置 $\bm p_i$ 为中心，截取边长为 $L$ 的正方形区域。
* **不进行旋转**，保持裁剪区域的边缘与全局地图的 X/Y 轴平行。
* 输出为一个局部栅格矩阵 $\bm M_{local} \in \{0, 1\}^{S \times S}$（$S$ 为栅格分辨率）。
* 在此观测中，智能体始终位于中心，但其面朝方向由 $\bm o_{\text{self}}$ 中的 $\theta_i$ 决定。

### 3.3 队友信息 ($\bm o_{\text{team}}$)
智能体可获取所有队友 $j$ 的状态信息。位置信息采用**全局轴向偏移量**：
$$
\bm o_{\text{team}}^j = (\Delta x_{ij}, \Delta y_{ij}, \theta_j, v_{x,j}, v_{y,j}, h_j, c_j, n_j, r_j)
$$
* $\Delta x_{ij} = x_j - x_i$
* $\Delta y_{ij} = y_j - y_i$
* $\theta_j$ 为队友的全局朝向。

### 3.4 敌方信息 ($\bm o_{\text{enemy}}$)
敌方信息的获取受**可视性（Visibility）** 约束。
* **可视条件**：当且仅当敌方 $k$ 满足以下两个条件时可见：
    1.  视线无遮挡：$\bm p_i$ 与 $\bm p_k$ 连线不经过障碍物。
    2.  视锥范围内：敌方位于智能体朝向 $\theta_i$ 的 $\pm \delta$ 夹角内。
* **观测内容**：若可见，则获取敌方信息（隐蔽备弹量，暴露换弹状态）：
$$
\bm o_{\text{enemy}}^k = \begin{cases}
(\Delta x_{ik}, \Delta y_{ik}, \theta_k, v_{x,k}, v_{y,k}, \omega_k, h_k, r_k), & \text{若可见} \\
\mathbf{0} \text{ 或 Mask标记}, & \text{若不可见}
\end{cases}
$$
* **信息隐蔽**：敌方的 $c_k$ (当前子弹), $n_k$ (备弹), $\sigma_k$ (散布) 对智能体不可见。
* **战术信息**：敌方的 $r_k$ (是否正在换弹) 对智能体可见。

---

## 4. 动作空间 (Action Space)

采用连续动作空间，支持全向移动。为了符合操作直觉，移动指令基于**机体坐标系（Body Frame）**，但在物理引擎中会转换为全局位移。

$$
\bm a_i = (a_{\text{surge}}, a_{\text{sway}}, a_{\omega}, a_{\text{fire}}, a_{\text{reload}})
$$

1.  **移动指令**：
    * $a_{\text{surge}} \in [-1, 1]$：期望的纵向加速度（正为前，负为后）。
    * $a_{\text{sway}} \in [-1, 1]$：期望的横向加速度（正为右，负为左）。
2.  **转向指令**：
    * $a_{\omega} \in [-1, 1]$：期望的角加速度。
3.  **战术指令**：
    * $a_{\text{fire}} \in [0, 1]$：当值 $> 0.5$ 时触发射击。
    * $a_{\text{reload}} \in [0, 1]$：当值 $> 0.5$ 时触发换弹。

---

## 5. 动力学模型 (Dynamics)

### 5.1 全向移动与位置更新
虽然动作输入是相对的，但物理状态更新在全局坐标系下进行。

1.  **加速度坐标转换**：
    利用旋转矩阵 $\mathbf{R}(\theta_i)$ 将机体加速度转换为全局加速度 $\bm a_{\text{global}}$：
    $$
    \bm a_{\text{global}} = \alpha \cdot \begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix} \begin{pmatrix} a_{\text{surge}} \\ a_{\text{sway}} \end{pmatrix}
    $$
    其中 $\alpha$ 为加速度标量系数。

2.  **速度更新与阻力**：
    $$
    \bm v_{t+1} = \bm v_t + \bm a_{\text{global}} \cdot \Delta t - \mu \bm v_t
    $$
    其中 $\mu$ 为模拟地面的摩擦阻力系数。

3.  **速度截断（各向同性）**：
    为了保证任意方向的最大移动能力一致：
    $$
    \text{If } \|\bm v_{t+1}\| > v_{\max}, \quad \text{Then } \bm v_{t+1} \leftarrow v_{\max} \cdot \frac{\bm v_{t+1}}{\|\bm v_{t+1}\|}
    $$

4.  **位置更新**：
    $$
    \bm p_{t+1} = \bm p_t + \bm v_{t+1} \cdot \Delta t
    $$
    *碰撞处理：若 $\bm p_{t+1}$ 位于障碍物内，则位置回退并重置速度为 0。*

### 5.2 散步动力学 (Spread Dynamics)
散步值 $\sigma$ 决定射击精度，受移动状态和射击行为共同影响。

**更新公式**：
$$
\sigma_{t+1} = \text{Clip} \left( \sigma_{\text{base}}(t) + \sigma_{\text{shoot}}(t), \ \sigma_0, \ \sigma_{\max} \right)
$$

**计算步骤**：

1.  **确定目标散步基准 ($\sigma_{\text{target}}$)**：
    * 若 $\|\bm v_t\| > v_{\text{stable}}$ 或 $|\omega_t| > \omega_{\text{stable}}$（处于非稳态），则 $\sigma_{\text{target}} = \sigma_1$。
    * 否则（处于稳态），$\sigma_{\text{target}} = \sigma_0$。

2.  **计算基准散步变化 ($\sigma_{\text{base}}$)**：
    * **恢复过程**（当 $\sigma_t > \sigma_{\text{target}}$）：采用线性衰减。
        $$
        \sigma_{\text{base}}(t) = \sigma_t - k_{\text{decay}} \cdot \Delta t
        $$
    * **恶化过程**（当 $\sigma_t < \sigma_{\text{target}}$）：发生突变。
        $$
        \sigma_{\text{base}}(t) = \sigma_{\text{target}}
        $$

3.  **计算射击惩罚 ($\sigma_{\text{shoot}}$)**：
    * 若本时间步发生射击，则 $\sigma_{\text{shoot}}(t) = \Delta \sigma$。
    * 否则 $\sigma_{\text{shoot}}(t) = 0$。

### 5.3 武器与伤害机制

1.  **射击冷却**：
    射速限制为 $\gamma$。系统记录上次射击时间 $t_{\text{last}}$。仅当 $t - t_{\text{last}} \ge 1/\gamma$ 时允许开火。

2.  **换弹逻辑**：
    * **触发**：当 $a_{\text{reload}} > 0.5$ 且 $n_i > 0$ 且 $r_i=0$ 时，进入换弹状态。
    * **状态持续**：置 $r_i = 1$，开启倒计时 $T_{\text{reload}}$。在此期间禁止射击。
    * **完成**：倒计时结束后，更新弹药：$c_i = s_{\text{mag}}, n_i = n_i - 1, r_i = 0$。

3.  **命中与伤害判定**：
    * **弹道模拟**：子弹被视为一条从发射点发出的射线。发射方向服从正态分布 $\mathcal{N}(\theta_i, \sigma_i^2)$。
    * **命中判定**：计算射线与目标 $k$ 的中心坐标 $\bm p_k$ 之间的垂直距离 $d_\perp$。若 $d_\perp < \lambda$（判定半径），则视为命中。
    * **伤害计算**：采用高斯衰减模型。
        $$
        \text{Damage} = D_{\max} \cdot \exp\left( - \frac{d_\perp^2}{2w^2} \right)
        $$
        其中 $w$ 为伤害分布宽度系数，模拟了从中心命中到边缘擦伤的伤害过渡。