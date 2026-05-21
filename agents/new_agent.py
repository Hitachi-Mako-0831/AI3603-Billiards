import math
import signal
import pooltool as pt
import numpy as np
import copy
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent

BALL_RADIUS = 0.028575  # 标准球半径（米）

# ======= 模拟超时保护 ========
class SimulationTimeout(Exception):
    """模拟超时异常"""
    pass

def _simulation_timeout_handler(signum, frame):
    raise SimulationTimeout("Simulation timed out")

def safe_simulate(shot, timeout_sec: float = 5.0):
    """
    带超时保护的物理模拟函数。
    
    参数：
        shot: pooltool System 对象
        timeout_sec: 超时秒数
    
    返回：
        success: 是否成功完成模拟
    """
    old_handler = signal.signal(signal.SIGALRM, _simulation_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    
    try:
        pt.simulate(shot, inplace=True)
        return True
    except SimulationTimeout:
        return False
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

# ======= 击球结果分析与评分 ========
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """分析击球结果并计算奖励分数（对齐 poolenv 规则，并对犯规给予强惩罚）"""

    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]

    # 首球碰撞（过滤非球对象）
    first_contact_ball_id = None
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    foul_first_hit = False
    if first_contact_ball_id is None:
        # 未击中任何球：清台后且只剩白球+黑8可不判犯规
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 必须首先击打目标球（清台后必须先碰黑8）
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True

    # 碰库判定（无进球时要求母球或首球碰库）
    cue_hit_cushion = False
    target_hit_cushion = False
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    foul_no_rail = False
    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True

    # 严重犯规/判负：强惩罚，避免策略选中
    if cue_pocketed and eight_pocketed:
        return -1_000_000.0
    if cue_pocketed:
        return -100_000.0
    if eight_pocketed and player_targets != ['8']:
        return -1_000_000.0

    # 会导致回滚换人的犯规：强惩罚
    if foul_first_hit or foul_no_rail:
        return -10_000.0

    # 正常得分：偏向“进自己球、少送球、尽量继续连杆”
    score = 0.0
    if eight_pocketed and player_targets == ['8']:
        score += 2000.0
    score += 800.0 * len(own_pocketed)
    score -= 300.0 * len(enemy_pocketed)
    if len(own_pocketed) == 0:
        score += 5.0  # 合法无进球给极小奖励（避免全是负分时乱选）
    return float(score)

def _angle_deg(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360)

def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def _is_segment_blocked(a: np.ndarray, b: np.ndarray, balls: dict, ignore_ids: set, clearance: float) -> bool:
    for bid, ball in balls.items():
        if bid in ignore_ids:
            continue
        if ball.state.s == 4:
            continue
        p = ball.state.rvw[0][:2]
        if _point_to_segment_distance(p, a, b) < clearance:
            return True
    return False

def _first_ball_hit_by_ray(cue_pos: np.ndarray, balls: dict, phi_deg: float, clearance: float) -> str | None:
    """用几何射线近似预测：给定出杆方向 phi，白球最先会碰到哪颗球。

    规则：若某球中心到射线距离 < 2R*clearance，则视为会被先撞到；取沿射线方向投影最小的那颗。
    这是开球阶段的快速合法性筛选（避免首碰对方球/黑8）。
    """
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))], dtype=float)
    best_id = None
    best_t = float('inf')

    for bid, ball in balls.items():
        if bid == 'cue' or ball is None or ball.state.s == 4:
            continue
        p = np.array(ball.state.rvw[0][:2], dtype=float)
        r = p - cue_pos
        t = float(np.dot(r, u))
        if t <= 0.0:
            continue
        perp = float(np.linalg.norm(r - t * u))
        if perp <= clearance:
            if t < best_t:
                best_t = t
                best_id = bid
    return best_id

class StateEncoder:
    """台球局面特征编码器 (从 rl_trainer.py 复制)"""

    def __init__(self, max_balls=7):
        self.max_balls = max_balls
        self.feature_dim = 2 + 2 + 14 + 14 + 15 + 10  # = 57
        self.table_length = 2.24
        self.table_width = 1.12

    def encode(self, balls, my_targets, table, pocket_positions=None):
        features = []
        if pocket_positions is None:
            if hasattr(table, 'pockets') and table.pockets:
                pocket_positions = [
                    (float(p.center[0]), float(p.center[1]))
                    for p in table.pockets.values()
                ]
            else:
                L = getattr(table, 'l', 2.24)
                W = getattr(table, 'w', 1.12)
                pocket_positions = [
                    (0.0, 0.0), (L/2, 0.0), (L, 0.0),
                    (0.0, W), (L/2, W), (L, W),
                ]

        cue_pos = self._get_ball_pos(balls, 'cue')
        features.extend(self._normalize_pos(cue_pos))

        eight_pos = self._get_ball_pos(balls, '8')
        features.extend(self._normalize_pos(eight_pos))

        target_positions = []
        for bid in my_targets:
            if bid != '8':
                pos = self._get_ball_pos(balls, bid)
                target_positions.append(pos)

        while len(target_positions) < self.max_balls:
            target_positions.append((-1, -1))

        for pos in target_positions[:self.max_balls]:
            features.extend(self._normalize_pos(pos))

        opponent_positions = []
        for bid, ball in balls.items():
            if bid not in my_targets and bid not in ['cue', '8']:
                if ball.state.s != 4:
                    pos = (ball.state.rvw[0][0], ball.state.rvw[0][1])
                    opponent_positions.append(pos)

        while len(opponent_positions) < self.max_balls:
            opponent_positions.append((-1, -1))

        for pos in opponent_positions[:self.max_balls]:
            features.extend(self._normalize_pos(pos))

        features.append(self._min_dist_to_pocket(cue_pos, pocket_positions))
        for pos in target_positions[:self.max_balls]:
            features.append(self._min_dist_to_pocket(pos, pocket_positions))
        for pos in opponent_positions[:self.max_balls]:
            features.append(self._min_dist_to_pocket(pos, pocket_positions))

        stats = self._compute_stats(balls, my_targets, cue_pos, pocket_positions)
        features.extend(stats)

        return np.array(features, dtype=np.float32)

    def _get_ball_pos(self, balls, ball_id):
        if ball_id in balls and balls[ball_id].state.s != 4:
            return (balls[ball_id].state.rvw[0][0], balls[ball_id].state.rvw[0][1])
        return (-1, -1)

    def _normalize_pos(self, pos):
        if pos[0] < 0:
            return [-1.0, -1.0]
        x = (pos[0] / self.table_length) * 2 - 1
        y = (pos[1] / self.table_width) * 2 - 1
        return [np.clip(x, -1, 1), np.clip(y, -1, 1)]

    def _min_dist_to_pocket(self, pos, pockets):
        if pos[0] < 0:
            return 1.0
        min_dist = float('inf')
        for pocket in pockets:
            dist = np.sqrt((pos[0] - pocket[0])**2 + (pos[1] - pocket[1])**2)
            min_dist = min(min_dist, dist)
        return min(min_dist / 2.5, 1.0)

    def _compute_stats(self, balls, my_targets, cue_pos, pockets):
        stats = []
        remaining_own = sum(1 for bid in my_targets
                          if bid in balls and balls[bid].state.s != 4 and bid != '8')
        stats.append(remaining_own / 7.0)

        remaining_opp = sum(1 for bid, b in balls.items()
                          if bid not in my_targets and bid not in ['cue', '8']
                          and b.state.s != 4)
        stats.append(remaining_opp / 7.0)

        is_targeting_eight = 1.0 if (remaining_own == 0 or my_targets == ['8']) else 0.0
        stats.append(is_targeting_eight)

        min_dist_to_target = 1.0
        for bid in my_targets:
            if bid in balls and balls[bid].state.s != 4:
                pos = (balls[bid].state.rvw[0][0], balls[bid].state.rvw[0][1])
                if cue_pos[0] >= 0:
                    dist = np.sqrt((cue_pos[0] - pos[0])**2 + (cue_pos[1] - pos[1])**2)
                    min_dist_to_target = min(min_dist_to_target, dist / 2.5)
        stats.append(min_dist_to_target)

        avg_dist = 0.0
        count = 0
        for bid in my_targets:
            if bid in balls and balls[bid].state.s != 4:
                pos = (balls[bid].state.rvw[0][0], balls[bid].state.rvw[0][1])
                avg_dist += self._min_dist_to_pocket(pos, pockets)
                count += 1
        stats.append(avg_dist / max(count, 1))

        if cue_pos[0] >= 0:
            dist_to_cushion = min(
                cue_pos[0], self.table_length - cue_pos[0],
                cue_pos[1], self.table_width - cue_pos[1]
            )
            stats.append(min(dist_to_cushion / 0.3, 1.0))
        else:
            stats.append(0.0)

        stats.extend([0.0, 0.0, 0.0, 0.0])
        return stats[:10]

# ======= 混合架构Agent定义 ========
class NewAgent(Agent):
    """混合架构Agent：规则生成候选 + 快速仿真 + 学习型评估"""
    
    def __init__(self, use_value_network=False, value_net_path=None):
        """初始化混合Agent
        
        参数：
            use_value_network: 是否使用学习型局面评估器
            value_net_path: 预训练评估网络路径
        """
        super().__init__()
        self.use_value_network = use_value_network
        self.value_net = None
        self.state_encoder = StateEncoder()
        
        if use_value_network and value_net_path:
            self._load_value_network(value_net_path)
        
        # 与 poolenv 噪声同量级的“鲁棒评估”参数
        self.exec_noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003,
        }
        
        print(f"[HybridAgent] 初始化完成 (评估器: {'学习型' if use_value_network else '启发式'})")
    
    def _load_value_network(self, path):
        """加载预训练的局面评估网络（可选）"""
        try:
            import torch
            import torch.nn as nn
            
            class ValueNetwork(nn.Module):
                def __init__(self, input_dim=57, hidden_dims=[256, 128, 64]):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.extend([
                            nn.Linear(prev_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.1)
                        ])
                        prev_dim = hidden_dim
                    layers.append(nn.Linear(prev_dim, 1))
                    layers.append(nn.Sigmoid())
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            self.value_net = ValueNetwork()
            checkpoint = torch.load(path, map_location='cpu')
            
            # 处理保存为字典的情况（包含优化器状态等）
            if isinstance(checkpoint, dict) and 'value_net' in checkpoint:
                self.value_net.load_state_dict(checkpoint['value_net'])
            else:
                self.value_net.load_state_dict(checkpoint)
                
            self.value_net.eval()
            print(f"[HybridAgent] 成功加载评估网络: {path}")
        except Exception as e:
            print(f"[HybridAgent] 加载评估网络失败: {e}, 将使用启发式评估")
            self.value_net = None

    def _is_break_state(self, balls, table) -> bool:
        """粗略识别是否为开球局面（15球完整且呈现紧密球堆）。

        说明：环境未显式提供 hit_count，这里用几何特征做判断。
        """
        if balls is None or table is None:
            return False
        if 'cue' not in balls or balls['cue'].state.s == 4:
            return False

        obj_ids = [str(i) for i in range(1, 16)]
        if any(balls.get(bid) is None for bid in obj_ids):
            return False
        if any(balls[bid].state.s == 4 for bid in obj_ids):
            return False

        obj_pos = np.array([balls[bid].state.rvw[0][:2] for bid in obj_ids], dtype=float)
        centroid = obj_pos.mean(axis=0)
        spread = obj_pos.std(axis=0)

        # pooltool 坐标通常为：x 轴≈桌宽 table.w，y 轴≈桌长 table.l
        l = float(getattr(table, 'l', 2.24))
        w = float(getattr(table, 'w', 1.12))

        # 典型 rack 在 y（桌长）方向的后半段，且 x（桌宽）方向围绕中线
        centroid_ok = (0.55 * l <= centroid[1] <= 0.92 * l) and (0.35 * w <= centroid[0] <= 0.65 * w)
        # 紧密程度：std 不能太大（rack 典型宽度约 0.25~0.35m）
        spread_ok = (spread[0] <= 0.18) and (spread[1] <= 0.15)
        return bool(centroid_ok and spread_ok)

    def _generate_break_candidates(self, balls, my_targets, table):
        """开球候选：只围绕球堆中心给少量、保守的直冲选项。

        思路：
        - 估计球堆质心方向 phi_center。
        - 仅在小范围偏角内采样（±3°），力度用两档中等速度。
        - 方向首碰预测不是己方球则丢弃，避免一杆直接犯规。
        """
        candidates: list[dict] = []
        cue_pos = np.array(balls['cue'].state.rvw[0][:2], dtype=float)

        obj_positions = []
        for bid in [str(i) for i in range(1, 16)]:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            obj_positions.append(np.array(ball.state.rvw[0][:2], dtype=float))
        if not obj_positions:
            return candidates

        centroid = np.mean(np.stack(obj_positions, axis=0), axis=0)
        phi_center = _angle_deg(centroid - cue_pos)

        clearance = 2.05 * BALL_RADIUS
        target_set = set(my_targets)

        for phi_offset in [-3.0, 0.0, 3.0]:
            phi = float((phi_center + phi_offset) % 360)
            first_id = _first_ball_hit_by_ray(cue_pos, balls, phi, clearance=clearance)
            if first_id is None or first_id not in target_set:
                continue

            for V0 in [5.5, 6.2]:
                candidates.append({
                    'V0': float(np.clip(V0, 0.5, 8.0)),
                    'phi': phi,
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0,
                    'type': 'break',
                    'target_ball': first_id,
                    'estimated_difficulty': 0.0,
                })

        return candidates
    
    def _extract_state_features(self, balls, my_targets, table):
        """模块1: 状态解析 - 提取桌面特征"""
        features = {
            'cue_pos': None,
            'target_balls': [],
            'opponent_balls': [],
            'distance_to_pockets': {},
            'blockage': {}
        }
        
        # 白球位置
        if 'cue' in balls and balls['cue'].state.s != 4:
            features['cue_pos'] = balls['cue'].state.rvw[0][:2]
        else:
            return None  # 白球不存在
        
        # 袋口中心（来自 table，避免硬编码尺寸）
        pocket_positions = []
        if table is not None and hasattr(table, 'pockets'):
            for _, pocket in table.pockets.items():
                pocket_positions.append(pocket.center[:2])
        if not pocket_positions:
            # 兜底：标准尺寸
            pocket_positions = [
                np.array((0.0, 0.0)),
                np.array((1.12, 0.0)),
                np.array((2.24, 0.0)),
                np.array((0.0, 1.12)),
                np.array((1.12, 1.12)),
                np.array((2.24, 1.12)),
            ]

        # 目标球信息
        for bid in my_targets:
            if bid in balls and balls[bid].state.s != 4:
                ball_pos = balls[bid].state.rvw[0][:2]
                
                # 计算到各袋口的距离
                distances = []
                for pocket_pos in pocket_positions:
                    dist = float(np.linalg.norm(ball_pos - pocket_pos))
                    distances.append((pocket_pos, dist))
                distances.sort(key=lambda x: x[1])
                
                features['target_balls'].append({
                    'id': bid,
                    'pos': ball_pos,
                    'nearest_pocket': distances[0],
                    'all_pockets': distances
                })
        
        # 对手球信息（用于计算阻挡）
        for bid, ball in balls.items():
            if bid not in my_targets and bid not in ['cue', '8'] and ball.state.s != 4:
                features['opponent_balls'].append({
                    'id': bid,
                    'pos': ball.state.rvw[0][:2]
                })
        
        return features
    
    def _generate_attack_candidates(self, features, balls, table):
        """模块2: 候选击球生成 - 进攻球（ghost-ball + 遮挡过滤）"""
        candidates = []
        cue_pos = features['cue_pos']

        # 用于遮挡判断的间隙（略大于 2R，给噪声留余量）
        clearance = 2.15 * BALL_RADIUS
        
        for target_info in features['target_balls']:
            target_pos = target_info['pos']
            target_id = target_info['id']
            
            # 对每个袋口生成候选（优先近袋口）
            for pocket_pos, dist_obj_to_pocket in target_info['all_pockets'][:4]:
                if dist_obj_to_pocket > 1.65:
                    continue

                obj_to_pocket = pocket_pos - target_pos
                d = float(np.linalg.norm(obj_to_pocket))
                if d < 1e-3:
                    continue
                unit = obj_to_pocket / d

                # ghost ball 位置：沿着目标->袋口的反方向退 2R
                ghost_pos = target_pos - unit * (2.0 * BALL_RADIUS)

                # 路径遮挡过滤：母球到 ghost、目标球到袋口
                if _is_segment_blocked(cue_pos, ghost_pos, balls, ignore_ids={'cue', target_id}, clearance=clearance):
                    continue
                if _is_segment_blocked(target_pos, pocket_pos, balls, ignore_ids={'cue', target_id}, clearance=clearance):
                    continue

                cue_to_ghost = ghost_pos - cue_pos
                dist_cue_to_ghost = float(np.linalg.norm(cue_to_ghost))
                if dist_cue_to_ghost < 1e-3:
                    continue

                phi_ideal = _angle_deg(cue_to_ghost)

                # 根据距离估计力度（保守，避免噪声下大幅走位失控）
                total_dist = dist_cue_to_ghost + float(dist_obj_to_pocket)
                v_base = 1.3 + 1.2 * total_dist
                v_base = float(np.clip(v_base, 1.0, 6.5))

                # 角度微调：对抗执行噪声
                for phi_offset in [0.0, -0.5, 0.5, -1.0, 1.0]:
                    for v_mul in [0.95, 1.0, 1.08]:
                        V0 = float(np.clip(v_base * v_mul, 0.5, 8.0))
                        phi = float((phi_ideal + phi_offset) % 360)

                        difficulty = 0.65 * dist_cue_to_ghost + 0.35 * float(dist_obj_to_pocket) + 0.05 * abs(phi_offset)
                        candidates.append({
                            'V0': V0,
                            'phi': phi,
                            'theta': 0.0,
                            'a': 0.0,
                            'b': 0.0,
                            'type': 'attack',
                            'target_ball': target_id,
                            'target_pocket': pocket_pos,
                            'phi_offset': phi_offset,
                            'estimated_difficulty': float(difficulty),
                        })
        
        return candidates
    
    def _generate_safe_candidates(self, features, balls, table):
        """模块2: 候选击球生成 - 安全球（防守）"""
        candidates = []
        cue_pos = features['cue_pos']
        
        if not features['target_balls']:
            return candidates
        
        # 选择最近的目标球（更容易做到“先碰目标球”从而不犯规）
        nearest_target = min(
            features['target_balls'],
            key=lambda t: float(np.linalg.norm(t['pos'] - cue_pos))
        )
        
        target_pos = nearest_target['pos']
        cue_to_target = target_pos - cue_pos
        dist = np.linalg.norm(cue_to_target)
        
        if dist > 0.01:
            phi_ideal = _angle_deg(cue_to_target)

            # 安全球策略：低速 + 小角度扰动 + 轻微侧旋（制造走位不确定性）
            for V0 in [0.75, 1.0, 1.25, 1.5]:
                for phi_offset in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    candidates.append({
                        'V0': float(V0),
                        'phi': float((phi_ideal + phi_offset) % 360),
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0,
                        'type': 'safe',
                        'target_ball': nearest_target['id'],
                        'estimated_difficulty': float(dist + abs(phi_offset) * 0.02),
                    })

            for side_spin in [-0.25, 0.25]:
                for phi_offset in [-0.5, 0.0, 0.5]:
                    candidates.append({
                        'V0': 1.35,
                        'phi': float((phi_ideal + phi_offset) % 360),
                        'theta': 0.0,
                        'a': float(side_spin),
                        'b': 0.0,
                        'type': 'safe',
                        'target_ball': nearest_target['id'],
                        'estimated_difficulty': float(dist + 0.1),
                    })
        
        return candidates
    
    def _fast_simulate_and_score(self, candidate, balls, my_targets, table, n_rollouts: int = 1):
        """模块3: 快速仿真 + 评分（带噪声rollout，增强鲁棒性）"""

        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        scores = []

        for _ in range(max(1, int(n_rollouts))):
            # 创建仿真环境
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

            # 评估时注入与环境同量级噪声，避免“极限球”在真实执行中崩掉
            noisy_V0 = float(np.clip(candidate['V0'] + np.random.normal(0, self.exec_noise_std['V0']), 0.5, 8.0))
            noisy_phi = float((candidate['phi'] + np.random.normal(0, self.exec_noise_std['phi'])) % 360)
            noisy_theta = float(np.clip(candidate['theta'] + np.random.normal(0, self.exec_noise_std['theta']), 0.0, 90.0))
            noisy_a = float(np.clip(candidate['a'] + np.random.normal(0, self.exec_noise_std['a']), -0.5, 0.5))
            noisy_b = float(np.clip(candidate['b'] + np.random.normal(0, self.exec_noise_std['b']), -0.5, 0.5))

            shot.cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)

            success = safe_simulate(shot, timeout_sec=2.0)
            if not success:
                scores.append(-50_000.0)
                continue

            # 基础规则评分
            rule_score = analyze_shot_for_reward(shot, last_state, my_targets)
            
            # 价值网络评分（仅在非严重犯规时启用）
            value_score = 0.0
            if self.use_value_network and self.value_net is not None and rule_score > -5000:
                try:
                    import torch
                    # 提取新状态特征
                    features = self.state_encoder.encode(shot.balls, my_targets, sim_table)
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    with torch.no_grad():
                        value = self.value_net(features_tensor).item()
                    
                    # 价值网络输出 [0, 1]，映射到规则分数量级 (例如 0.5 -> 0, 1.0 -> +500)
                    # 这里假设 value 代表胜率，胜率越高越好
                    value_score = (value - 0.5) * 1000.0
                except Exception as e:
                    print(f"[HybridAgent] 价值评估出错: {e}")
            
            scores.append(rule_score + value_score)

        mean = float(np.mean(scores))
        std = float(np.std(scores))
        return mean - 0.35 * std
    
    def _filter_and_rank(self, candidates_with_scores):
        """模块4: 决策与约束过滤"""
        # 优先过滤掉“必然犯规/判负”的候选（analyze_shot_for_reward 会给到 ~-10000 或更低）
        non_catastrophic = [c for c in candidates_with_scores if c.get('score', -1e9) > -9_000]
        if non_catastrophic:
            non_catastrophic.sort(key=lambda x: x['score'], reverse=True)
            return non_catastrophic[0]

        # 若全都很差，仍返回分数最高的（比直接随机更不容易反复犯规）
        if candidates_with_scores:
            candidates_with_scores.sort(key=lambda x: x.get('score', -1e9), reverse=True)
            return candidates_with_scores[0]
        return None
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策流程"""
        if balls is None or my_targets is None:
            print("[HybridAgent] 缺少必要信息，使用随机动作")
            return self._random_action()
        
        try:
            # 开球专用策略：避免开局 attack_candidates=0 时退化为随机导致直接犯规
            if balls is not None and table is not None and self._is_break_state(balls, table):
                print("[HybridAgent] 检测到开球局面，启用开球策略")
                break_candidates = self._generate_break_candidates(balls, my_targets, table)
                if break_candidates:
                    # 开球只执行一次，允许更稳健的复评
                    scored = []
                    for cand in break_candidates:
                        score = self._fast_simulate_and_score(cand, balls, my_targets, table, n_rollouts=4)
                        scored.append({**cand, 'score': float(score)})
                    scored.sort(key=lambda x: x['score'], reverse=True)
                    best = scored[0]

                    # 若所有开球都接近必然犯规（例如确实找不到“稳定首碰合法球”的方向），再回退常规策略
                    if float(best.get('score', -1e9)) <= -9_000:
                        print(f"[HybridAgent] 开球候选整体过差(最佳={best['score']:.1f})，回退常规策略")
                    else:
                        action = {
                            'V0': float(best['V0']),
                            'phi': float(best['phi']),
                            'theta': float(best['theta']),
                            'a': float(best['a']),
                            'b': float(best['b']),
                        }
                        print(f"[HybridAgent] 开球决策 (得分: {best['score']:.1f}): V0={action['V0']:.2f}, phi={action['phi']:.1f}°")
                        return action

            # 检查是否需要打8号球
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[HybridAgent] 目标切换为8号球")
            
            # 模块1: 状态解析
            features = self._extract_state_features(balls, my_targets, table)
            if features is None or features['cue_pos'] is None:
                return self._random_action()
            
            print(f"[HybridAgent] 分析局面: {len(features['target_balls'])} 个目标球")
            
            # 模块2: 生成候选
            attack_candidates = self._generate_attack_candidates(features, balls, table)
            safe_candidates = self._generate_safe_candidates(features, balls, table)
            
            all_candidates = attack_candidates + safe_candidates
            
            if not all_candidates:
                print("[HybridAgent] 无法生成候选，使用随机动作")
                return self._random_action()
            
            print(f"[HybridAgent] 生成 {len(attack_candidates)} 个进攻候选, {len(safe_candidates)} 个防守候选")
            
            # 模块3: 快速仿真与评分（两阶段：粗评 -> 鲁棒复评）
            all_candidates.sort(key=lambda c: c.get('estimated_difficulty', 0.0))

            print(f"[HybridAgent] 开始评分 {len(all_candidates)} 个候选击球动作")
            stage1 = all_candidates[:min(18, len(all_candidates))]
            stage1_scored = []
            for candidate in stage1:
                score = self._fast_simulate_and_score(candidate, balls, my_targets, table, n_rollouts=1)
                stage1_scored.append({**candidate, 'score': float(score)})

            stage1_scored.sort(key=lambda x: x['score'], reverse=True)
            stage2 = stage1_scored[:min(6, len(stage1_scored))]
            candidates_with_scores = []
            for candidate in stage2:
                score = self._fast_simulate_and_score(candidate, balls, my_targets, table, n_rollouts=3)
                candidates_with_scores.append({**candidate, 'score': float(score)})
            
            # 模块4: 过滤与决策
            best = self._filter_and_rank(candidates_with_scores)
            
            if best is None:
                print("[HybridAgent] 所有候选都不合格，使用随机动作")
                return self._random_action()
            
            action = {
                'V0': float(best['V0']),
                'phi': float(best['phi']),
                'theta': float(best['theta']),
                'a': float(best['a']),
                'b': float(best['b'])
            }
            
            print(f"[HybridAgent] 决策 ({best['type']}, 得分: {best['score']:.1f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.1f}°")
            
            return action
        
        except Exception as e:
            print(f"[HybridAgent] 决策出错: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()