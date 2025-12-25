import math
import signal
import pooltool as pt
import numpy as np
import copy
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent


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

def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


class HybridAgent(Agent):
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
        
        if use_value_network and value_net_path:
            self._load_value_network(value_net_path)
        
        # 袋口位置（标准8球台）
        self.pocket_positions = [
            (0.0, 0.0),      # 左下
            (1.12, 0.0),     # 中下
            (2.24, 0.0),     # 右下
            (0.0, 1.12),     # 左上
            (1.12, 1.12),    # 中上
            (2.24, 1.12),    # 右上
        ]
        
        print(f"[HybridAgent] 初始化完成 (评估器: {'学习型' if use_value_network else '启发式'})")
    
    def _load_value_network(self, path):
        """加载预训练的局面评估网络（可选）"""
        try:
            import torch
            import torch.nn as nn
            
            class ValueNetwork(nn.Module):
                def __init__(self, input_dim=128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            self.value_net = ValueNetwork()
            self.value_net.load_state_dict(torch.load(path))
            self.value_net.eval()
            print(f"[HybridAgent] 成功加载评估网络: {path}")
        except Exception as e:
            print(f"[HybridAgent] 加载评估网络失败: {e}, 将使用启发式评估")
            self.value_net = None
    
    def _extract_state_features(self, balls, my_targets):
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
        
        # 目标球信息
        for bid in my_targets:
            if bid in balls and balls[bid].state.s != 4:
                ball_pos = balls[bid].state.rvw[0][:2]
                
                # 计算到各袋口的距离
                distances = []
                for pocket in self.pocket_positions:
                    dist = np.linalg.norm(ball_pos - np.array(pocket))
                    distances.append((pocket, dist))
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
        """模块2: 候选击球生成 - 进攻球（优化版）"""
        candidates = []
        cue_pos = features['cue_pos']
        ball_radius = 0.028575  # 标准球半径
        
        for target_info in features['target_balls']:
            target_pos = target_info['pos']
            target_id = target_info['id']
            
            # 对每个袋口生成候选
            for pocket_info in target_info['all_pockets'][:4]:  # 考虑最近4个袋口
                pocket = pocket_info[0]
                dist_target_to_pocket = pocket_info[1]
                
                # 跳过太远的袋口
                if dist_target_to_pocket > 1.5:
                    continue
                
                # 计算目标球→袋口方向
                pocket_arr = np.array(pocket)
                target_to_pocket = pocket_arr - target_pos
                dist = np.linalg.norm(target_to_pocket)
                
                if dist < 0.01:
                    continue
                
                target_to_pocket_norm = target_to_pocket / dist
                
                # 生成多个角度的候选（直球、薄球）
                # 0度=直球(full hit), ±15度=薄球(cut shot)
                for angle_offset in [0, -15, 15, -30, 30]:
                    offset_rad = np.radians(angle_offset)
                    
                    # 旋转击球方向
                    cos_off = np.cos(offset_rad)
                    sin_off = np.sin(offset_rad)
                    adjusted_dir = np.array([
                        target_to_pocket_norm[0] * cos_off - target_to_pocket_norm[1] * sin_off,
                        target_to_pocket_norm[0] * sin_off + target_to_pocket_norm[1] * cos_off
                    ])
                    
                    # 计算理想击球点
                    hit_point = target_pos - adjusted_dir * (2 * ball_radius)
                    
                    # 计算白球→击球点的方向和距离
                    cue_to_hit = hit_point - cue_pos
                    dist_cue_to_hit = np.linalg.norm(cue_to_hit)
                    
                    if dist_cue_to_hit < 0.01:
                        continue
                    
                    # 计算击球角度
                    phi = np.degrees(np.arctan2(cue_to_hit[1], cue_to_hit[0])) % 360
                    
                    # 计算击球难度（距离+角度偏移）
                    difficulty = dist_cue_to_hit * 0.5 + dist_target_to_pocket * 0.3 + abs(angle_offset) * 0.01
                    
                    # 根据距离和难度调整力度
                    total_dist = dist_cue_to_hit + dist_target_to_pocket
                    if total_dist < 0.5:
                        V0 = 1.8
                    elif total_dist < 1.0:
                        V0 = 2.5
                    elif total_dist < 1.5:
                        V0 = 3.5
                    elif total_dist < 2.0:
                        V0 = 4.5
                    else:
                        V0 = 5.5
                    
                    # 薄球需要更大力度
                    if abs(angle_offset) > 20:
                        V0 *= 1.2
                    
                    # 限制力度范围
                    V0 = np.clip(V0, 0.5, 8.0)
                    
                    # 生成候选动作
                    candidate = {
                        'V0': V0,
                        'phi': phi,
                        'theta': 0.0,  # 水平击打
                        'a': 0.0,      # 中心击打
                        'b': 0.0,
                        'type': 'attack',
                        'target_ball': target_id,
                        'target_pocket': pocket,
                        'angle_offset': angle_offset,
                        'estimated_difficulty': difficulty
                    }
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_safe_candidates(self, features, balls, table):
        """模块2: 候选击球生成 - 安全球（防守）"""
        candidates = []
        cue_pos = features['cue_pos']
        
        if not features['target_balls']:
            return candidates
        
        # 选择最近的目标球
        nearest_target = min(features['target_balls'], 
                           key=lambda t: np.linalg.norm(t['pos'] - cue_pos))
        
        target_pos = nearest_target['pos']
        cue_to_target = target_pos - cue_pos
        dist = np.linalg.norm(cue_to_target)
        
        if dist > 0.01:
            phi = np.degrees(np.arctan2(cue_to_target[1], cue_to_target[0])) % 360
            
            # 生成多种力度的安全球
            for V0 in [0.8, 1.2, 1.5]:
                candidate = {
                    'V0': V0,
                    'phi': phi,
                    'theta': 0.0,
                    'a': 0.0,
                    'b': 0.0,
                    'type': 'safe',
                    'target_ball': nearest_target['id'],
                    'estimated_difficulty': 0.0
                }
                candidates.append(candidate)
            
            # 添加侧旋球候选（让白球偏转）
            for side_spin in [-0.3, 0.3]:
                candidate = {
                    'V0': 1.5,
                    'phi': phi,
                    'theta': 0.0,
                    'a': side_spin,
                    'b': 0.0,
                    'type': 'safe',
                    'target_ball': nearest_target['id'],
                    'estimated_difficulty': 0.0
                }
                candidates.append(candidate)
        
        return candidates
    
    def _fast_simulate_and_score(self, candidate, balls, my_targets, table):
        """模块3: 快速仿真 + 评分"""
        # 创建仿真环境
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        shot.cue.set_state(
            V0=candidate['V0'],
            phi=candidate['phi'],
            theta=candidate['theta'],
            a=candidate['a'],
            b=candidate['b']
        )
        
        # 保存初始状态
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        # 执行仿真
        success = safe_simulate(shot, timeout_sec=2.0)
        
        if not success:
            return -1000  # 仿真失败，极低分
        
        # 使用现有的评分函数
        reward = analyze_shot_for_reward(shot, last_state, my_targets)
        
        # 如果使用学习型评估器，这里可以融合神经网络评分
        if self.use_value_network and self.value_net is not None:
            # TODO: 实现特征向量化并通过网络评估
            pass
        
        return reward
    
    def _filter_and_rank(self, candidates_with_scores):
        """模块5: 决策与约束过滤"""
        # 过滤掉评分很低的（可能犯规）
        valid_candidates = [c for c in candidates_with_scores if c['score'] > -50]
        
        if not valid_candidates:
            return None
        
        # 按得分排序
        valid_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return valid_candidates[0]
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策流程"""
        if balls is None or my_targets is None:
            print("[HybridAgent] 缺少必要信息，使用随机动作")
            return self._random_action()
        
        try:
            # 检查是否需要打8号球
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[HybridAgent] 目标切换为8号球")
            
            # 模块1: 状态解析
            features = self._extract_state_features(balls, my_targets)
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
            
            # 模块3: 快速仿真与评分
            # 优先评估看起来简单的候选（距离近、角度小）
            all_candidates.sort(key=lambda c: c.get('estimated_difficulty', 0))
            
            candidates_with_scores = []
            max_candidates_to_eval = min(20, len(all_candidates))  # 最多评估20个
            
            for i, candidate in enumerate(all_candidates[:max_candidates_to_eval]):
                score = self._fast_simulate_and_score(candidate, balls, my_targets, table)
                candidates_with_scores.append({
                    **candidate,
                    'score': score
                })
                if (i + 1) % 5 == 0:
                    print(f"[HybridAgent] 已评估 {i+1}/{max_candidates_to_eval} 个候选")
            
            # 模块5: 过滤与决策
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