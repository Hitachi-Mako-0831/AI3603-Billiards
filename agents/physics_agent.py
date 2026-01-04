import numpy as np
import math
import random
import copy
import pooltool as pt
from agents.agent import Agent

class PhysicsAgent(Agent):
    """
    基于几何物理规则的 Heuristic Agent (Physics-Based)
    不使用神经网络，直接通过几何计算寻找最佳击球线路。
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.ball_radius = 0.028575 # 台球标准半径 (米)
        self.top_k_candidates = 8
        self.mc_top_k = 5
        self.mc_samples = 4
        self.mc_sim_budget = 20
        self.mc_std_coef = 0.25
        self.mc_cat_coef = 2000.0
        self.mcts_c = 1.414
        self.black_safety_samples = 12
        self.black_safety_max_samples = 96
        self.black_safety_noise_scale = 1.6
        self.eight_move_penalty = 650.0
        self.eight_near_pocket_d1 = 0.14
        self.eight_near_pocket_d2 = 0.25
        self.cue8_touch_score_penalty = 300.0
        self.cue8_touch_mc_penalty = 700.0
        self.speed_penalty_coef = 25.0
        self.enemy_pocket_base_penalty = 220.0
        self.enemy_pocket_multi_penalty = 260.0
        self.mc_noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003,
        }
        self.robust_noise_std = {
            'V0': 0.1,
            'phi': 0.15,
            'theta': 0.1,
            'a': 0.005,
            'b': 0.005,
        }
        self.clearance_threshold = 2 * self.ball_radius * 0.95
        print("[PhysicsAgent] 初始化为物理几何 Agent (Physics-Based Heuristic)")

    def decision(self, balls, my_targets, table):
        """
        核心决策逻辑：
        1. 遍历所有我的目标球
        2. 对每个目标球，遍历所有袋口
        3. 计算击球参数（角度、幽灵球位置）
        4. 检查路径是否被阻挡
        5. 选择成功率最高（切角最小、距离适中）的线路
        """
        
        # 1. 获取关键对象位置
        if 'cue' not in balls:
            return self._random_action() 
            
        cue_ball = balls['cue']
        cue_pos = np.array(cue_ball.state.rvw[0][:2]) # [x, y]
        
        candidates = []

        stage_targets, stage_is_eight = self._compute_stage_targets(balls, my_targets)
        target_ids = stage_targets

        max_speed = 7.0 if stage_is_eight else 6.6
        if not stage_is_eight:
            remaining_n = int(len(stage_targets))
            if remaining_n <= 1:
                max_speed = min(max_speed, 4.2)
            elif remaining_n <= 2:
                max_speed = min(max_speed, 4.8)
        if (not stage_is_eight) and ("8" in balls) and getattr(getattr(balls["8"], "state", None), "s", None) != 4:
            eight_pos = np.array(balls["8"].state.rvw[0][:2])
            dmin = None
            for _pid, pocket in table.pockets.items():
                ppos = np.array(pocket.center[:2])
                d = float(np.linalg.norm(ppos - eight_pos))
                if dmin is None or d < dmin:
                    dmin = d
            if dmin is not None and dmin <= float(self.eight_near_pocket_d2):
                max_speed = min(max_speed, 5.5)
        
        # 获取所有袋口中心坐标
        pockets = []
        for pid, pocket in table.pockets.items():
            # pocket.center 是 [x, y, z]
            pockets.append(np.array(pocket.center[:2]))
            
        # 2. 遍历所有可能的目标球和袋口组合
        for tid in target_ids:
            if tid not in balls:
                continue
            if getattr(getattr(balls[tid], "state", None), "s", None) == 4:
                continue
                
            target_ball = balls[tid]
            target_pos = np.array(target_ball.state.rvw[0][:2])
            
            for pocket_pos in pockets:
                # --- A. 计算幽灵球位置 (Ghost Ball Position) ---
                # 向量：目标球 -> 袋口
                target_to_pocket = pocket_pos - target_pos
                dist_target_pocket = np.linalg.norm(target_to_pocket)
                
                if dist_target_pocket < 1e-6:
                    continue
                    
                # 单位向量
                dir_target_pocket = target_to_pocket / dist_target_pocket
                
                # 幽灵球位置：目标球沿反方向延伸 2倍半径
                # 母球中心必须到达这里，才能把目标球沿 dir_target_pocket 撞出去
                ghost_pos = target_pos - dir_target_pocket * (2 * self.ball_radius)
                
                # --- B. 计算击球向量 (Cue -> Ghost) ---
                cue_to_ghost = ghost_pos - cue_pos
                dist_cue_ghost = np.linalg.norm(cue_to_ghost)
                
                if dist_cue_ghost < 1e-6:
                    continue
                    
                dir_cue_ghost = cue_to_ghost / dist_cue_ghost
                
                # --- C. 计算切球角度 (Cut Angle) ---
                # 击球方向(Cue->Ghost) 与 进球方向(Target->Pocket) 的夹角
                # 夹角越小越容易进，90度是理论极限（薄球），超过90度打不到
                cos_angle = np.dot(dir_cue_ghost, dir_target_pocket)
                # 限制在 -1 到 1 之间
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                cut_angle_rad = np.arccos(cos_angle)
                cut_angle_deg = np.degrees(cut_angle_rad)
                
                if cut_angle_deg > 85: # 角度太大，很难打进
                    continue
                    
                # --- D. 路径阻挡检测 (Obstruction Check) ---
                clearance_cue = self._min_clearance(cue_pos, ghost_pos, balls, exclude_ids=['cue', tid])
                if clearance_cue < self.clearance_threshold:
                    continue # 母球到幽灵球路径受阻
                    
                clearance_obj = self._min_clearance(target_pos, pocket_pos, balls, exclude_ids=['cue', tid])
                if clearance_obj < self.clearance_threshold:
                    continue # 目标球到袋口路径受阻
                
                # --- E. 评分 (Difficulty Score) ---
                # 角度越小越好，距离越近越好
                # 简单的启发式评分：角度权重 + 距离权重
                difficulty = cut_angle_deg + 10 * dist_cue_ghost + 5 * dist_target_pocket
                
                phi_rad = math.atan2(dir_cue_ghost[1], dir_cue_ghost[0])
                phi_deg = np.degrees(phi_rad)
                if phi_deg < 0:
                    phi_deg += 360

                required_speed = 1.0 + (dist_cue_ghost + dist_target_pocket) * 1.7
                required_speed = min(max(required_speed, 0.5), float(max_speed))

                candidates.append({
                    'difficulty': difficulty,
                    'cut_angle_deg': float(cut_angle_deg),
                    'dist_cue_ghost': float(dist_cue_ghost),
                    'dist_target_pocket': float(dist_target_pocket),
                    'clearance_total': float(clearance_cue + clearance_obj),
                    'action': {
                        'V0': required_speed,
                        'phi': float(phi_deg),
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0
                    }
                })
                if (not stage_is_eight) and int(len(stage_targets)) >= 4 and float(required_speed) <= float(max_speed - 0.8):
                    candidates.append({
                        'difficulty': float(difficulty) + 0.8,
                        'cut_angle_deg': float(cut_angle_deg),
                        'dist_cue_ghost': float(dist_cue_ghost),
                        'dist_target_pocket': float(dist_target_pocket),
                        'clearance_total': float(clearance_cue + clearance_obj),
                        'action': {
                            'V0': float(min(float(required_speed) + 1.0, float(max_speed))),
                            'phi': float(phi_deg),
                            'theta': 0.0,
                            'a': 0.0,
                            'b': 0.0
                        }
                    })
                    
        # 3. 返回最佳决策
        if candidates:
            candidates.sort(key=lambda c: c['difficulty'])
            top_k = candidates[:max(1, int(self.top_k_candidates))]
            best = self._select_by_monte_carlo(top_k, balls, my_targets, table)
            if best is not None:
                return best
        else:
            return self._safe_fallback_action(balls, my_targets, table, target_ids)

    def _is_obstructed(self, start_pos, end_pos, balls, exclude_ids):
        """
        检查从 start_pos 到 end_pos 的线段上是否有障碍球
        简化模型：将路径视为宽为 2*R 的矩形/胶囊体
        """
        vec = end_pos - start_pos
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return False
        dir_vec = vec / length
        
        for bid, ball in balls.items():
            if bid in exclude_ids:
                continue
            if getattr(getattr(ball, "state", None), "s", None) == 4:
                continue
            
            ball_pos = np.array(ball.state.rvw[0][:2])
            
            # 计算球心到线段的最短距离
            # 向量 start -> ball
            start_to_ball = ball_pos - start_pos
            
            # 投影长度
            proj = np.dot(start_to_ball, dir_vec)
            
            # 如果投影在线段之外，且距离端点较远，则忽略
            # 但这里为了安全，我们检查球心到直线的距离
            
            if proj < 0 or proj > length:
                # 球在路径前后，检查端点距离
                dist_to_start = np.linalg.norm(ball_pos - start_pos)
                dist_to_end = np.linalg.norm(ball_pos - end_pos)
                closest_dist = min(dist_to_start, dist_to_end)
            else:
                # 球在路径中间
                # 垂直距离
                perp_dist = np.linalg.norm(start_to_ball - proj * dir_vec)
                closest_dist = perp_dist
                
            # 判定阈值：两球半径之和 (2 * R)
            # 为了保险，稍微留点余量
            if closest_dist < (2 * self.ball_radius * 0.95):
                return True # 发生碰撞
                
        return False

    def _min_clearance(self, start_pos, end_pos, balls, exclude_ids):
        vec = end_pos - start_pos
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return float('inf')
        dir_vec = vec / length
        min_dist = float('inf')
        for bid, ball in balls.items():
            if bid in exclude_ids:
                continue
            if getattr(getattr(ball, "state", None), "s", None) == 4:
                continue
            ball_pos = np.array(ball.state.rvw[0][:2])
            start_to_ball = ball_pos - start_pos
            proj = np.dot(start_to_ball, dir_vec)
            if proj < 0:
                closest_dist = np.linalg.norm(ball_pos - start_pos)
            elif proj > length:
                closest_dist = np.linalg.norm(ball_pos - end_pos)
            else:
                closest_dist = np.linalg.norm(start_to_ball - proj * dir_vec)
            if closest_dist < min_dist:
                min_dist = float(closest_dist)
        return min_dist

    def _compute_stage_targets(self, balls, my_targets):
        targets = my_targets if my_targets else []
        remaining = []
        for bid in targets:
            if bid == '8':
                continue
            b = balls.get(bid)
            if b is None:
                continue
            if getattr(getattr(b, "state", None), "s", None) != 4:
                remaining.append(bid)
        if len(remaining) == 0:
            return ['8'], True
        return remaining, False

    def _perturb_action(self, action, noise_scale: float = 1.0, std_override=None):
        noise_scale = float(max(0.0, noise_scale))
        std = self.mc_noise_std if std_override is None else std_override
        noisy = {
            'V0': float(action['V0']) + float(np.random.normal(0, float(std['V0']) * noise_scale)),
            'phi': float(action['phi']) + float(np.random.normal(0, float(std['phi']) * noise_scale)),
            'theta': float(action.get('theta', 0.0)) + float(np.random.normal(0, float(std['theta']) * noise_scale)),
            'a': float(action.get('a', 0.0)) + float(np.random.normal(0, float(std['a']) * noise_scale)),
            'b': float(action.get('b', 0.0)) + float(np.random.normal(0, float(std['b']) * noise_scale)),
        }
        noisy['V0'] = float(np.clip(noisy['V0'], 0.5, 8.0))
        noisy['phi'] = float(noisy['phi'] % 360.0)
        noisy['theta'] = float(np.clip(noisy['theta'], 0.0, 90.0))
        noisy['a'] = float(np.clip(noisy['a'], -0.5, 0.5))
        noisy['b'] = float(np.clip(noisy['b'], -0.5, 0.5))
        return noisy

    def _minimize_legal8_speed(self, action, balls, my_targets, table, iters: int = 7):
        base_action = dict(action)
        r0, _cat0, ev0 = self._simulate_and_score(base_action, balls, my_targets, table)
        if ev0 != "LEGAL_8":
            return base_action

        hi = float(base_action.get("V0", 0.5))
        lo = 0.5
        best_v = hi
        for _ in range(int(max(1, iters))):
            mid = 0.5 * (lo + hi)
            a2 = dict(base_action)
            a2["V0"] = float(np.clip(mid, 0.5, 8.0))
            _r, _cat, ev = self._simulate_and_score(a2, balls, my_targets, table)
            if ev == "LEGAL_8":
                best_v = float(a2["V0"])
                hi = float(a2["V0"])
            else:
                lo = float(a2["V0"])
        base_action["V0"] = float(np.clip(best_v, 0.5, 8.0))
        return base_action

    def _simulate_and_score(self, action, balls, my_targets, table):
        stage_targets, stage_is_eight = self._compute_stage_targets(balls, my_targets)
        remaining_n = int(len(stage_targets)) if (not stage_is_eight) else 0
        balls_before_s = {bid: int(getattr(ball.state, "s", -1)) for bid, ball in balls.items()}
        eight_before_pos = None
        if "8" in balls and getattr(getattr(balls["8"], "state", None), "s", None) != 4:
            eight_before_pos = np.array(balls["8"].state.rvw[0][:2])
        balls_sim = copy.deepcopy(balls)
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=balls_sim, cue=cue)
        try:
            cue.set_state(V0=action["V0"], phi=action["phi"], theta=action["theta"], a=action['a'], b=action['b'])
            pt.simulate(shot, inplace=True)
        except Exception:
            return -8000.0, True, "SIM_FAIL"

        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls_before_s.get(bid, -1) != 4]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        eight_after_pos = None
        if (not eight_pocketed) and ("8" in shot.balls) and getattr(getattr(shot.balls["8"], "state", None), "s", None) != 4:
            eight_after_pos = np.array(shot.balls["8"].state.rvw[0][:2])

        if cue_pocketed and eight_pocketed:
            return -10000.0, True, "CUE_AND_8"
        if eight_pocketed and (not stage_is_eight):
            return -9000.0, True, "ILLEGAL_8"
        if cue_pocketed:
            return -6000.0, True, "SCRATCH"
        if stage_is_eight and eight_pocketed:
            return 8000.0, False, "LEGAL_8"

        if stage_is_eight:
            own_pocketed = ["8"] if eight_pocketed else []
        else:
            own_pocketed = [bid for bid in new_pocketed if bid in stage_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in stage_targets and bid not in ["cue", "8"]]

        eight_risk = False
        eight_risk_penalty = 0.0
        if (not stage_is_eight) and (eight_before_pos is not None) and (eight_after_pos is not None):
            eight_disp = float(np.linalg.norm(eight_after_pos - eight_before_pos))
            dmin_before = None
            dmin_after = None
            for _pid, pocket in table.pockets.items():
                ppos = np.array(pocket.center[:2])
                db = float(np.linalg.norm(ppos - eight_before_pos))
                da = float(np.linalg.norm(ppos - eight_after_pos))
                if dmin_before is None or db < dmin_before:
                    dmin_before = db
                if dmin_after is None or da < dmin_after:
                    dmin_after = da
            if (dmin_after is not None) and (dmin_before is not None):
                if (dmin_after <= float(self.eight_near_pocket_d2)) and (eight_disp >= 0.006):
                    eight_risk = True
                    eight_risk_penalty = max(eight_risk_penalty, 2800.0)
                if (dmin_before - dmin_after) >= 0.05 and eight_disp >= 0.01:
                    eight_risk = True
                    eight_risk_penalty = max(eight_risk_penalty, 2400.0)

        events = shot.events
        first_contact_ball_id = None
        cue_touched_eight = False
        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
        for e in events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break

        cue_hit_cushion = False
        target_hit_cushion = False
        for e in events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids) and ('8' in ids):
                cue_touched_eight = True
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True

        no_hit = first_contact_ball_id is None
        foul_first_hit = False
        if not no_hit:
            if stage_is_eight:
                foul_first_hit = first_contact_ball_id != '8'
            else:
                foul_first_hit = first_contact_ball_id not in set(stage_targets)

        no_pocket_no_rail = (len(new_pocketed) == 0) and (not cue_hit_cushion) and (not target_hit_cushion)

        catastrophic = False
        score = 0.0
        event_tag = "OK"

        if no_hit:
            return -5000.0, True, "NO_HIT"
        if foul_first_hit:
            return -4500.0, True, "FOUL_FIRST_HIT"
        if no_pocket_no_rail:
            catastrophic = True
            score -= 2200.0

        own_reward = 220.0
        if (not stage_is_eight) and remaining_n >= 4:
            own_reward = 260.0
        elif (not stage_is_eight) and remaining_n == 3:
            own_reward = 240.0
        score += float(own_reward) * float(len(own_pocketed))
        enemy_n = int(len(enemy_pocketed))
        if enemy_n > 0:
            score -= float(self.enemy_pocket_base_penalty) * float(enemy_n)
            if enemy_n >= 2:
                score -= float(self.enemy_pocket_multi_penalty) * float(enemy_n - 1)

        if (not stage_is_eight) and (eight_before_pos is not None) and (eight_after_pos is not None):
            eight_disp = float(np.linalg.norm(eight_after_pos - eight_before_pos))
            score -= float(self.eight_move_penalty) * eight_disp

        if (not stage_is_eight) and eight_risk:
            remaining_n = int(len(stage_targets))
            risk_pen = float(eight_risk_penalty)
            if remaining_n <= 2:
                risk_pen *= 1.4
            score -= risk_pen
            event_tag = "EIGHT_RISK"

        if (not stage_is_eight) and cue_touched_eight:
            touch_pen = float(self.cue8_touch_score_penalty)
            remaining_n = int(len(stage_targets))
            if remaining_n <= 2:
                touch_pen *= 2.0
            if eight_before_pos is not None:
                dmin_before = None
                for _pid, pocket in table.pockets.items():
                    ppos = np.array(pocket.center[:2])
                    db = float(np.linalg.norm(ppos - eight_before_pos))
                    if dmin_before is None or db < dmin_before:
                        dmin_before = db
                if dmin_before is not None and dmin_before <= float(self.eight_near_pocket_d2):
                    touch_pen *= 2.0
            score -= touch_pen
            event_tag = "CUE_TOUCH_8"

        v0 = float(action.get("V0", 0.0))
        if not stage_is_eight:
            thr = 3.0
            coef = float(self.speed_penalty_coef)
            if remaining_n >= 4:
                thr = 4.5
                coef = 10.0
            elif remaining_n == 3:
                thr = 3.8
                coef = 18.0
            if v0 > float(thr):
                dv = float(v0 - float(thr))
                score -= float(coef) * dv * dv

        if (len(new_pocketed) == 0) and (not no_hit) and (not foul_first_hit) and (not no_pocket_no_rail) and (not cue_pocketed) and (not eight_pocketed):
            safe_bonus = 40.0
            if (not stage_is_eight) and remaining_n >= 4:
                safe_bonus = 0.0
            elif (not stage_is_eight) and remaining_n == 3:
                safe_bonus = 15.0
            score += float(safe_bonus)

        if no_pocket_no_rail:
            return score, catastrophic, "NO_POCKET_NO_RAIL"
        return score, catastrophic, event_tag

    def _mc_evaluate_action(self, base_action, balls, my_targets, table):
        scores = []
        cat_cnt = 0
        best_sample_action = base_action
        best_sample_score = -float('inf')
        for _ in range(int(self.mc_samples)):
            a = self._perturb_action(base_action, std_override=self.robust_noise_std)
            s, catastrophic, _event = self._simulate_and_score(a, balls, my_targets, table)
            scores.append(float(s))
            if catastrophic:
                cat_cnt += 1
            if s > best_sample_score:
                best_sample_score = float(s)
                best_sample_action = a

        mean = float(np.mean(scores)) if scores else -float('inf')
        std = float(np.std(scores)) if scores else 0.0
        cat_rate = float(cat_cnt) / float(len(scores)) if scores else 1.0
        agg = mean - self.mc_std_coef * std - self.mc_cat_coef * cat_rate
        return agg, best_sample_action

    def _passes_black_safety(self, action, balls, my_targets, table, stage_is_eight, samples_override=None):
        stage_targets = None
        remaining_n = None
        eight_near = False
        if not stage_is_eight:
            stage_targets, _stage_is_eight = self._compute_stage_targets(balls, my_targets)
            remaining_n = int(len(stage_targets))

        v0 = float(action.get("V0", 0.0))
        if (not stage_is_eight) and ("8" in balls) and getattr(getattr(balls["8"], "state", None), "s", None) != 4:
            eight_pos = np.array(balls["8"].state.rvw[0][:2])
            dmin = None
            for _pid, pocket in table.pockets.items():
                ppos = np.array(pocket.center[:2])
                d = float(np.linalg.norm(ppos - eight_pos))
                if dmin is None or d < dmin:
                    dmin = d
            if dmin is not None and dmin <= float(self.eight_near_pocket_d2):
                eight_near = True

        base_samples = int(self.black_safety_samples)
        if stage_is_eight:
            base_samples = max(base_samples, 48)
            if v0 >= 4.0:
                base_samples = max(base_samples, 72)
        else:
            if remaining_n is not None and remaining_n <= 1:
                base_samples = max(base_samples, 72)
            elif remaining_n is not None and remaining_n <= 2:
                base_samples = max(base_samples, 56)
            if eight_near:
                base_samples = max(base_samples, 56)
            if v0 >= 5.5:
                base_samples = max(base_samples, 56)
            elif v0 >= 4.0:
                base_samples = max(base_samples, 40)
            else:
                base_samples = min(base_samples, 16)

        samples = int(min(int(self.black_safety_max_samples), base_samples))
        if samples_override is not None:
            samples = int(max(0, min(int(samples), int(samples_override))))
        if samples <= 0:
            return True

        _r0, _cat0, event0 = self._simulate_and_score(action, balls, my_targets, table)
        if event0 in ("CUE_AND_8", "ILLEGAL_8"):
            return False
        if stage_is_eight and event0 != "LEGAL_8":
            return False
        if (not stage_is_eight) and event0 == "EIGHT_RISK":
            if eight_near or (remaining_n is not None and remaining_n <= 2):
                return False

        scratch_rate_max = 0.05 if stage_is_eight else 0.0
        legal8_rate_min = 0.45 if stage_is_eight else 0.0

        samples_target = int(samples)
        if stage_is_eight and event0 == "LEGAL_8":
            samples_target = int(max(samples_target, 96))
        if samples_override is not None:
            samples_target = int(max(0, min(samples_target, int(samples_override))))
        samples_target = int(min(int(self.black_safety_max_samples), samples_target))

        cue8_cnt = 0
        legal8_cnt = 0
        for _ in range(samples_target):
            noise_scale = float(self.black_safety_noise_scale)
            if stage_is_eight:
                noise_scale = max(noise_scale, 2.0)
            a = self._perturb_action(action, noise_scale=noise_scale, std_override=self.robust_noise_std)
            _r, _catastrophic, event = self._simulate_and_score(a, balls, my_targets, table)

            if stage_is_eight:
                if event == "CUE_AND_8":
                    cue8_cnt += 1
                elif event == "LEGAL_8":
                    legal8_cnt += 1
                continue

            if event == "CUE_AND_8":
                return False
            if event == "ILLEGAL_8":
                return False
            if event == "EIGHT_RISK":
                if eight_near or (remaining_n is not None and remaining_n <= 2):
                    return False

        if stage_is_eight and samples_target > 0:
            cue8_rate = float(cue8_cnt) / float(samples_target)
            legal8_rate = float(legal8_cnt) / float(samples_target)
            if cue8_rate > float(scratch_rate_max):
                return False
            if legal8_rate < float(legal8_rate_min):
                return False
        return True

    def _select_by_monte_carlo(self, candidates, balls, my_targets, table):
        if not candidates:
            return None

        cand_eval = candidates[:max(1, int(self.mc_top_k))]
        _stage_targets, stage_is_eight = self._compute_stage_targets(balls, my_targets)
        n_candidates = int(len(cand_eval))
        if n_candidates <= 0:
            candidates.sort(key=lambda c: c['difficulty'])
            return candidates[0]['action']

        remaining_n = 0
        if not stage_is_eight:
            stage_targets, _stage_is_eight = self._compute_stage_targets(balls, my_targets)
            remaining_n = int(len(stage_targets))
        eight_near = False
        if (not stage_is_eight) and ("8" in balls) and getattr(getattr(balls["8"], "state", None), "s", None) != 4:
            eight_pos = np.array(balls["8"].state.rvw[0][:2])
            dmin = None
            for _pid, pocket in table.pockets.items():
                ppos = np.array(pocket.center[:2])
                d = float(np.linalg.norm(ppos - eight_pos))
                if dmin is None or d < dmin:
                    dmin = d
            if dmin is not None and dmin <= float(self.eight_near_pocket_d2):
                eight_near = True

        key_stage = bool(stage_is_eight or (remaining_n > 0 and remaining_n <= 2) or eight_near)
        mc_samples = int(self.mc_samples)
        mc_budget = int(self.mc_sim_budget)
        if key_stage:
            mc_samples = max(mc_samples, 7)
            mc_budget = max(mc_budget, 60)
        else:
            mc_samples = max(2, min(mc_samples, 4))
            mc_budget = max(16, min(mc_budget, 24))

        total_sims = max(1, int(mc_samples) * n_candidates)
        total_sims = int(min(int(total_sims), int(mc_budget)))
        N = np.zeros(n_candidates, dtype=np.int64)
        mean = np.zeros(n_candidates, dtype=np.float64)
        M2 = np.zeros(n_candidates, dtype=np.float64)
        cat_cnt = np.zeros(n_candidates, dtype=np.int64)
        illegal8_cnt = np.zeros(n_candidates, dtype=np.int64)
        cue8_cnt = np.zeros(n_candidates, dtype=np.int64)
        cue8_touch_cnt = np.zeros(n_candidates, dtype=np.int64)
        best_action_per = [cand_eval[i]['action'] for i in range(n_candidates)]
        best_score_per = [-float('inf') for _ in range(n_candidates)]

        for i in range(n_candidates):
            base_action = cand_eval[i]["action"]
            a0 = self._perturb_action(base_action, std_override=self.robust_noise_std)
            r0, _cat0, ev0 = self._simulate_and_score(a0, balls, my_targets, table)
            N[i] = 1
            mean[i] = float(r0)
            M2[i] = 0.0
            if ev0 == "CUE_AND_8":
                cue8_cnt[i] += 1
            if ev0 == "ILLEGAL_8":
                illegal8_cnt[i] += 1
            if ev0 == "CUE_TOUCH_8":
                cue8_touch_cnt[i] += 1
            if float(r0) > float(best_score_per[i]):
                best_score_per[i] = float(r0)
                best_action_per[i] = a0

        for t in range(total_sims):
            if t < n_candidates:
                idx = t
            else:
                total_n = float(np.sum(N))
                denom = N.astype(np.float64) + 1e-6
                ucb = mean + float(self.mcts_c) * np.sqrt(np.log(total_n + 1.0) / denom)
                idx = int(np.argmax(ucb))

            base_action = cand_eval[idx]['action']
            a = self._perturb_action(base_action, std_override=self.robust_noise_std)
            r, catastrophic, event = self._simulate_and_score(a, balls, my_targets, table)

            N[idx] += 1
            delta = float(r) - float(mean[idx])
            mean[idx] += delta / float(N[idx])
            delta2 = float(r) - float(mean[idx])
            M2[idx] += delta * delta2

            if catastrophic:
                cat_cnt[idx] += 1
            if event == "CUE_AND_8":
                cue8_cnt[idx] += 1
            if event == "ILLEGAL_8":
                illegal8_cnt[idx] += 1
            if event == "CUE_TOUCH_8":
                cue8_touch_cnt[idx] += 1
            if float(r) > float(best_score_per[idx]):
                best_score_per[idx] = float(r)
                best_action_per[idx] = a

        best_idx = None
        best_agg = -float('inf')
        has_feasible = False
        agg_by_idx = {}
        for i in range(n_candidates):
            if N[i] <= 0:
                continue
            if (not stage_is_eight) and cue8_cnt[i] > 0:
                continue
            if (not stage_is_eight) and illegal8_cnt[i] > 0:
                continue
            has_feasible = True
            var = float(M2[i]) / float(N[i])
            std = float(math.sqrt(max(var, 0.0)))
            cat_rate = float(cat_cnt[i]) / float(N[i])
            touch_rate = float(cue8_touch_cnt[i]) / float(N[i])
            agg = (
                float(mean[i])
                - float(self.mc_std_coef) * std
                - float(self.mc_cat_coef) * cat_rate
                - float(self.cue8_touch_mc_penalty) * touch_rate
            )
            agg_by_idx[i] = agg
            if agg > best_agg:
                best_agg = agg
                best_idx = i

        if best_idx is not None:
            for i, _agg in sorted(agg_by_idx.items(), key=lambda kv: kv[1], reverse=True):
                action = best_action_per[int(i)]
                if self._passes_black_safety(action, balls, my_targets, table, stage_is_eight):
                    return action
            base = best_action_per[int(best_idx)]
            if self._passes_black_safety(base, balls, my_targets, table, stage_is_eight):
                return base
            for scale in (0.85, 0.7, 0.55):
                a2 = dict(base)
                a2["V0"] = float(np.clip(float(a2.get("V0", 0.0)) * float(scale), 0.5, 8.0))
                if self._passes_black_safety(a2, balls, my_targets, table, stage_is_eight):
                    return a2
            if not stage_is_eight:
                return self._safe_fallback_action(balls, my_targets, table, _stage_targets)
            return base

        if has_feasible:
            candidates.sort(key=lambda c: c['difficulty'])
            a = candidates[0]['action']
            if self._passes_black_safety(a, balls, my_targets, table, stage_is_eight):
                return a
            if not stage_is_eight:
                return self._safe_fallback_action(balls, my_targets, table, _stage_targets)
            return a

        best_idx = None
        best_badness = float("inf")
        best_agg = -float("inf")
        for i in range(n_candidates):
            if N[i] <= 0:
                continue
            badness = float(cue8_cnt[i] + illegal8_cnt[i]) / float(N[i])
            var = float(M2[i]) / float(N[i])
            std = float(math.sqrt(max(var, 0.0)))
            cat_rate = float(cat_cnt[i]) / float(N[i])
            touch_rate = float(cue8_touch_cnt[i]) / float(N[i])
            agg = (
                float(mean[i])
                - float(self.mc_std_coef) * std
                - float(self.mc_cat_coef) * cat_rate
                - float(self.cue8_touch_mc_penalty) * touch_rate
            )
            if (badness < best_badness) or (badness == best_badness and agg > best_agg):
                best_badness = badness
                best_agg = agg
                best_idx = i
        if best_idx is not None:
            base = best_action_per[int(best_idx)]
            if self._passes_black_safety(base, balls, my_targets, table, stage_is_eight):
                return base
            if not stage_is_eight:
                return self._safe_fallback_action(balls, my_targets, table, _stage_targets)
            return base

        candidates.sort(key=lambda c: c['difficulty'])
        a = candidates[0]['action']
        if self._passes_black_safety(a, balls, my_targets, table, stage_is_eight):
            return a
        if not stage_is_eight:
            return self._safe_fallback_action(balls, my_targets, table, _stage_targets)
        return a

    def _gentle_contact_candidates(self, balls, target_ids):
        if 'cue' not in balls:
            return []
        cue_pos = np.array(balls['cue'].state.rvw[0][:2])
        actions = []
        for tid in target_ids:
            b = balls.get(tid)
            if b is None:
                continue
            if getattr(getattr(b, "state", None), "s", None) == 4:
                continue
            target_pos = np.array(b.state.rvw[0][:2])
            direction = target_pos - cue_pos
            if float(np.linalg.norm(direction)) < 1e-6:
                continue
            phi_rad = math.atan2(direction[1], direction[0])
            phi_deg = float(np.degrees(phi_rad))
            if phi_deg < 0:
                phi_deg += 360.0
            for v0 in (0.9, 1.2, 1.5, 1.8, 2.1, 2.4):
                for dphi in (0.0, -0.8, 0.8, -1.6, 1.6):
                    actions.append({
                        'V0': float(v0),
                        'phi': float((phi_deg + dphi) % 360.0),
                        'theta': 0.0,
                        'a': 0.0,
                        'b': 0.0
                    })
        random.shuffle(actions)
        return actions

    def _simulate_fallback_features(self, action, balls, my_targets, table, stage_targets, stage_is_eight):
        balls_before_s = {bid: int(getattr(ball.state, "s", -1)) for bid, ball in balls.items()}
        eight_before_pos = None
        if (not stage_is_eight) and ("8" in balls) and getattr(getattr(balls["8"], "state", None), "s", None) != 4:
            eight_before_pos = np.array(balls["8"].state.rvw[0][:2])

        balls_sim = copy.deepcopy(balls)
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=balls_sim, cue=cue)
        try:
            cue.set_state(V0=action["V0"], phi=action["phi"], theta=action["theta"], a=action['a'], b=action['b'])
            pt.simulate(shot, inplace=True)
        except Exception:
            return None

        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls_before_s.get(bid, -1) != 4]
        if ("cue" in new_pocketed) or ("8" in new_pocketed):
            return None

        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
        first_contact_ball_id = None
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break

        if first_contact_ball_id is None:
            return None

        if stage_is_eight:
            if first_contact_ball_id != "8":
                return None
        else:
            if first_contact_ball_id not in set(stage_targets):
                return None

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

        no_pocket_no_rail = (len(new_pocketed) == 0) and (not cue_hit_cushion) and (not target_hit_cushion)
        if no_pocket_no_rail:
            return None

        cue_pos_after = np.array(shot.balls["cue"].state.rvw[0][:2])

        enemy_min_dist = None
        for bid, b in shot.balls.items():
            if bid in ("cue", "8"):
                continue
            if bid not in valid_ball_ids:
                continue
            if getattr(getattr(b, "state", None), "s", None) == 4:
                continue
            if (not stage_is_eight) and (bid in set(stage_targets)):
                continue
            pos = np.array(b.state.rvw[0][:2])
            d = float(np.linalg.norm(pos - cue_pos_after))
            if enemy_min_dist is None or d < enemy_min_dist:
                enemy_min_dist = d
        if enemy_min_dist is None:
            enemy_min_dist = 0.0

        pocket_min_dist = None
        for _pid, pocket in table.pockets.items():
            ppos = np.array(pocket.center[:2])
            d = float(np.linalg.norm(ppos - cue_pos_after))
            if pocket_min_dist is None or d < pocket_min_dist:
                pocket_min_dist = d
        if pocket_min_dist is None:
            pocket_min_dist = 0.0

        rail_dist = None
        tx = float(cue_pos_after[0])
        ty = float(cue_pos_after[1])
        if hasattr(table, "l") and hasattr(table, "w"):
            rail_dist = float(min(tx, float(table.l) - tx, ty, float(table.w) - ty))
        if rail_dist is None:
            rail_dist = 0.0

        own_pocketed = ["8"] if stage_is_eight else [bid for bid in new_pocketed if bid in set(stage_targets)]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in set(stage_targets) and bid not in ["cue", "8"]]

        eight_delta_min_pocket = 0.0
        if (not stage_is_eight) and (eight_before_pos is not None) and ("8" in shot.balls) and getattr(getattr(shot.balls["8"], "state", None), "s", None) != 4:
            eight_after_pos = np.array(shot.balls["8"].state.rvw[0][:2])
            dmin_before = None
            dmin_after = None
            for _pid, pocket in table.pockets.items():
                ppos = np.array(pocket.center[:2])
                db = float(np.linalg.norm(ppos - eight_before_pos))
                da = float(np.linalg.norm(ppos - eight_after_pos))
                if dmin_before is None or db < dmin_before:
                    dmin_before = db
                if dmin_after is None or da < dmin_after:
                    dmin_after = da
            if (dmin_before is not None) and (dmin_after is not None):
                eight_delta_min_pocket = float(dmin_after - dmin_before)

        return {
            "enemy_min_dist": float(enemy_min_dist),
            "pocket_min_dist": float(pocket_min_dist),
            "rail_dist": float(rail_dist),
            "own_pocketed_n": int(len(own_pocketed)),
            "enemy_pocketed_n": int(len(enemy_pocketed)),
            "eight_delta_min_pocket": float(eight_delta_min_pocket),
        }

    def _fallback_defense_score(self, feats):
        enemy_min_dist = float(feats.get("enemy_min_dist", 0.0))
        pocket_min_dist = float(feats.get("pocket_min_dist", 0.0))
        rail_dist = float(feats.get("rail_dist", 0.0))
        enemy_pocketed_n = int(feats.get("enemy_pocketed_n", 0))
        eight_delta_min_pocket = float(feats.get("eight_delta_min_pocket", 0.0))

        rail_bonus = 0.0
        if rail_dist > 0.0:
            rail_bonus = float(max(0.0, 0.11 - rail_dist) / 0.11)

        score = 0.0
        score += 600.0 * enemy_min_dist
        score += 420.0 * pocket_min_dist
        score += 220.0 * rail_bonus
        score -= 900.0 * float(enemy_pocketed_n)
        if eight_delta_min_pocket < 0.0:
            score += 1200.0 * eight_delta_min_pocket
        return float(score)

    def _safe_fallback_action(self, balls, my_targets, table, target_ids):
        _stage_targets, stage_is_eight = self._compute_stage_targets(balls, my_targets)
        if not stage_is_eight:
            stage_targets = list(target_ids)
            cand = []
            base_phi_actions = self._gentle_contact_candidates(balls, stage_targets)
            for a0 in base_phi_actions[:48]:
                r0, cat0, ev0 = self._simulate_and_score(a0, balls, my_targets, table)
                if ev0 in ("NO_HIT", "FOUL_FIRST_HIT", "NO_POCKET_NO_RAIL", "CUE_AND_8", "ILLEGAL_8", "EIGHT_RISK"):
                    continue
                proxy = float(r0) - (2000.0 if cat0 else 0.0) - 160.0 * float(a0.get("V0", 0.0)) - (800.0 if ev0 == "CUE_TOUCH_8" else 0.0)
                cand.append((proxy, a0))
            cand.sort(key=lambda x: x[0], reverse=True)
            cand = cand[:8]

            scored = []
            for _proxy, a0 in cand:
                feats = self._simulate_fallback_features(a0, balls, my_targets, table, stage_targets=stage_targets, stage_is_eight=stage_is_eight)
                if feats is None:
                    continue
                scored.append((self._fallback_defense_score(feats), a0))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _s, a0 in scored[:8]:
                if self._passes_black_safety(a0, balls, my_targets, table, stage_is_eight, samples_override=12):
                    return a0

        for a0 in self._gentle_contact_candidates(balls, target_ids):
            _r0, _cat0, ev0 = self._simulate_and_score(a0, balls, my_targets, table)
            if ev0 in ("NO_HIT", "FOUL_FIRST_HIT", "NO_POCKET_NO_RAIL"):
                continue
            if self._passes_black_safety(a0, balls, my_targets, table, stage_is_eight, samples_override=8):
                return a0
        base = self._random_action_towards_ball(balls, target_ids)
        if self._passes_black_safety(base, balls, my_targets, table, stage_is_eight, samples_override=8):
            return base
        for scale in (0.85, 0.7, 0.55, 0.4):
            a2 = dict(base)
            a2["V0"] = float(np.clip(float(a2.get("V0", 0.0)) * float(scale), 0.5, 8.0))
            if self._passes_black_safety(a2, balls, my_targets, table, stage_is_eight, samples_override=8):
                return a2
        for _ in range(24):
            a3 = self._random_action_towards_ball(balls, target_ids)
            a3["V0"] = float(np.clip(float(a3.get("V0", 0.0)), 0.5, 2.6))
            if self._passes_black_safety(a3, balls, my_targets, table, stage_is_eight):
                return a3
        for _ in range(40):
            a4 = self._random_action_towards_ball(balls, target_ids)
            a4["V0"] = float(np.clip(float(a4.get("V0", 0.0)), 0.5, 2.0))
            a4["theta"] = 0.0
            a4["a"] = 0.0
            a4["b"] = 0.0
            _r0, _cat0, ev0 = self._simulate_and_score(a4, balls, my_targets, table)
            if ev0 in ("NO_HIT", "FOUL_FIRST_HIT"):
                continue
            if self._passes_black_safety(a4, balls, my_targets, table, stage_is_eight):
                return a4
        return base

    def _random_action_towards_ball(self, balls, target_ids):
        """朝某个目标球打一杆（保底策略）"""
        if 'cue' not in balls:
            return self._random_action()
            
        cue_pos = np.array(balls['cue'].state.rvw[0][:2])
        
        # 找最近的目标球
        min_dist = float('inf')
        target_pos = None
        
        for tid in target_ids:
            if tid in balls:
                pos = np.array(balls[tid].state.rvw[0][:2])
                dist = np.linalg.norm(pos - cue_pos)
                if dist < min_dist:
                    min_dist = dist
                    target_pos = pos
                    
        if target_pos is not None:
            direction = target_pos - cue_pos
            phi_rad = math.atan2(direction[1], direction[0])
            phi_deg = np.degrees(phi_rad)
            if phi_deg < 0: phi_deg += 360
            
            return {
                'V0': random.uniform(1.6, 3.2),
                'phi': phi_deg, # 加一点随机扰动？
                'theta': 0,
                'a': 0,
                'b': 0
            }
        else:
            return self._random_action()

    def _random_action(self):
        import random
        return {
            'V0': random.uniform(0.5, 8.0),
            'phi': random.uniform(0, 360),
            'theta': random.uniform(0, 90),
            'a': random.uniform(-0.5, 0.5),
            'b': random.uniform(-0.5, 0.5)
        }
