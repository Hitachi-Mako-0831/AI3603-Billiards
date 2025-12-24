"""
curriculum_env.py - 课程学习环境包装器

实现渐进式难度的台球训练环境：
- Level 1: 白球 + 1个目标球（学习基本击球）
- Level 2: 白球 + 3个目标球（学习选择目标）
- Level 3: 白球 + 5个目标球（接近半场）
- Level 4: 白球 + 7个目标球（完整半场）
- Level 5: 完整游戏（所有球）

核心思路：在reset时移除部分球，简化学习任务
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import random
from typing import List, Dict, Tuple, Optional


def save_balls_state(balls):
    """保存球状态（深拷贝）"""
    return {bid: copy.deepcopy(ball) for bid, ball in balls.items()}


def restore_balls_state(saved_state):
    """恢复球状态（深拷贝）"""
    return {bid: copy.deepcopy(ball) for bid, ball in saved_state.items()}


class CurriculumPoolEnv:
    """
    课程学习台球环境
    
    支持不同难度级别，通过控制球的数量来实现渐进式学习
    """
    
    # 课程难度配置
    CURRICULUM_LEVELS = {
        1: {'num_targets': 1, 'num_enemies': 0, 'description': '1个目标球'},
        2: {'num_targets': 3, 'num_enemies': 0, 'description': '3个目标球'},
        3: {'num_targets': 5, 'num_enemies': 2, 'description': '5个目标球+2个干扰球'},
        4: {'num_targets': 7, 'num_enemies': 4, 'description': '7个目标球+4个干扰球'},
        5: {'num_targets': 7, 'num_enemies': 7, 'description': '完整游戏'},
    }
    
    def __init__(self, verbose: bool = False, record_shots: bool = False, single_player_mode: bool = True):
        """初始化环境
        
        Args:
            verbose: 是否打印详细日志
            record_shots: 是否记录每次击球
            single_player_mode: 单人训练模式（不切换玩家，所有击球都是A玩家）
        """
        self.table = None
        self.balls = None
        self.cue = None
        
        self.verbose = verbose
        self.record_shots = record_shots
        self.single_player_mode = single_player_mode  # 单人模式标志
        
        self.player_targets = None
        self.hit_count = 0
        self.last_state = None
        self.players = ["A", "B"]
        self.curr_player = 0
        self.done = False
        self.winner = None
        self.MAX_HIT_COUNT = 30  # 课程学习用更短的回合
        self.shot_record = pt.MultiSystem()
        
        # 噪声配置（课程学习初期可以关闭）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        # 课程学习配置
        self.curriculum_level = 1
        self.episode_count = 0
        self.success_history = []  # 记录最近N局的成功率
        self.history_window = 50
        self.promotion_threshold = 0.3  # 30%成功率升级
        self.auto_curriculum = True  # 是否自动升级难度
        
        # 统计信息
        self.level_stats = {i: {'episodes': 0, 'successes': 0} for i in range(1, 6)}
    
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
    
    def set_curriculum_level(self, level: int):
        """手动设置课程难度级别"""
        if level < 1 or level > 5:
            raise ValueError(f"Invalid curriculum level: {level}. Must be 1-5.")
        self.curriculum_level = level
        self._log(f"📚 课程难度设置为 Level {level}: {self.CURRICULUM_LEVELS[level]['description']}")
    
    def get_curriculum_level(self) -> int:
        """获取当前课程难度"""
        return self.curriculum_level
    
    def _check_promotion(self, success: bool):
        """检查是否需要升级难度"""
        self.success_history.append(1 if success else 0)
        if len(self.success_history) > self.history_window:
            self.success_history.pop(0)
        
        # 更新统计
        self.level_stats[self.curriculum_level]['episodes'] += 1
        if success:
            self.level_stats[self.curriculum_level]['successes'] += 1
        
        if not self.auto_curriculum:
            return
        
        # 检查是否达到升级条件
        if len(self.success_history) >= self.history_window:
            success_rate = sum(self.success_history) / len(self.success_history)
            if success_rate >= self.promotion_threshold and self.curriculum_level < 5:
                old_level = self.curriculum_level
                self.curriculum_level += 1
                self.success_history = []  # 重置历史
                self._log(f"🎉 升级！Level {old_level} -> Level {self.curriculum_level} "
                         f"(成功率: {success_rate*100:.1f}%)")
    
    def _setup_curriculum_balls(self, target_ball: str):
        """根据课程难度设置球的配置"""
        level_config = self.CURRICULUM_LEVELS[self.curriculum_level]
        num_targets = level_config['num_targets']
        num_enemies = level_config['num_enemies']
        
        # 确定目标球和敌方球的ID
        if target_ball == 'solid':
            all_targets = [str(i) for i in range(1, 8)]  # 1-7
            all_enemies = [str(i) for i in range(9, 16)]  # 9-15
        else:  # stripe
            all_targets = [str(i) for i in range(9, 16)]  # 9-15
            all_enemies = [str(i) for i in range(1, 8)]  # 1-7
        
        # 随机选择要保留的球
        random.shuffle(all_targets)
        random.shuffle(all_enemies)
        
        keep_targets = all_targets[:num_targets]
        keep_enemies = all_enemies[:num_enemies]
        keep_balls = set(['cue', '8'] + keep_targets + keep_enemies)
        
        # 移除不需要的球（将其标记为已进袋）
        for bid, ball in self.balls.items():
            if bid not in keep_balls:
                # 将球移到袋外并标记为进袋状态
                ball.state.s = 4  # 进袋状态
                ball.state.rvw[0] = np.array([-1.0, -1.0, 0.0])  # 移到场外
        
        # 随机化保留球的位置，避免固定模式
        self._randomize_ball_positions(keep_targets + keep_enemies)
        
        return keep_targets
    
    def _randomize_ball_positions(self, ball_ids: List[str]):
        """随机化球的位置，使训练更多样化"""
        if not ball_ids:
            return
        
        table_l = self.table.l if self.table else 1.98
        table_w = self.table.w if self.table else 0.99
        
        # 安全边距
        margin = 0.08
        ball_radius = 0.028575  # 标准台球半径
        
        # 获取白球位置
        cue_pos = self.balls['cue'].state.rvw[0][:2]
        
        # 为每个球生成不重叠的随机位置
        placed_positions = [cue_pos]
        
        for bid in ball_ids:
            ball = self.balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            
            # 尝试找到一个不重叠的位置
            placed = False
            for attempt in range(100):
                x = random.uniform(margin, table_l - margin)
                y = random.uniform(margin, table_w - margin)
                
                # 检查与已放置球的距离
                valid = True
                for px, py in placed_positions:
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    if dist < ball_radius * 3:  # 至少3倍球径距离
                        valid = False
                        break
                
                if valid:
                    ball.state.rvw[0] = np.array([x, y, ball_radius])
                    ball.state.rvw[1] = np.array([0.0, 0.0, 0.0])  # 静止
                    ball.state.rvw[2] = np.array([0.0, 0.0, 0.0])
                    ball.state.s = 0  # 静止状态
                    placed_positions.append((x, y))
                    placed = True
                    break
            
            # 如果100次都失败，使用降低标准的后备方案
            if not placed:
                for attempt in range(50):
                    x = random.uniform(margin, table_l - margin)
                    y = random.uniform(margin, table_w - margin)
                    valid = True
                    for px, py in placed_positions:
                        dist = math.sqrt((x - px)**2 + (y - py)**2)
                        if dist < ball_radius * 2.5:  # 降低到2.5倍
                            valid = False
                            break
                    if valid:
                        ball.state.rvw[0] = np.array([x, y, ball_radius])
                        ball.state.rvw[1] = np.array([0.0, 0.0, 0.0])
                        ball.state.rvw[2] = np.array([0.0, 0.0, 0.0])
                        ball.state.s = 0
                        placed_positions.append((x, y))
                        break
    
    def get_observation(self, player=None, copy_state: bool = True):
        """获取观测"""
        if player is None:
            player = self.get_curr_player()
        if copy_state:
            return copy.deepcopy(self.balls), self.player_targets[player], copy.deepcopy(self.table)
        return self.balls, self.player_targets[player], self.table
    
    def get_curr_player(self) -> str:
        return self.players[self.curr_player]
    
    def get_done(self) -> Tuple[bool, dict]:
        if self.done:
            return True, {'winner': self.winner, 'hit_count': self.hit_count}
        return False, {}
    
    def reset(self, state=None, target_ball: str = None):
        """重置环境"""
        if state is not None:
            raise NotImplementedError("不支持恢复到指定state")
        
        if target_ball not in ['solid', 'stripe']:
            target_ball = random.choice(['solid', 'stripe'])
        
        # 初始化标准台球布局
        self.table = pt.Table.default()
        self.balls = pt.get_rack(pt.GameType.EIGHTBALL, self.table)
        self.cue = pt.Cue(cue_ball_id="cue")
        
        # 设置目标球
        if target_ball == 'solid':
            self.player_targets = {
                "A": [str(i) for i in range(1, 8)],
                "B": [str(i) for i in range(9, 16)],
            }
        else:
            self.player_targets = {
                "A": [str(i) for i in range(9, 16)],
                "B": [str(i) for i in range(1, 8)],
            }
        
        # 应用课程学习配置
        active_targets = self._setup_curriculum_balls(target_ball)
        
        # 更新目标球列表（只包含实际存在的球）
        # 注意：active_targets只包含当前玩家的目标球，需要根据target_ball判断
        if target_ball == 'solid':
            self.player_targets["A"] = [t for t in self.player_targets["A"] if t in active_targets or t == '8']
            # B玩家的球可能被移除了，只保留存在的
            self.player_targets["B"] = [t for t in self.player_targets["B"] 
                                        if self.balls.get(t) and self.balls[t].state.s != 4 or t == '8']
        else:
            self.player_targets["A"] = [t for t in self.player_targets["A"] if t in active_targets or t == '8']
            self.player_targets["B"] = [t for t in self.player_targets["B"] 
                                        if self.balls.get(t) and self.balls[t].state.s != 4 or t == '8']
        
        # 重置状态
        self.hit_count = 0
        self.last_state = save_balls_state(self.balls)
        self.curr_player = 0
        self.done = False
        self.winner = None
        self.shot_record = pt.MultiSystem()
        self.episode_count += 1
        
        self._log(f"🎱 Level {self.curriculum_level}: {self.CURRICULUM_LEVELS[self.curriculum_level]['description']}")
    
    def take_shot(self, action: dict) -> dict:
        """执行击球"""
        # 添加噪声
        if self.enable_noise:
            action = {
                'V0': np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0),
                'phi': (action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360,
                'theta': np.clip(action['theta'] + np.random.normal(0, self.noise_std['theta']), 0, 90),
                'a': np.clip(action['a'] + np.random.normal(0, self.noise_std['a']), -0.5, 0.5),
                'b': np.clip(action['b'] + np.random.normal(0, self.noise_std['b']), -0.5, 0.5),
            }
        
        # 物理模拟
        shot = pt.System(table=self.table, balls=self.balls, cue=self.cue)
        self.cue.set_state(V0=action["V0"], phi=action["phi"], theta=action["theta"], 
                         a=action['a'], b=action['b'])
        pt.simulate(shot, inplace=True)
        
        if self.record_shots:
            self.shot_record.append(copy.deepcopy(shot))
        
        self.balls = shot.balls
        
        # 检测进袋
        new_pocketed = [bid for bid, b in shot.balls.items() 
                       if b.state.s == 4 and self.last_state[bid].state.s != 4]
        
        # 检测首次碰撞
        events = shot.events
        first_contact_ball_id = None
        for e in events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue']
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        
        # 分类进袋球
        player = self.get_curr_player()
        own_pocketed = [bid for bid in new_pocketed 
                       if bid in self.player_targets[player]]
        enemy_pocketed = [bid for bid in new_pocketed 
                        if bid not in self.player_targets[player] 
                        and bid not in ["cue", "8"]]
        
        # === 课程学习特殊规则：简化判定 ===
        result = {
            'ME_INTO_POCKET': own_pocketed,
            'ENEMY_INTO_POCKET': enemy_pocketed,
            'WHITE_BALL_INTO_POCKET': "cue" in new_pocketed,
            'BLACK_BALL_INTO_POCKET': "8" in new_pocketed,
            'FOUL_FIRST_HIT': False,
            'NO_POCKET_NO_RAIL': False,
            'NO_HIT': first_contact_ball_id is None,
            'BALLS': copy.deepcopy(self.balls),
        }
        
        # 白球进袋
        if "cue" in new_pocketed:
            self._log("⚪ 白球落袋")
            self.balls = restore_balls_state(self.last_state)
            # 单人模式不切换玩家
            if not self.single_player_mode:
                self.curr_player = 1 - self.curr_player
            self.hit_count += 1
            self._check_game_end()
            return result
        
        # 黑8进袋
        if "8" in new_pocketed:
            remaining = [bid for bid in self.player_targets[player] 
                        if bid != '8' and self.balls.get(bid) 
                        and self.balls[bid].state.s != 4]
            if len(remaining) == 0:
                self._log(f"🏆 成功打进黑8！")
                self.winner = player
                self._check_promotion(success=True)
            else:
                self._log(f"💥 过早打进黑8")
                self.winner = self.players[1 - self.curr_player]
                self._check_promotion(success=False)
            self.done = True
            return result
        
        # 检查是否清空目标球（课程学习成功条件）
        remaining_targets = [bid for bid in self.player_targets[player] 
                           if bid != '8' and self.balls.get(bid) 
                           and self.balls[bid].state.s != 4]
        
        if len(remaining_targets) == 0:
            self._log(f"🎯 清空所有目标球！")
            # 在课程学习中，清空目标球就算成功
            if self.curriculum_level < 5:
                self.winner = player
                self.done = True
                self._check_promotion(success=True)
                return result
        
        # 未击中球
        if first_contact_ball_id is None:
            result['NO_HIT'] = True
            if not self.single_player_mode:
                self.curr_player = 1 - self.curr_player
        # 首先碰到非目标球
        elif first_contact_ball_id not in self.player_targets[player]:
            result['FOUL_FIRST_HIT'] = True
            if not self.single_player_mode:
                self.curr_player = 1 - self.curr_player
        # 打进自己球，继续
        elif own_pocketed:
            pass
        # 没进球，换人
        else:
            if not self.single_player_mode:
                self.curr_player = 1 - self.curr_player
        
        self.last_state = save_balls_state(self.balls)
        self.hit_count += 1
        self._check_game_end()
        
        return result
    
    def _check_game_end(self):
        """检查游戏是否结束"""
        if self.hit_count >= self.MAX_HIT_COUNT:
            self._log(f"⏰ 达到最大击球数")
            self.done = True
            
            # 计算剩余球数
            a_left = sum(1 for bid in self.player_targets["A"] 
                        if bid != '8' and self.balls.get(bid) 
                        and self.balls[bid].state.s != 4)
            b_left = sum(1 for bid in self.player_targets["B"] 
                        if bid != '8' and self.balls.get(bid) 
                        and self.balls[bid].state.s != 4)
            
            if a_left < b_left:
                self.winner = "A"
            elif b_left < a_left:
                self.winner = "B"
            else:
                self.winner = "SAME"
            
            # 记录为失败（除非清空了目标球）
            success = (a_left == 0) if self.curr_player == 0 else (b_left == 0)
            self._check_promotion(success=success)
    
    def get_curriculum_stats(self) -> dict:
        """获取课程学习统计信息"""
        stats = {
            'current_level': self.curriculum_level,
            'episode_count': self.episode_count,
            'recent_success_rate': sum(self.success_history) / max(1, len(self.success_history)),
            'level_stats': self.level_stats,
        }
        return stats


# 测试代码
if __name__ == '__main__':
    env = CurriculumPoolEnv(verbose=True)
    env.auto_curriculum = False
    
    for level in range(1, 6):
        env.set_curriculum_level(level)
        env.reset(target_ball='solid')
        
        balls, targets, table = env.get_observation()
        active_balls = [bid for bid, b in balls.items() if b.state.s != 4]
        print(f"Level {level}: 活跃球 = {active_balls}, 目标球 = {targets}")
        print()
