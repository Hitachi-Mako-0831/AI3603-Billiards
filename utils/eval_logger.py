import datetime
import inspect
import json
import os
from typing import Any, Dict, Optional


class _EvaluateLogger:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._game_meta_by_env_id: Dict[int, Dict[str, Any]] = {}

    def _ensure_parent_dir(self) -> None:
        parent = os.path.dirname(self.file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        self._ensure_parent_dir()
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_reset(self, env: Any, target_ball: Optional[str], caller_frame) -> None:
        meta: Dict[str, Any] = {}
        i = caller_frame.f_globals.get("i")
        players = caller_frame.f_globals.get("players")

        meta["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        meta["game_id"] = int(i) if isinstance(i, int) else None
        meta["target_ball"] = target_ball

        if isinstance(players, (list, tuple)) and len(players) == 2:
            meta["fixed_agents"] = {
                "AGENT_A": players[0],
                "AGENT_B": players[1],
                "AGENT_A_CLASS": players[0].__class__.__name__,
                "AGENT_B_CLASS": players[1].__class__.__name__,
            }
            if isinstance(i, int):
                meta["player_to_agent"] = {
                    "A": players[i % 2],
                    "B": players[(i + 1) % 2],
                }
                meta["player_to_agent_class"] = {
                    "A": players[i % 2].__class__.__name__,
                    "B": players[(i + 1) % 2].__class__.__name__,
                }

        self._game_meta_by_env_id[id(env)] = meta

    def _remaining_balls_for_player(self, env: Any, player_id: str):
        targets = getattr(env, "player_targets", {}).get(player_id, [])
        balls = getattr(env, "balls", {})
        remaining = []
        for bid in targets:
            if bid == "8":
                continue
            b = balls.get(bid)
            if b is None:
                continue
            if getattr(getattr(b, "state", None), "s", None) != 4:
                remaining.append(bid)
        return remaining

    def _eight_ball_pocketed(self, env: Any) -> Optional[bool]:
        balls = getattr(env, "balls", None)
        if balls is None or not isinstance(balls, dict):
            return None
        eight = balls.get("8")
        if eight is None:
            return None
        state = getattr(eight, "state", None)
        if state is None:
            return None
        s = getattr(state, "s", None)
        if s is None:
            return None
        return bool(s == 4)

    def _infer_end_reason(self, env: Any, step_info: Dict[str, Any], shooter_player: Optional[str]) -> str:
        if getattr(env, "hit_count", 0) >= getattr(env, "MAX_HIT_COUNT", 0):
            return "MAX_HIT_COUNT"

        white = bool(step_info.get("WHITE_BALL_INTO_POCKET", False))
        black = bool(step_info.get("BLACK_BALL_INTO_POCKET", False))
        if white and black:
            return "CUE_AND_8_FOUL_LOSS"
        if black:
            winner = getattr(env, "winner", None)
            if shooter_player is not None and winner == shooter_player:
                return "LEGAL_8_POCKETED"
            return "ILLEGAL_8_POCKETED"
        if step_info.get("NO_HIT", False):
            return "NO_HIT_FOUL"
        if step_info.get("FOUL_FIRST_HIT", False):
            return "FOUL_FIRST_HIT"
        if step_info.get("NO_POCKET_NO_RAIL", False):
            return "NO_POCKET_NO_RAIL_FOUL"
        return "UNKNOWN"

    def on_take_shot_end(self, env: Any, step_info: Dict[str, Any], shooter_player: Optional[str]) -> None:
        if not getattr(env, "done", False):
            return

        meta = self._game_meta_by_env_id.get(id(env), {})
        winner_player = getattr(env, "winner", None)
        reason = self._infer_end_reason(env, step_info, shooter_player)

        winner_agent_class = None
        if winner_player in ("A", "B"):
            winner_agent_class = meta.get("player_to_agent_class", {}).get(winner_player)
        elif winner_player == "SAME":
            winner_agent_class = "SAME"

        hit_count = getattr(env, "hit_count", None)
        remaining_a = self._remaining_balls_for_player(env, "A")
        remaining_b = self._remaining_balls_for_player(env, "B")
        eight_pocketed = self._eight_ball_pocketed(env)

        offender_player = None
        offender_agent_class = None
        if reason in ("ILLEGAL_8_POCKETED", "CUE_AND_8_FOUL_LOSS") and winner_player in ("A", "B"):
            offender_player = "B" if winner_player == "A" else "A"
            offender_agent_class = meta.get("player_to_agent_class", {}).get(offender_player)

        record: Dict[str, Any] = {
            "type": "game_end",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "game_id": meta.get("game_id"),
            "winner_agent_class": winner_agent_class,
            "winner_player": winner_player,
            "winner_reason": reason,
            "player_a_agent_class": meta.get("player_to_agent_class", {}).get("A"),
            "player_b_agent_class": meta.get("player_to_agent_class", {}).get("B"),
            "target_ball": meta.get("target_ball"),
            "hit_count": hit_count,
            "remaining_a": remaining_a,
            "remaining_b": remaining_b,
            "remaining_a_count": len(remaining_a),
            "remaining_b_count": len(remaining_b),
            "eight_pocketed": eight_pocketed,
            "offender_player": offender_player,
            "offender_agent_class": offender_agent_class,
        }

        if isinstance(meta.get("game_id"), int):
            i = meta["game_id"]
            if winner_player in ("A", "B"):
                winner_idx = (i % 2) if winner_player == "A" else ((i + 1) % 2)
                record["winner_agent"] = "AGENT_A" if winner_idx == 0 else "AGENT_B"
            elif winner_player == "SAME":
                record["winner_agent"] = "SAME"

        self._append_jsonl(record)

    def write_summary(self, results: Dict[str, Any]) -> None:
        payload = {
            "type": "summary",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "results": results,
        }
        self._append_jsonl(payload)


_LOGGER: Optional[_EvaluateLogger] = None


def _get_default_log_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.environ.get("EVAL_LOG_PATH")
    if env_path:
        return env_path
    return os.path.join(project_root, "logs", "evaluate_log.jsonl")


def _get_logger() -> _EvaluateLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = _EvaluateLogger(_get_default_log_path())
    return _LOGGER


def _patch_pool_env() -> None:
    import poolenv

    PoolEnv = poolenv.PoolEnv
    if getattr(PoolEnv, "_eval_logger_patched", False):
        return

    orig_reset = PoolEnv.reset
    orig_take_shot = PoolEnv.take_shot

    def reset(self, *args, **kwargs):
        caller = inspect.currentframe().f_back
        target_ball = kwargs.get("target_ball", None)
        if target_ball is None and len(args) >= 2:
            target_ball = args[1]
        result = orig_reset(self, *args, **kwargs)
        if caller is not None:
            _get_logger().on_reset(self, target_ball, caller)
        return result

    def take_shot(self, action):
        shooter_player = self.get_curr_player() if hasattr(self, "get_curr_player") else None
        step_info = orig_take_shot(self, action)
        if isinstance(step_info, dict):
            _get_logger().on_take_shot_end(self, step_info, shooter_player)
        return step_info

    PoolEnv.reset = reset
    PoolEnv.take_shot = take_shot
    PoolEnv._eval_logger_patched = True


def write_evaluation_log_summary(results: Dict[str, Any]) -> None:
    _patch_pool_env()
    _get_logger().write_summary(results)


_patch_pool_env()
