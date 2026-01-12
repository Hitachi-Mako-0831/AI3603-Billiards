import argparse
import json
import os
from collections import Counter, defaultdict


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _is_illegal_black_end(reason: str) -> bool:
    return reason in ("ILLEGAL_8_POCKETED", "CUE_AND_8_FOUL_LOSS")

def _infer_offender(rec: dict):
    offender_player = rec.get("offender_player")
    offender_agent_class = rec.get("offender_agent_class")
    if offender_player is not None and offender_agent_class is not None:
        return offender_player, offender_agent_class

    winner_player = rec.get("winner_player")
    if winner_player not in ("A", "B"):
        return None, None
    offender_player = "B" if winner_player == "A" else "A"
    if offender_player == "A":
        offender_agent_class = rec.get("player_a_agent_class")
    else:
        offender_agent_class = rec.get("player_b_agent_class")
    return offender_player, offender_agent_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=os.path.join("logs", "evaluate_log.jsonl"))
    parser.add_argument("--our", type=str, default="PhysicsAgent")
    parser.add_argument("--limit_illegal_print", type=int, default=50)
    args = parser.parse_args()

    game_end = []
    summaries = []
    for rec in _iter_jsonl(args.log):
        if rec.get("type") == "game_end":
            game_end.append(rec)
        elif rec.get("type") == "summary":
            summaries.append(rec)

    if not game_end:
        print(f"no game_end records found in {args.log}")
        return

    total = 0
    our_win = 0
    same = 0
    by_reason = Counter()
    illegal_by_agent = Counter()
    illegal_details = []

    for rec in sorted(game_end, key=lambda r: (r.get("game_id") is None, r.get("game_id", -1))):
        winner_class = rec.get("winner_agent_class")
        reason = rec.get("winner_reason")
        by_reason[reason] += 1

        total += 1
        if winner_class == args.our:
            our_win += 1
        elif winner_class == "SAME":
            same += 1

        if _is_illegal_black_end(reason):
            _offender_player, offender_class = _infer_offender(rec)
            illegal_by_agent[offender_class or "UNKNOWN"] += 1
            illegal_details.append(rec)

    our_score = our_win + 0.5 * same
    our_winrate = our_score / float(total)

    print(f"log={args.log}")
    print(f"total_games={total}")
    print(f"our_agent_class={args.our}")
    print(f"our_win={our_win} same={same} loss={total - our_win - same}")
    print(f"our_score={our_score:.1f}/{total} winrate={our_winrate:.3f}")
    print(f"winner_reason_counts={dict(by_reason)}")
    print(f"illegal_black_by_offender_agent_class={dict(illegal_by_agent)}")

    ours_illegal = []
    for r in illegal_details:
        _offender_player, offender_class = _infer_offender(r)
        if offender_class == args.our:
            ours_illegal.append(r)
    print()
    print(f"our_illegal_black_events={len(ours_illegal)}")
    if not ours_illegal:
        return
    for rec in ours_illegal[: max(0, int(args.limit_illegal_print))]:
        gid = rec.get("game_id")
        offender_player, offender_class = _infer_offender(rec)
        print(
            json.dumps(
                {
                    "game_id": gid,
                    "target_ball": rec.get("target_ball"),
                    "winner_agent_class": rec.get("winner_agent_class"),
                    "winner_player": rec.get("winner_player"),
                    "winner_reason": rec.get("winner_reason"),
                    "offender_player": offender_player,
                    "offender_agent_class": offender_class,
                    "hit_count": rec.get("hit_count"),
                    "remaining_a_count": rec.get("remaining_a_count"),
                    "remaining_b_count": rec.get("remaining_b_count"),
                    "remaining_a": rec.get("remaining_a"),
                    "remaining_b": rec.get("remaining_b"),
                    "eight_pocketed": rec.get("eight_pocketed"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
