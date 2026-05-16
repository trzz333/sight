"""H5 behavior audit: classify failure modes from godot.ndjson per-step traces.

Pause-and-reframe diagnostic approved 2026-05-15 by Jeff. Does NOT train,
does NOT load model weights. Reads existing godot.ndjson event streams
captured during the most recent state-comparator and Phase E eval runs.

Per godot-eval/godot.ndjson schema (verified live):
- run_start: physics_hz, screen_width, screen_height, hazard_size,
  spawn_interval_frames, seed (Godot RNG seed, not eval seed)
- episode_start: marks the start of an episode (one per eval seed)
- h3_step: action (-1/0/+1), action_wire (0/1/2), frame, player_x,
  player_y, reward, terminated, truncated, terminal_reason
- spawn: hazard spawn events with position and velocity
- collision: terminal collision events
- death: terminal death events

The audit:
1. Picks one early-collision episode, one longest-non-timeout, one
   timeout (full survival) per model where available.
2. Emits action distribution, player_x trajectory summary, hazard
   proximity at collision, and a failure-mode classification.
3. Writes a markdown evidence note. NO new training, NO new eval runs.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

REPO_ROOT = Path(r"C:\Projects\Sight")
STATE_EVAL = REPO_ROOT / "runs" / "rl" / "signal_dodge_ppo_h5_state_comparator" / "h5_eval_state_comparator_seed2_10k_trained_only" / "godot-eval-trained_cnn" / "godot.ndjson"
PIXEL_EVAL = REPO_ROOT / "runs" / "rl" / "signal_dodge_ppo_h5_pixel_entropy" / "h5_eval_phase_e_seed2_entropy_10k_trained_only" / "godot-eval-trained_cnn" / "godot.ndjson"

STATE_LABEL = "state_comparator_seed2"
PIXEL_LABEL = "phase_e_seed2"


@dataclass
class Episode:
    index: int  # 1-based episode index within the file
    eval_seed: int | None  # filled from the matching python.ndjson reset event
    steps: list[dict] = field(default_factory=list)  # h3_step rows
    spawns: list[dict] = field(default_factory=list)  # spawn rows in this episode window
    terminal_reason: str = ""
    terminated: bool = False
    truncated: bool = False
    length: int = 0
    reward: float = 0.0
    collision_frame: int | None = None
    collision_hazard_x: float | None = None
    death_x: float | None = None


def load_episodes(path: Path, eval_seeds: list[int]) -> list[Episode]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    # Episode boundaries are episode_start events.
    episodes: list[Episode] = []
    cur: Episode | None = None
    ep_idx = 0
    for r in rows:
        t = r.get("type")
        if t == "episode_start":
            if cur is not None:
                episodes.append(cur)
            ep_idx += 1
            cur = Episode(index=ep_idx, eval_seed=None)
        elif cur is None:
            continue
        elif t == "h3_step":
            cur.steps.append(r)
            if r.get("terminated") or r.get("truncated"):
                cur.terminated = bool(r.get("terminated"))
                cur.truncated = bool(r.get("truncated"))
                cur.terminal_reason = str(r.get("terminal_reason", ""))
                cur.length = int(r.get("frame", 0))
                cur.reward = float(r.get("reward", 0.0))
        elif t == "spawn":
            cur.spawns.append(r)
        elif t == "collision":
            cur.collision_frame = int(r.get("frame", -1))
        elif t == "death":
            cur.death_x = float(r.get("player_x", -1))
    if cur is not None:
        episodes.append(cur)
    # Filter empty episodes: Godot emits two episode_start events per reset
    # in this env config, producing alternating empty episode shells. Keep
    # only data-bearing episodes for the audit.
    real_episodes = [e for e in episodes if e.steps]
    # Attach eval seeds positionally to the real episodes.
    for ep, seed in zip(real_episodes, eval_seeds):
        ep.eval_seed = seed
    return real_episodes


def classify_episode(ep: Episode, screen_w: float = 720.0, wall_margin: float = 40.0) -> dict:
    """Compute action distribution, player_x stats, and a failure label."""
    actions = [s.get("action", 0) for s in ep.steps]
    xs = [s.get("player_x", 0.0) for s in ep.steps]
    n = len(actions)
    if n == 0:
        return {
            "label": "empty_episode",
            "n": 0,
            "actions_pct": {},
            "idle_ratio": 0.0,
            "wall_ratio": 0.0,
            "reversal_rate": 0.0,
            "mean_x": None,
            "min_x": None,
            "max_x": None,
            "span_x": None,
            "samples_x": [],
            "final_x": None,
            "death_x": None,
            "collision_frame": None,
            "terminated": ep.terminated,
            "truncated": ep.truncated,
            "terminal_reason": ep.terminal_reason,
            "length": ep.length,
            "reward": ep.reward,
        }
    action_counts = Counter(actions)
    pct = {str(a): round(c / n, 3) for a, c in action_counts.items()}
    # Sample player_x at terciles for trajectory summary.
    samples_x = [xs[0], xs[n // 3], xs[2 * n // 3], xs[-1]] if n >= 4 else xs
    mean_x = round(statistics.fmean(xs), 1)
    min_x = round(min(xs), 1)
    max_x = round(max(xs), 1)
    span_x = round(max(xs) - min(xs), 1)
    # Wall-hugging: median proportion of frames within wall_margin of an edge.
    wall_frames = sum(
        1 for x in xs if x < wall_margin or x > (screen_w - wall_margin)
    )
    wall_ratio = round(wall_frames / n, 3)
    # Idle: pct of action=0 (stay).
    idle_ratio = round(action_counts.get(0, 0) / n, 3)
    # Direction-reversal frequency: changes of sign in action.
    reversals = 0
    last_nonzero = 0
    for a in actions:
        if a != 0:
            if last_nonzero != 0 and a * last_nonzero < 0:
                reversals += 1
            last_nonzero = a
    reversal_rate = round(reversals / n, 3)
    # Final position vs death.
    final_x = xs[-1] if xs else None
    # Failure-mode label: rough first-pass; the human report can refine.
    if ep.terminal_reason == "timeout" or ep.truncated:
        label = "survived_to_timeout"
    elif idle_ratio >= 0.80:
        label = "idle_dominant_into_collision"
    elif wall_ratio >= 0.50:
        label = "wall_hugging_into_collision"
    elif reversal_rate >= 0.10 and idle_ratio < 0.50:
        label = "high_frequency_oscillation_into_collision"
    elif idle_ratio >= 0.50 and reversal_rate < 0.02:
        label = "stay_drift_into_collision"
    else:
        label = "mixed_action_into_collision"
    return {
        "label": label,
        "n": n,
        "actions_pct": pct,
        "idle_ratio": idle_ratio,
        "wall_ratio": wall_ratio,
        "reversal_rate": reversal_rate,
        "mean_x": mean_x,
        "min_x": min_x,
        "max_x": max_x,
        "span_x": span_x,
        "samples_x": [round(x, 1) for x in samples_x],
        "final_x": round(final_x, 1) if final_x is not None else None,
        "death_x": round(ep.death_x, 1) if ep.death_x is not None else None,
        "collision_frame": ep.collision_frame,
        "terminated": ep.terminated,
        "truncated": ep.truncated,
        "terminal_reason": ep.terminal_reason,
        "length": ep.length,
        "reward": ep.reward,
    }


def pick_representatives(eps: list[Episode]) -> dict[str, Episode | None]:
    """Pick 3 representatives: shortest collision, longest collision, any survival."""
    collisions = [e for e in eps if not (e.truncated or e.terminal_reason == "timeout")]
    survivals = [e for e in eps if e.truncated or e.terminal_reason == "timeout"]
    shortest_coll = min(collisions, key=lambda e: e.length) if collisions else None
    longest_coll = max(collisions, key=lambda e: e.length) if collisions else None
    any_survival = max(survivals, key=lambda e: e.length) if survivals else None
    return {"shortest_collision": shortest_coll, "longest_collision": longest_coll, "survival": any_survival}


def render_episode_block(label: str, ep: Episode | None) -> str:
    if ep is None:
        return f"### {label}\n\n_No episode of this kind in the run._\n"
    c = classify_episode(ep)
    sx = ", ".join(str(x) for x in c["samples_x"])
    return (
        f"### {label} (eval seed {ep.eval_seed}, episode index {ep.index})\n\n"
        f"- length={c['length']}, reward={c['reward']}, terminal_reason='{c['terminal_reason']}', terminated={c['terminated']}, truncated={c['truncated']}\n"
        f"- action distribution: {c['actions_pct']} (idle={c['idle_ratio']}, reversal_rate={c['reversal_rate']})\n"
        f"- player_x trajectory: mean={c['mean_x']}, min={c['min_x']}, max={c['max_x']}, span={c['span_x']}, samples_at_terciles=[{sx}]\n"
        f"- wall_hug_ratio (frames within 40px of edge): {c['wall_ratio']}\n"
        f"- final_x={c['final_x']}, death_x={c['death_x']}, collision_frame={c['collision_frame']}\n"
        f"- **classification: {c['label']}**\n"
    )


def main() -> None:
    eval_seeds = list(range(1000, 1010))
    state_eps = load_episodes(STATE_EVAL, eval_seeds)
    pixel_eps = load_episodes(PIXEL_EVAL, eval_seeds)

    def summary(label: str, eps: list[Episode]) -> str:
        lines = [f"## {label}\n"]
        lines.append(f"Total episodes parsed: {len(eps)}\n")
        # Per-episode quick table.
        lines.append("| seed | ep_idx | length | terminal_reason | classification |")
        lines.append("| ---- | ------ | ------ | --------------- | -------------- |")
        for ep in eps:
            c = classify_episode(ep)
            lines.append(
                f"| {ep.eval_seed} | {ep.index} | {ep.length} | {c['terminal_reason'] or '(collision)'} | {c['label']} |"
            )
        lines.append("")
        # Aggregate classification counts.
        ccounts = Counter(classify_episode(e)["label"] for e in eps)
        lines.append(f"**Aggregate label distribution:** {dict(ccounts)}\n")
        # Representatives.
        reps = pick_representatives(eps)
        lines.append(render_episode_block("Shortest collision", reps["shortest_collision"]))
        lines.append(render_episode_block("Longest non-timeout collision", reps["longest_collision"]))
        lines.append(render_episode_block("Survival / timeout", reps["survival"]))
        return "\n".join(lines)

    out = []
    out.append("# H5 Behavior Audit Evidence\n")
    out.append("Pause-and-reframe diagnostic approved by Jeff on 2026-05-15 after the state-observation comparator slice closed as a negative result.\n")
    out.append("Approach: read per-step `godot.ndjson` traces from the latest two eval runs and classify failure modes without retraining.\n")
    out.append("Models inspected:\n")
    out.append(f"- **state_comparator_seed2** (`MlpPolicy`, 10k timesteps, recipe inherited from Phase D/F): `{STATE_EVAL}`\n")
    out.append(f"- **phase_e_seed2** (`CnnPolicy/NatureCNN`, 10k timesteps, entropy recipe): `{PIXEL_EVAL}`\n")
    out.append("\n---\n")
    out.append(summary(STATE_LABEL, state_eps))
    out.append("\n---\n")
    out.append(summary(PIXEL_LABEL, pixel_eps))

    text = "\n".join(out)
    Path(REPO_ROOT / "docs" / "h5-behavior-audit-evidence.md").write_text(text, encoding="utf-8")
    print("wrote docs/h5-behavior-audit-evidence.md")
    print("---")
    print(text[:4000])


if __name__ == "__main__":
    main()
