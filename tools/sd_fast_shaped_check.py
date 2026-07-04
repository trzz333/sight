"""Regression + sanity check for reward_mode='shaped' on SignalDodgeFast.
(1) Dynamics must be identical to reward_mode='none' under the same seed+actions
    (reward cannot affect termination). (2) 'none' reward must equal survival
    length (pure +1/step). (3) shaped reward must be finite."""
import numpy as np
from sight_agent.rl.sd_fast import SignalDodgeFast


def run(env, seed, acts):
    env.reset(seed=seed)
    L, R = 0, 0.0
    for a in acts:
        _, r, t, tr, _ = env.step(int(a))
        L += 1
        R += r
        if t or tr:
            break
    return L, R


def main():
    lens_n, lens_s, rew_n, rew_s = [], [], [], []
    for s in range(20):
        acts = np.random.default_rng(1000 + s).integers(0, 3, 2000)
        en = SignalDodgeFast()
        es = SignalDodgeFast(reward_mode="shaped")
        ln, rn = run(en, 5000 + s, acts)
        ls, rs = run(es, 5000 + s, acts)
        lens_n.append(ln); lens_s.append(ls); rew_n.append(rn); rew_s.append(rs)
    lens_n, lens_s = np.array(lens_n), np.array(lens_s)
    rew_n, rew_s = np.array(rew_n), np.array(rew_s)
    print("dynamics identical none vs shaped:", bool((lens_n == lens_s).all()))
    print("none reward == length (pure survival):", bool(np.allclose(rew_n, lens_n)))
    print("shaped reward finite:", bool(np.isfinite(rew_s).all()))
    print("shaped total-reward / length (first 6):",
          [f"{rs:.0f}/{L}" for rs, L in zip(rew_s[:6], lens_s[:6])])
    print("shaped reward stays close to survival (mean ratio):",
          round(float((rew_s / lens_s).mean()), 3))


if __name__ == "__main__":
    main()
