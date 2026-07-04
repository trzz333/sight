import sys
sys.path.insert(0, "src")
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sight_agent.rl.sd_fast import SignalDodgeFast

O = r"C:\Projects\Sight\runs\sd_fast"


def seq_for(run, seed=5000):
    m = PPO.load(f"{O}\\{run}.zip", device="cpu")
    vn = VecNormalize.load(f"{O}\\{run}_vecnormalize.pkl",
                           DummyVecEnv([lambda: SignalDodgeFast()]))
    vn.training = False
    vn.norm_reward = False
    raw = SignalDodgeFast(max_steps=1800)
    o, _ = raw.reset(seed=seed)
    s = []
    while True:
        a = int(m.predict(vn.normalize_obs(np.asarray(o, dtype=np.float32)),
                          deterministic=True)[0])
        s.append(a)
        o, _, t, tr, _ = raw.step(a)
        if t or tr:
            break
    return s


for run in ["sd_fast_m21sh_s0_5M", "sd_fast_m21sh_s1_5M", "sd_fast_m21_s0_5M"]:
    s = seq_for(run)
    hist = [s.count(0), s.count(1), s.count(2)]
    print(run, "len", len(s), "L/S/R", hist, "first40", s[:40])

sa = seq_for("sd_fast_m21sh_s0_5M")
sb = seq_for("sd_fast_m21sh_s1_5M")
print("shaped s0 vs s1 greedy-action-sequence identical on seed5000:", sa == sb)
