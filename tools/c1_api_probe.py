"""Verify the exact forward + flatten/load API used by c1_es_train."""
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import numpy as np
import torch
sys.argv = ["x"]
from c1_es_train import build_policy, flatten_actor, load_actor, predict_action

pol, keys, numel = build_policy()
print("actor numel:", numel, "keys:", len(keys))

v = flatten_actor(pol, keys)
print("flat vec shape:", v.shape, "dtype:", v.dtype)

# Forward path works and is deterministic.
obs = np.random.RandomState(0).uniform(-1, 1, size=(10,)).astype(np.float32)
a1 = predict_action(pol, obs)
print("action:", a1)

# Load a zero vector -> all logits equal -> argmax ties to action 0.
load_actor(pol, keys, np.zeros(numel))
a_zero = predict_action(pol, obs)
print("action after zero-load:", a_zero)

# Load a vector that forces action 2: set action_net.bias[2] large.
v2 = flatten_actor(pol, keys)
# action_net.bias is the last 3 entries (verified order: ...,action_net.weight(192),action_net.bias(3))
v2[-3:] = np.array([-10.0, -10.0, 10.0])
load_actor(pol, keys, v2)
a_forced = predict_action(pol, obs)
print("action after forcing bias->2:", a_forced)

# Confirm round-trip: flatten after load equals what we loaded.
v3 = flatten_actor(pol, keys)
print("roundtrip max abs diff:", float(np.max(np.abs(v3 - v2))))
