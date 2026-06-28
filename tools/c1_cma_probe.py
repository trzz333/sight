"""C1 probe: confirm pycma separable CMA-ES API on a trivial sphere.

Verifies: CMAEvolutionStrategy with CMA_diagonal=True (separable mode,
O(d) not O(d^2)), ask/tell loop, and that it descends. Also times one
ask/tell at d=5059 to confirm optimizer overhead is negligible vs Godot
rollouts (the real cost).
"""

import time
import numpy as np
import cma

# 1) Trivial sphere, small d, confirm it descends with diagonal mode.
es = cma.CMAEvolutionStrategy(
    5 * [0.5], 0.3,
    {"CMA_diagonal": True, "popsize": 8, "seed": 1, "verbose": -9},
)
for _ in range(30):
    xs = es.ask()
    fs = [float(np.sum(np.asarray(x) ** 2)) for x in xs]
    es.tell(xs, fs)
print("sphere best f after 30 gens:", es.result.fbest)

# 2) Timing at the real dimensionality, popsize ~ 4+3ln(d).
d = 5059
popsize = int(4 + 3 * np.log(d))
print("d=", d, "default popsize=", popsize)
es2 = cma.CMAEvolutionStrategy(
    d * [0.0], 0.1,
    {"CMA_diagonal": True, "popsize": popsize, "seed": 1, "verbose": -9},
)
t0 = time.time()
xs = es2.ask()
t_ask = time.time() - t0
fs = [float(np.sum(np.asarray(x) ** 2)) for x in xs]
t0 = time.time()
es2.tell(xs, fs)
t_tell = time.time() - t0
print("popsize used:", len(xs))
print("ask seconds:", round(t_ask, 4), "tell seconds:", round(t_tell, 4))
print("optimizer overhead per gen (s):", round(t_ask + t_tell, 4))
