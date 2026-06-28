import numpy, cma, torch, stable_baselines3 as sb3, gymnasium
for m in (numpy, cma, torch, sb3, gymnasium):
    print(m.__name__, getattr(m, "__version__", "?"), m.__file__)
import site, sys
print("USER_SITE", site.getusersitepackages())
print("ENABLE_USER_SITE", site.ENABLE_USER_SITE)
print("PREFIX_SITE", site.getsitepackages())
