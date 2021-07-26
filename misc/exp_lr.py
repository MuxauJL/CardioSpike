# Exponential LR gamma
import numpy as np
MAX_LR = 2e-4
MIN_LR = 1e-5
EPOCHES = 300


gamma =  np.power(MIN_LR / MAX_LR, 1 / EPOCHES)
print('gamma for exp lr:', gamma)

