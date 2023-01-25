import matplotlib.pyplot as plt
import numpy as np

savename = 'without_noise_tmp'
readname = 'without_noise'

with open(f'{readname}.npy', 'rb') as f:
    en_dep = np.load(f)

plt.figure(num=0, figsize=(12, 6))
plt.clf()
plt.imshow(en_dep.sum(axis=2).squeeze(axis=0), interpolation="nearest", origin="upper", aspect="auto")
plt.colorbar()
plt.savefig(savename)