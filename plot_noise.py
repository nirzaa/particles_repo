import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.express as px

savename = 'without_noise_tmp'
readname = 'without_noise'

with open(f'{readname}.npy', 'rb') as f:
    en_dep = np.load(f)

# ==== 2d figure ==== #

# plt.figure(num=0, figsize=(12, 6))
# plt.clf()
# plt.imshow(en_dep.sum(axis=2).squeeze(axis=0), interpolation="nearest", origin="upper", aspect="auto")
# plt.colorbar()
# plt.savefig(savename)

# =================== #

# ==== 3d figure ==== #

# Python env: pip install plotly-express
# Anaconda env: conda install -c plotly plotly_express 

# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#               color='species')
# fig.show()

import seaborn as sns
en = en_dep.squeeze(axis=0)
fig, axs = plt.subplots(5, 2)

for i in range(10):
    sns.heatmap(en[:,i,:], ax=axs[i%5,i%2])
plt.savefig("heatmap_perlayer")



# from mpl_toolkits.mplot3d import Axes3D
# M = en_dep.squeeze(axis=0)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# counter1, counter2, counter3 = range(M.shape[0]), range(M.shape[1]), range(M.shape[2])
# x,y,z = np.meshgrid(counter1, counter2, counter3)
# ax.scatter(x,y,z, c=M.flat)


# plt.show()


# =================== #



