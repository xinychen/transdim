
```python
import numpy as np
np.random.seed(1)
import time

missing_rate = 0.95
print('Missing rate = {}'.format(missing_rate))

dense_mat = np.load('volume.npy')
d = 3
dense_vec = dense_mat[0, : 96 * d]
T = dense_vec.shape[0]
sparse_vec = dense_vec * np.round(np.random.rand(T) + 0.5 - missing_rate)

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize = (5.5, 2.4))
ax = fig.add_subplot(111)
plt.plot(dense_vec[: 96 * d], 'dodgerblue', linewidth = 2)
plt.xlabel('Time')
plt.ylabel('Volume')
plt.xticks(np.arange(0, 96 * d + 1, 48))
plt.xlim([0, 96 * d])
plt.yticks(np.arange(0, 301, 100))
plt.ylim([0, 300])
plt.grid(linestyle = '-.', linewidth = 0.5)
ax.tick_params(direction = 'in')

plt.savefig('volume_time_series.pdf', bbox_inches = "tight")
plt.show()
```

```python
import numpy as np
np.random.seed(1)
import time

missing_rate = 0.95
print('Missing rate = {}'.format(missing_rate))

dense_mat = np.load('volume.npy')
d = 3
dense_vec = dense_mat[0, : 96 * d]
T = dense_vec.shape[0]
sparse_vec = dense_vec * np.round(np.random.rand(T) + 0.5 - missing_rate)

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize = (5.5, 2.4))
ax = fig.add_subplot(111)
pos = np.where(sparse_vec != 0)
plt.plot(pos[0], sparse_vec[pos], 'o', 
         markeredgecolor = 'darkblue', 
         markerfacecolor = 'deepskyblue', markersize = 10)
plt.xlabel('Time')
plt.ylabel('Volume')
plt.xticks(np.arange(0, 96 * d + 1, 48))
plt.xlim([0, 96 * d])
plt.yticks(np.arange(0, 301, 100))
plt.ylim([0, 300])
plt.grid(linestyle = '-.', linewidth = 0.5)
ax.tick_params(direction = 'in')

plt.savefig('volumes_masked_time_series_95.pdf', bbox_inches = "tight")
plt.show()
```
