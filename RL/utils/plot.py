import json
import pickle
import traceback
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def get_ma(x, ma_len):
    ma = []
    for i in range(len(x)-ma_len):
        ma.append(sum(x[i:i+ma_len])/ma_len)
    return ma


def plot(pkl_dir, ep_avg, ma_len, verbose, lim=None):
    fnames = sorted(glob(pkl_dir + '*.pkl'))
    n = len(fnames)

    try:
        a = max(2, int(np.ceil(np.sqrt(n))))
        fig, ax = plt.subplots(a, a, figsize=(20, 10))
        fig.tight_layout(pad=3.0)
        for i in range(0, a):
            for j in range(0, a):
                k = i*a+j
                with open(fnames[k], 'rb') as f:
                    log = pickle.load(f)

                if verbose:
                    print('File', k)
                    print('Optimizer Steps', log['episodes'][-1]['optim_steps'])
                    print(json.dumps(log['params'], indent=2))
                    print('-'*13)

                rs = []
                max_r = 0
                min_r = 0
                for ep_info in log['episodes']:
                    r = ep_info['sum_reward']
                    if ep_avg:
                        r = r/(ep_info['t_ep'])
                    rs.append(r)
                    if r > max_r:
                        max_r = r
                    if r < min_r:
                        min_r = r
                if lim is not None:
                    min_r = lim[0]
                    max_r = lim[1]

                ma = get_ma(rs, ma_len)

                if ep_avg:
                    y_lab = 'Avg Reward'
                else:
                    y_lab = 'Sum Reward'

                ax[i, j].scatter(range(len(rs)), rs, s=1)
                ax[i, j].plot(range(ma_len, len(rs)), ma, color='red', label=str(ma_len)+'-MA')
                ax[i, j].set_title(fnames[k].split('/')[-1])
                ax[i, j].set_xlabel('Episode')
                ax[i, j].set_ylim(min_r-.1*abs(min_r), max_r+.1*abs(max_r))
                ax[i, j].set_ylabel(y_lab)
                ax[i, j].legend(loc='upper left')
    except Exception as e:
#         traceback.print_exc()
        pass
