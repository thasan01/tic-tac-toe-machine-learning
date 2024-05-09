import sys
import math
import numpy as np
import random
from itertools import count
import pandas as pd
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import t3stats


threshold = 5
plt.tight_layout()


def get_arg(idx, val):
    if idx < len(sys.argv):
        return sys.argv[idx]
    else:
        return val


def create_bucket(list_size, grp_threshold):
    ls = list(range(0, list_size))
    return list(map(lambda x: math.ceil((x + 1) / grp_threshold), ls))


class Animation:
    def __init__(self, fig, filename):
        self.filename = filename
        self.fig = fig
        self.anim = FuncAnimation(fig, self.callback_func, interval=10000, cache_frame_data=False, repeat=False)

    def callback_func(self, i):

        stats = t3stats.load_stats(self.filename)
        num_epochs = len(stats.avg_loss)
        complete = stats.max_epochs == num_epochs

        group_type = create_bucket(num_epochs, threshold)
        df = pd.DataFrame({
            "p1_wins": stats.p1_wins,
            "p2_wins": stats.p2_wins,
            "p1_dq": stats.p1_dq,
            "p2_dq": stats.p2_dq,
            "game_draws": stats.game_draws,
            "group_type": group_type
        })

        grp = df.groupby("group_type")
        p1_wins = grp["p1_wins"].mean()
        p2_wins = grp["p2_wins"].mean()
        p1_dq = grp["p1_dq"].mean()
        p2_dq = grp["p2_dq"].mean()
        game_draws = grp["game_draws"].mean()
        x_axis = np.arange(start=0, stop=len(p1_wins)*threshold, step=threshold)

        plt.cla()
        plt.plot(x_axis, p1_wins, label='p1 wins')
        plt.plot(x_axis, p2_wins, label='p2 wins')
        plt.plot(x_axis, game_draws, label='draws')
        plt.plot(x_axis, p1_dq, label='p1 dq')
        plt.plot(x_axis, p2_dq, label='p2 dq')

        plt.legend(loc='upper left')
        plt.tight_layout()

        if complete:
            self.anim.event_source.stop()
            #sys.exit(1)


anim = Animation(plt.gcf(), "data/model/t3-stats.dat")
plt.show()
