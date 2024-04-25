import pickle
from typing import TextIO


class GameStats:
    def __init__(self, max_epochs, max_sessions):
        self.max_epochs = max_epochs
        self.max_sessions = max_sessions

        self.p1_wins = []
        self.p2_wins = []
        self.p1_dq = []
        self.p2_dq = []
        self.game_draws = []
        self.avg_loss = []
        self.exploration_rate = []

    def add_epoch_stats(self, epoch_stats):
        self.p1_wins.append(epoch_stats["p1_wins"])
        self.p2_wins.append(epoch_stats["p2_wins"])
        self.p1_dq.append(epoch_stats["p1_dq"])
        self.p2_dq.append(epoch_stats["p2_dq"])
        self.game_draws.append(epoch_stats["game_draws"])
        self.avg_loss.append(epoch_stats["avg_loss"])
        self.exploration_rate.append(epoch_stats["exploration_rate"])


def save_stats(filename, stats):
    file: TextIO
    with open(filename, "wb") as file:
        pickle.dump(stats, file)

def load_stats(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
