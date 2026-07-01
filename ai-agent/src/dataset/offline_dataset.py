import json
import os
import random
import struct

import torch
from torch.utils.data import Dataset

# Record layout (little-endian, no padding):
#   state    : 30 floats  (one-hot board + player turn)
#   action   : 1 int      (chosen cell 0-8)
#   reward   : 1 float    (0 mid-game; +1 win, -1 loss, +0.5 draw at terminal)
#   outcome  : 1 int      (1=win, 0=draw, -1=loss for current player)
#   done     : 1 int      (1 if terminal step)
#   prev_idx : 1 int      (index of previous step in same game, -1 if first)
#   next_idx : 1 int      (index of next step in same game, -1 if last)
_FMT = '<30f i f i i i i'
_RECORD_SIZE = struct.calcsize(_FMT)  # 144 bytes

OUTCOME_WIN = 1
OUTCOME_DRAW = 0
OUTCOME_LOSS = -1


def _encode_state(board: list, player: int) -> list:
    """One-hot encode 9 board cells (3 values each) + player turn (3 values) → 30 floats."""
    state = []
    for cell in board:
        state.extend([float(cell == 0), float(cell == 1), float(cell == 2)])
    state.extend([float(player == 1), float(player == 2), 0.0])
    return state


def _parse_outcome(winner, player: int) -> int:
    if winner is None:
        return OUTCOME_DRAW
    return OUTCOME_WIN if winner == player else OUTCOME_LOSS


class OfflineDataset(Dataset):
    """
    Offline dataset built from tic-tac-toe session JSON files.

    On first use (or when rebuild=True) it recursively scans src_dir for .txt
    session files, converts every game step into a fixed-size binary record,
    shuffles the records (while keeping prev_idx / next_idx links valid), and
    writes them to dat_file.  Subsequent uses just memory-map the file.
    """

    def __init__(self, src_dir: str, dat_file: str, rebuild: bool = False):
        self.dat_file = dat_file
        self._file = None
        self._file_pid: int | None = None

        if rebuild or not os.path.exists(dat_file):
            self._build(src_dir, dat_file)

        self._count = os.path.getsize(dat_file) // _RECORD_SIZE

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, src_dir: str, dat_file: str) -> None:
        txt_files = []
        for root, _, files in os.walk(src_dir):
            for fname in sorted(files):
                if fname.endswith('.txt'):
                    txt_files.append(os.path.join(root, fname))

        # Each element: [state, action, reward, outcome, done, prev_orig, next_orig]
        records: list[list] = []

        for filepath in txt_files:
            with open(filepath, 'r') as fh:
                try:
                    data = json.load(fh)
                except json.JSONDecodeError:
                    continue

            history = data.get('history', [])
            if not history:
                continue

            winner = data.get('winner')  # None → draw
            n = len(history)
            game_start = len(records)

            for i, step in enumerate(history):
                player: int = step['player']
                board: list = step['board']
                action: int = step['choice']
                is_last = (i == n - 1)

                state = _encode_state(board, player)
                outcome = _parse_outcome(winner, player)

                if is_last:
                    reward = {OUTCOME_WIN: 1.0, OUTCOME_DRAW: 0.5, OUTCOME_LOSS: -1.0}[outcome]
                else:
                    reward = 0.0

                prev_orig = (game_start + i - 1) if i > 0 else -1
                next_orig = (game_start + i + 1) if not is_last else -1

                records.append([state, action, reward, outcome, int(is_last), prev_orig, next_orig])

        # Shuffle and remap prev/next links to new positions
        n_total = len(records)
        order = list(range(n_total))
        random.shuffle(order)

        orig_to_new = [0] * n_total
        for new_i, orig_i in enumerate(order):
            orig_to_new[orig_i] = new_i

        dat_dir = os.path.dirname(dat_file)
        if dat_dir:
            os.makedirs(dat_dir, exist_ok=True)

        with open(dat_file, 'wb') as fh:
            for new_i in range(n_total):
                orig_i = order[new_i]
                state, action, reward, outcome, done, prev_orig, next_orig = records[orig_i]
                prev_new = orig_to_new[prev_orig] if prev_orig != -1 else -1
                next_new = orig_to_new[next_orig] if next_orig != -1 else -1
                fh.write(struct.pack(_FMT, *state, action, reward, outcome, done, prev_new, next_new))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx: int) -> dict:
        # Open file lazily, one handle per process (fork-safe).
        pid = os.getpid()
        if self._file is None or self._file_pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = open(self.dat_file, 'rb')
            self._file_pid = pid

        self._file.seek(idx * _RECORD_SIZE)
        fields = struct.unpack(_FMT, self._file.read(_RECORD_SIZE))

        return {
            'state':    torch.tensor(fields[:30],  dtype=torch.float32),
            'action':   torch.tensor(fields[30],   dtype=torch.long),
            'reward':   torch.tensor(fields[31],   dtype=torch.float32),
            'outcome':  torch.tensor(fields[32],   dtype=torch.long),
            'done':     torch.tensor(fields[33],   dtype=torch.bool),
            'prev_idx': int(fields[34]),
            'next_idx': int(fields[35]),
        }
