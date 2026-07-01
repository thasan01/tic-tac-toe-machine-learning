import os
import os.path as path
import shutil

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset.offline_dataset import OfflineDataset
from model.t3dqn import get_model
from utils.device_util import get_device
from utils.random_util import Random

MODEL_DIR            = "/tmp/t3"
BEST_MODEL_DIR       = "/tmp/t3-best"
MODEL_FILE           = path.join(MODEL_DIR, "t3_model.pth")
TRAINING_CONFIG_FILE = path.join(MODEL_DIR, "t3_training.pth")
SESSION_DIR          = "game/build/sessions"
DAT_FILE             = "game/build/sessions/memories.dat"

LEARN_RATE         = 0.00005
DISCOUNT_RATE      = 0.9
BATCH_SIZE         = 64
MAX_EPOCHS         = 100
SAVE_PER_EPOCH     = 10
INVALID_MOVE_SCORE = -5.0
OUTPUT_SIZE        = 9
HUBER_DELTA        = 1.0


def _huber_loss(y: torch.Tensor, yhat: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    """Huber loss via basic elementwise ops (compatible with DirectML)."""
    diff = torch.abs(y - yhat)
    return torch.where(diff <= delta, 0.5 * diff.pow(2), delta * (diff - 0.5 * delta)).mean()


def _valid_actions(state: torch.Tensor) -> list[int]:
    """Derive valid (empty) moves from one-hot state: cell i is empty when state[i*3] == 1."""
    return [i for i in range(OUTPUT_SIZE) if state[i * 3].item() == 1.0]


def _fetch_next_states(dataset: OfflineDataset, next_indices: list[int],
                       done: torch.Tensor, state_dim: int, device) -> torch.Tensor:
    """Look up next states by index; terminal steps get a zero vector."""
    next_states = torch.zeros(len(next_indices), state_dim, device=device)
    for i, (ni, is_done) in enumerate(zip(next_indices, done.tolist())):
        if not is_done and ni != -1:
            next_states[i] = dataset[ni]['state'].to(device)
    return next_states


def _train_step(policy_dqn, target_dqn, optimizer,
                batch, dataset, device, discount_rate):
    target_dqn.eval()
    policy_dqn.train()

    states       = batch['state'].to(device)        # (B, 30)
    actions      = batch['action'].to(device)       # (B,)
    rewards      = batch['reward'].to(device)       # (B,)
    done         = batch['done'].to(device)         # (B, bool)
    next_indices = batch['next_idx'].tolist()       # list[int]

    B = states.shape[0]
    next_states = _fetch_next_states(dataset, next_indices, done, states.shape[1], device)

    with torch.no_grad():
        # The next state is the opponent's turn in a two-player zero-sum game.
        # Their best move hurts us, so we subtract the discounted opponent Q-value.
        next_q = target_dqn(next_states)  # (B, 9)
        for i in range(B):
            if not done[i].item() and next_indices[i] != -1:
                invalid = [j for j in range(OUTPUT_SIZE)
                           if j not in _valid_actions(next_states[i])]
                if invalid:
                    next_q[i, invalid] = INVALID_MOVE_SCORE

        max_next_q = next_q.max(dim=1).values                               # (B,)
        not_done   = (~done).float()
        target_q   = rewards + not_done * (-discount_rate * max_next_q)     # (B,)

        # Build per-sample target vector: copy target_dqn output, then substitute
        # the Bellman value at the action that was actually taken.
        yhat = target_dqn(states).clone()                              # (B, 9)
        yhat[torch.arange(B, device=device), actions] = target_q
        for i in range(B):
            invalid = [j for j in range(OUTPUT_SIZE)
                       if j not in _valid_actions(states[i])]
            if invalid:
                yhat[i, invalid] = INVALID_MOVE_SCORE

    y    = policy_dqn(states)   # (B, 9)
    loss = _huber_loss(y, yhat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    device = get_device()
    os.makedirs(MODEL_DIR, exist_ok=True)

    seed = 369
    rng = Random(seed)
    model_args = {"random": rng}
    policy_dqn = get_model(filename=MODEL_FILE, input_args=model_args).to(device)
    target_dqn = get_model(filename=None, input_args=model_args).to(device)

    optimizer = optim.Adam(policy_dqn.parameters(), lr=LEARN_RATE, foreach=False)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    init_epoch = 0
    best_loss  = float('inf')

    if path.isfile(TRAINING_CONFIG_FILE):
        train_config = torch.load(TRAINING_CONFIG_FILE, weights_only=False)
        if "epoch"     in train_config: init_epoch = train_config["epoch"]
        
        if "best_loss" in train_config: best_loss  = train_config["best_loss"]

        if "optimizer" in train_config:
            optimizer.load_state_dict(train_config["optimizer"])

        if "scheduler" in train_config: 
            scheduler.load_state_dict(train_config["scheduler"])

    dataset    = OfflineDataset(SESSION_DIR, DAT_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(init_epoch, init_epoch + MAX_EPOCHS):
        print(f"Entering {epoch} epoch")
        # Sync target network to policy at the start of each epoch
        target_dqn.load_state_dict(policy_dqn.state_dict())

        epoch_loss = 0.0
        for batch_idx, memories in enumerate(dataloader):
            epoch_loss += _train_step(
                policy_dqn, target_dqn, optimizer,
                memories, dataset, device, DISCOUNT_RATE,
            )

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch:04d}  loss={avg_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch % SAVE_PER_EPOCH == 0) or is_best:
            torch.save(policy_dqn, MODEL_FILE)
            torch.save({
                "epoch":     epoch + 1,
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, TRAINING_CONFIG_FILE)

            if is_best:
                shutil.copytree(MODEL_DIR, BEST_MODEL_DIR, dirs_exist_ok=True)
