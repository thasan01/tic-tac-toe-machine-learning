import os
import re
import sys
import json
import shutil
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from src.model.t3dqn import T3DQNet, load_model, save_model_checkpoint

def onehot_encode_state(action, player_id:int):
    onehot_player = [0, 0, 0]
    onehot_player[player_id] = 1
    return action["board"] + onehot_player

class T3DQLDataset(Dataset):
    def __init__(self, root_dir: str, file_expression: str = ".*", exploration_rate=0.0):
        self.root_dir = root_dir
        self.file_pattern = re.compile(file_expression)
        self.memories = [None]
        self.board_states = []

    def pre_step(self, epoch: int):
        global default_move_score
        global good_move_score
        self.memories *= 0
        self.board_states = []

        files_to_scan = self.__scan_dir()
        for filename in files_to_scan:
            with open(filename) as file:
                current_player = int(filename.split('-')[-1].split('.')[0])
                parsed_json = json.load(file)
                winner = parsed_json["winner"] if "winner" in parsed_json else None
                history = parsed_json["history"]
                max_actions = len(history)

                for act_idx, action in enumerate(history):
                    curr_choice = action["choice"]
                    is_game_end = (act_idx == max_actions - 1)

                    curr_state = torch.tensor(onehot_encode_state(action, current_player))
                    curr_state_idx = len(self.board_states)

                    # If it's the last action, there is no next state.
                    next_state_idx = -1
                    if not is_game_end:
                        next_state_idx = len(self.board_states) + 1

                    reward = default_move_score
                    if is_game_end:
                        # Reward for the final move is based on the game's outcome
                        if winner == 0:  # Draw
                            reward = 0
                        elif current_player == winner:
                            reward = good_move_score  # Win
                        else:
                            reward = -good_move_score  # Loss

                    self.board_states.append(curr_state)
                    self.memories.append([curr_state_idx, next_state_idx, curr_choice, reward, is_game_end])

        if len(self.board_states) > 0:
            self.board_states = torch.stack(self.board_states).float().to(device)


    def post_step(self):
        pass

    def __scan_dir(self):
        files_to_scan = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                if self.file_pattern.match(filename):
                    files_to_scan.append(os.path.join(root, filename))
        files_to_scan.sort()
        return files_to_scan

    def __len__(self):
        return len(self.memories)

    def __getitem__(self, idx):
        return self.memories[idx]

def archive_func_generator(src_dir, dest_dir):
    if dest_dir:
        return lambda src, dest: shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
    else:
        return lambda src, dest: None

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Script is missing required input args")
        sys.exit(1)

    _, model_dir, data_dir, archive_dir, init_config_filename = sys.argv[:5]

    with open(init_config_filename) as file:
        init_config = json.load(file)

    num_input_nodes = init_config.get("num_input_nodes",30)
    num_hidden_layer_nodes = init_config.get("num_hidden_layer_nodes",[128, 256, 512, 256, 128])
    num_output_nodes = init_config.get("num_output_nodes",9)
    relu_rate = init_config.get("relu_rate",0.1)
    dropout_rate = init_config.get("dropout_rate",0.1)

    default_move_score = init_config.get("default_move_score", -0.01)
    good_move_score = init_config.get("good_move_score", 10)
    discount_factor = init_config.get("discount_factor", 0.9)
    learn_step_size = init_config.get("learn_step_size", 500)
    max_epochs = init_config.get("max_epochs",500)
    max_sessions = init_config.get("max_sessions",500)
    batch_size = init_config.get("batch_size", 512)
    learn_rate_range = init_config.get("learn_rate_range", [1e-3, 1e-6])
    min_lr, max_lr = learn_rate_range

    archive_model = archive_func_generator(model_dir, archive_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t3policy_dqn, t3config = load_model(model_dir, is_inference=False, num_input_nodes=num_input_nodes,
                                        num_hidden_layer_nodes=num_hidden_layer_nodes,
                                        num_output_nodes=num_output_nodes, relu_rate=relu_rate,
                                        dropout_rate=dropout_rate)
    t3policy_dqn.train().to(device)

    t3target_dqn = T3DQNet(num_input_nodes, num_hidden_layer_nodes, num_output_nodes, relu_rate=relu_rate,
                           dropout_rate=dropout_rate).to(device)
    t3target_dqn.eval()

    dataset = T3DQLDataset(data_dir, file_expression = "training-(.*).txt")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    loss_fn = torch.nn.SmoothL1Loss() # torch.nn.MSELoss()
    optimizer = optim.Adam(t3policy_dqn.parameters(), lr=max_lr)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, mode='triangular2', step_size_up= learn_step_size) # last_epoch=learn_step_size - 1, uncomment to swap order

    avg_loss = None
    total_loss = 0.0
    num_batches = 0
    initial_epoch = 0

    dataset.pre_step(0)
    for epoch in range(initial_epoch, max_epochs):
        for i, batch in enumerate(loader):
            curr_state_idx, next_state_idx, curr_choice, reward, is_game_end = batch

            # Move tensors to the GPU
            curr_state_idx = curr_state_idx.to(device)
            next_state_idx = next_state_idx.to(device)
            curr_choice = curr_choice.to(device)
            reward = reward.to(device)
            is_game_end = is_game_end.to(device)

            # Calculate predicted Q-values
            curr_states = dataset.board_states[curr_state_idx]
            policy_q_values = t3policy_dqn(curr_states)
            predicted_q_values = policy_q_values.gather(1, curr_choice.unsqueeze(1)).squeeze(1)

            # Calculate target Q-values
            next_q_values = torch.zeros(predicted_q_values.size(), device=device)

            # Only calculate next Q-values for non-terminal states
            non_terminal_indices = torch.where(~is_game_end)
            if non_terminal_indices[0].size(0) > 0:
                next_states = dataset.board_states[next_state_idx[non_terminal_indices]]
                next_q_values_temp = t3target_dqn(next_states).detach()
                #max_next_q_values, _ = next_q_values_temp.max(dim=1)
                policy_next_actions = t3policy_dqn(next_states).argmax(dim=1)  # Select actions with policy net
                max_next_q_values = next_q_values_temp.gather(1, policy_next_actions.unsqueeze(1)).squeeze(1)  # Evaluate with target net
                next_q_values[non_terminal_indices] = max_next_q_values

            target_q_values = reward + discount_factor * next_q_values
            target_q_values = target_q_values.float()

            # Compute loss
            loss = loss_fn(predicted_q_values, target_q_values)
            total_loss += loss.item()
            num_batches += 1

            # Backward propagate and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(t3policy_dqn.parameters(), 100)
            optimizer.step()

        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else -1
        current_learn_rate = scheduler.get_last_lr()[0]
        print(f"TRAINING epoch: {epoch} loss: {avg_loss}, learn_rate: {current_learn_rate:.2e}")

        dataset.post_step()

    if avg_loss:
        t3target_dqn.load_state_dict(t3policy_dqn.state_dict())
        save_model_checkpoint(model_dir, t3policy_dqn, optimizer_state=optimizer.state_dict(), epoch=max_epochs, loss=avg_loss,  scheduler_state=scheduler.state_dict())
        archive_model(model_dir, archive_dir)
