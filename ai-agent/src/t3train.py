import fnmatch
import glob
import os
import os.path as path
import sys
import time
import re
import json
import signal
import threading
import subprocess
import requests
import random
import shutil
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import src.server.t3server as t3server
from src.model.t3dqn import T3DQNet, load_model, save_model_checkpoint
from src.distribution import sample_dist

def wait_for_server(base_url):
    url = f"{base_url}/ping"
    for i in range(50):
        try:
            response = requests.get(url)
            body = response.json()

            if body["alive"]:
                return
            else:
                break

        except Exception:
            time.sleep(5)

    sys.exit(-1)


def request_shutdown(base_url: str):
    api_url = f"{base_url}/shutdown"
    requests.post(api_url)


def run_games(epoch, session_template, max_sessions, exploration_rate, player_id, swap_players):
    global p1_profile
    global p2_profile
    global data_dir

    players = [p1_profile, p2_profile]
    curr_player_id = player_id

    for i in range(max_sessions):
        session = session_template.format(epoch, i, curr_player_id)

        subprocess.run(
            "node ./game/build/tic-tac-toe.console.js {} {} --outdir {} --suppressOutput --sessionName {} --encoder OneHotEncoder --explorationRate {}".format(
                players[0], players[1], data_dir, session, exploration_rate),
            shell=True,
            capture_output=True,
            text=True
        )

        if swap_players:
            players[0], players[1] = players[1], players[0]
            curr_player_id = (curr_player_id % 2) + 1

def calculate_session_stats(stats, player_id, winner, status_msg, filename):
    if winner is None:
        if "draw" in status_msg:
            stats["draws"] += 1
            stats["sessions"][1].append(filename)
    elif winner == player_id:
        stats["wins"] += 1
        stats["sessions"][0].append(filename)
    else:
        stats["losses"] += 1
        stats["sessions"][2].append(filename)
    return


def onehot_encode_state(action, player_id:int):
    onehot_player = [0, 0, 0]
    onehot_player[agent_player_id] = 1
    return action["board"] + onehot_player

def eval_model(epoch):
    global data_dir
    eval_template = "eval-{:06d}-{:06d}-{:01d}"
    #run_games(epoch, eval_template, max_sessions//2, 0, 1, False)
    run_games(epoch, eval_template, max_sessions, 0, 2, False)

    eval_stats = {"wins": 0, "losses": 0, "draws": 0, "sessions": [[], [], []]}
    file_pattern = re.compile("eval-(.*).txt")

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if file_pattern.match(filename):
                file_path = os.path.join(root, filename)
                with open(file_path) as file:
                    parsed_json = json.load(file)
                    current_player = int(filename.split('-')[-1].split('.')[0])
                    winner = parsed_json["winner"] if "winner" in parsed_json else None
                    calculate_session_stats(eval_stats, current_player, winner, parsed_json["status"], filename)
                os.remove(file_path)

    print(f"EVAL epoch: {epoch} wins: {dataset.stats["wins"]}, losses: {dataset.stats["losses"]}, draws: {dataset.stats["draws"]}")

class T3DQLDataset(Dataset):
    def __init__(self, root_dir: str, file_expression: str = ".*", delete_training_files=True, exploration_rate=0.0, exploration_decay = 0.0, experience_replay = 0.5):
        self.delete_training_files = delete_training_files
        self.root_dir = root_dir
        self.file_pattern = re.compile(file_expression)
        self.memories = [None] # Added single element to bypass the dataloader
        self.board_states = []
        self.stats = {}
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.experience_replay = experience_replay
        self.new_sessions = int(max_sessions *  (1 - experience_replay))
        self.exp_rate_range = [0.01, 1.0]

    def __reset_stats(self):
        self.stats = {"wins": 0, "losses": 0, "draws": 0, "sessions": [[],[],[]]}

    def pre_step(self, epoch: int):
        # reset memories and board states
        self.memories *= 0
        self.board_states = []
        self.__reset_stats()

        current_player = agent_player_id
        run_games(epoch, session_template, self.new_sessions, exploration_rate=self.exploration_rate, player_id=agent_player_id, swap_players=swap_players)
        self.exploration_rate *= self.exploration_decay

        if self.exploration_rate < self.exp_rate_range[0]:
            ratio = 1 - (epoch / max_epochs)
            self.exp_rate_range[1] = min(self.exp_rate_range[1], ratio)
            self.exploration_decay = 1 / self.exploration_decay

        if self.exploration_rate > self.exp_rate_range[1]:
            ratio = (epoch / max_epochs)
            self.exp_rate_range[0] = min(self.exp_rate_range[0], ratio)
            self.exploration_decay = 1 / self.exploration_decay

        files_to_scan = self.__scan_dir()
        for filename in files_to_scan:
            with open(filename) as file:
                parsed_json = json.load(file)
                history = parsed_json["history"]
                max_actions = len(history)
                winner = parsed_json["winner"] if "winner" in parsed_json else None

                if swap_players:
                    current_player = int(filename.split('-')[-1].split('.')[0])
                    
                calculate_session_stats(self.stats, current_player, winner, parsed_json["status"], filename)

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
                        elif agent_player_id == winner:
                            reward = good_move_score  # Win
                        else:
                            reward = -good_move_score  # Loss

                    self.board_states.append(curr_state)
                    self.memories.append([curr_state_idx, next_state_idx, curr_choice, reward, is_game_end])

        if len(self.board_states) > 0:
            self.board_states = torch.stack(self.board_states).float().to(device)

    def post_step(self):
        if self.delete_training_files:
            files_to_delete, _ = sample_dist(retain_dist=experience_retain_dist,
                                           retain_ratio=experience_replay,
                                           data=self.stats["sessions"],
                                           total=max_sessions,
                                           rng_seed=42)

            for filename in files_to_delete:
                try:
                    os.remove(filename)
                except Exception as e:
                    print(f"Error deleting file {filename}: {e}")


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(sys.argv) < 4:
        print("Script is missing required input args")
        sys.exit(1)

    _, data_dir, logs_dir, model_dir = sys.argv[:4]
    archive_dir = sys.argv[4] if len(sys.argv) >= 5 else None
    archive_model = archive_func_generator(model_dir, archive_dir)


    # contains all the values for initial training
    init_config_filename = sys.argv[6] if len(sys.argv) >= 6 else "./ai-agent/data/init_train_config.json"
    with open(init_config_filename) as file:
        init_config = json.load(file)

    # server params
    server_host = init_config.get("server_host", "127.0.0.1")
    server_port = init_config.get("server_port",5000)
    server_base_url = f"http://{server_host}:{server_port}"

    num_input_nodes = init_config.get("num_input_nodes",30)
    num_hidden_layer_nodes = init_config.get("num_hidden_layer_nodes",[128, 256, 512, 256, 128])
    num_output_nodes = init_config.get("num_output_nodes",9)
    relu_rate = init_config.get("relu_rate",0.1)
    dropout_rate = init_config.get("dropout_rate",0.1)

    # training params
    max_epochs = init_config.get("max_epochs",500)
    learn_rate_range = init_config.get("learn_rate_range",[1e-3, 1e-6])
    learn_step_size = init_config.get("learn_step_size", 500)
    batch_size = init_config.get("batch_size",500)
    max_sessions = init_config.get("max_sessions",500)

    p1_profile = init_config.get("p1_profile","rl-agent")
    p2_profile = init_config.get("p2_profile","random-agent")
    agent_player_id = init_config.get("agent_player_id","1")
    swap_players = init_config.get("swap_players", False)
    archive_rate = init_config.get("archive_rate", 5)

    session_template = init_config.get("session_template","training-{:06d}-{:06d}-{:01d}")
    good_move_score = init_config.get("good_move_score",1)
    invalid_move_score = init_config.get("invalid_move_score",-10)
    default_move_score = init_config.get("default_move_score",-0.1)
    discount_factor = init_config.get("discount_factor",0.9)
    exploration_rate = init_config.get("exploration_rate",1.0)
    exploration_decay = init_config.get("exploration_decay",0.9)
    policy_sync_rate = init_config.get("policy_sync_rate",10)
    experience_replay = init_config.get("experience_replay",0.5)
    experience_retain_dist = init_config.get("experience_retain_dist",[0.8, 0.1, 0.1])
    eval_rate = init_config.get("eval_rate",None)


    min_lr, max_lr = learn_rate_range
    # exp_rate_range = [0.01, 1.0]  # [min_value, max_value]

    t3policy_dqn, t3config = load_model(model_dir, is_inference=False, num_input_nodes=num_input_nodes,
                                        num_hidden_layer_nodes=num_hidden_layer_nodes,
                                        num_output_nodes=num_output_nodes, relu_rate=relu_rate,
                                        dropout_rate=dropout_rate)
    t3policy_dqn.train().to(device)

    t3target_dqn = T3DQNet(num_input_nodes, num_hidden_layer_nodes, num_output_nodes, relu_rate=relu_rate,
                           dropout_rate=dropout_rate).to(device)
    t3target_dqn.eval()

    # Server startup
    def run_webapp():
        agent_dqn = T3DQNet(num_input_nodes, num_hidden_layer_nodes, num_output_nodes, relu_rate=relu_rate,
                            dropout_rate=dropout_rate).to(device)
        agent_dqn.eval()
        t3server.model_dir = model_dir
        t3server.agent = t3server.Agent(agent_dqn)
        # Run the Flask app
        t3server.app.run(host=server_host, port=server_port, use_reloader=False)

    server_thread = threading.Thread(target=run_webapp)
    server_thread.start()

    optimizer = optim.Adam(t3policy_dqn.parameters(), lr=max_lr)
    if "optimizer_state" in t3config and t3config["optimizer_state"]:
        optimizer.load_state_dict(t3config["optimizer_state"])

    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']  # Set initial_lr to the current learning rate


    if "exploration_rate" in t3config:
        exploration_rate = t3config["exploration_rate"]

    if "exploration_decay" in t3config:
        exploration_decay = t3config["exploration_decay"]

    dataset = T3DQLDataset(data_dir, "training-(.*).txt", exploration_rate=exploration_rate, exploration_decay=exploration_decay, experience_replay=experience_replay)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    loss_fn = torch.nn.SmoothL1Loss() # torch.nn.MSELoss()

    tb_log = SummaryWriter(logs_dir)

    # create the initial half sessions
    run_games(-1, session_template, int(max_sessions * experience_replay), exploration_rate=exploration_rate, player_id=agent_player_id, swap_players=swap_players)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, mode='triangular2', step_size_up= learn_step_size, last_epoch=learn_step_size - 1)
    if "scheduler_state" in t3config:
        scheduler.load_state_dict(t3config["scheduler_state"])

    # training loop
    init_epoch = t3config["epoch"] if "epoch" in t3config else 0
    avg_loss = t3config["loss"] if "loss" in t3config else 0

    print(f"Starting training. init_epoch: {init_epoch}, max_epochs: {max_epochs}, loss: {avg_loss}")
    avg_loss = None

    for epoch in range(init_epoch, max_epochs):
        dataset.pre_step(epoch)

        tb_log.add_scalars('Stats', {
            'Wins': dataset.stats["wins"],
            'Losses': dataset.stats["losses"],
            'Draws': dataset.stats["draws"]
        }, epoch)

        if epoch % policy_sync_rate == 0:
            t3target_dqn.load_state_dict(t3policy_dqn.state_dict())

        total_loss = 0.0
        num_batches = 0

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
        tb_log.add_scalar('Loss/train', avg_loss, epoch)
        tb_log.add_scalar('Exp Rate/train', dataset.exploration_rate, epoch)
        tb_log.add_scalar('Learn Rate/train', scheduler.get_last_lr()[0], epoch)
        print(f"TRAINING epoch: {epoch} loss: {avg_loss}, learn_rate: {scheduler.get_last_lr()[0]:.2e}, exp_rate: {dataset.exploration_rate}, wins: {dataset.stats["wins"]}, losses: {dataset.stats["losses"]}, draws: {dataset.stats["draws"]}")

        dataset.post_step()

        save_model_checkpoint(model_dir, t3policy_dqn, optimizer_state=optimizer.state_dict(), epoch=epoch, loss=avg_loss, exploration_rate=dataset.exploration_rate, exploration_decay=dataset.exploration_decay, experience_replay=experience_replay, scheduler_state=scheduler.state_dict())
        requests.post(f"{server_base_url}/model/reload", json={})

        if epoch % archive_rate == 0:
            archive_model(model_dir, archive_dir)

        if eval_rate and epoch % eval_rate == 0:
            eval_model(epoch)

    if avg_loss:
        t3target_dqn.load_state_dict(t3policy_dqn.state_dict())
        save_model_checkpoint(model_dir, t3policy_dqn, optimizer_state=optimizer.state_dict(), epoch=max_epochs, loss=avg_loss, exploration_rate=dataset.exploration_rate, exploration_decay=dataset.exploration_decay, experience_replay=experience_replay, scheduler_state=scheduler.state_dict())
        archive_model(model_dir, archive_dir)

    tb_log.close()

    # Server shutdown
    try:
        print("Sending shutdown request")
        request_shutdown(server_base_url)  # Ensure this correctly calls the shutdown endpoint
        print("Waiting for server to shutdown")
        t3server.shutdown_event.wait(timeout=60)  # Wait for the shutdown event
    finally:
        try:
            print("Waiting for server thread to join")
            server_thread.join(timeout=60)  # Wait for the server thread to finish
        finally:
            print("Exiting the script")
            os._exit(os.EX_OK)
