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
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import src.server.t3server as t3server
from src.model.t3dqn import T3DQNet, load_model, save_model_checkpoint

def wait_for_server(base_url):
    url = f"{base_url}/ping"
    for i in range(50):
        try:
            response = requests.get(url)
            body = response.json()
            # print(f'Health API response: {body}')

            if body["alive"]: return
            else: break

        # wait for a bit
        except Exception: time.sleep(5)
    #
    sys.exit(-1)

def request_shutdown(base_url:str):
    api_url = f"{base_url}/shutdown"
    requests.post(api_url)

def run_games(epoch, session_template, max_sessions, exploration_rate):
    for i in range(max_sessions):
        session = session_template.format(epoch, i)

        subprocess.run(
            "node ./game/build/tic-tac-toe.console.js random-agent random-agent --outdir {} --suppressOutput --sessionName {} --encoder OneHotEncoder".format(data_dir, session),
            shell=True,
            capture_output=True,
            text=True
        )

def onehot_encode_state(action):
    onehot_player = [0, 0, 0]
    onehot_player[action["player"]] = 1
    return action["board"] + onehot_player


class T3DQLDataset(Dataset):
    def __init__(self, root_dir: str, file_expression: str = ".*", delete_training_files=True):
        self.delete_training_files = delete_training_files
        self.root_dir = root_dir
        self.file_pattern = re.compile(file_expression)
        self.memories = []
        self.board_states = []
        self.stats = {}

    def __reset_stats(self):
        self.stats = {"p1_wins": 0, "p2_wins": 0, "draws": 0, "p1_dqs": 0, "p2_dqs": 0}

    def __calculate_session_stats(self, winner, status_msg):
        if winner == 0:
            if "draws" in status_msg:
                self.stats["draws"] += 1
        elif winner is None:
            if status_msg == "Player1 disqualified!":
                self.stats["p1_dqs"] += 1
            elif status_msg == "Player2 disqualified!":
                self.stats["p2_dqs"] += 1
        elif winner == 1:
            self.stats["p1_wins"] += 1
        elif winner == 2:
            self.stats["p2_wins"] += 1
        return

    def pre_step(self, epoch:int):
        # reset memories and board states
        self.memories *= 0
        self.board_states = []
        self.__reset_stats()
        #self.board_states *= 0

        run_games(epoch, session_template, max_sessions, exploration_rate=1.0)

        files_to_scan = self.__scan_dir()
        for filename in files_to_scan:
            with open(filename) as file:
                parsed_json = json.load(file)
                history = parsed_json["history"]
                max_actions = len(history)
                winner = parsed_json["winner"] if "winner" in parsed_json else None
                self.__calculate_session_stats(winner, parsed_json["status"])

                for act_idx, action in enumerate(history):
                    curr_choice = action["choice"]

                    curr_state = torch.tensor(onehot_encode_state(action))
                    curr_state_idx = len(self.board_states)

                    next_state_idx = (act_idx + 1) % max_actions
                    reward = good_move_score if next_state_idx == 0 else default_move_score
                    self.board_states.append(curr_state)
                    self.memories.append([curr_state_idx,next_state_idx,curr_choice,reward])

        if len(self.board_states) > 0:
            self.board_states = torch.stack(self.board_states).float()

    def post_step(self):
        if self.delete_training_files:
            for dir_path, dir_names, filenames in os.walk(self.root_dir):
                for filename in filenames:
                    if self.file_pattern.match(filename):
                        file_path = os.path.join(dir_path, filename)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Error deleting file {file_path}: {e}")

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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(sys.argv) < 4:
        print("Script is missing required input args")
        sys.exit(1)
    else:
        _, data_dir, logs_dir, model_dir = sys.argv

    # server params
    server_host = "127.0.0.1"
    server_port = 5000
    server_base_url = f"http://{server_host}:{server_port}"

    num_input_nodes=30
    num_hidden_layer_nodes=[128, 256, 512, 256, 128]
    num_output_nodes=9
    relu_rate=0.1
    dropout_rate=0.1

    # training params
    max_epochs = 500
    learn_rate = 1e-6
    batch_size=500
    max_sessions=500

    session_template = "training-{:06d}-{:06d}"
    good_move_score = 1
    invalid_move_score = -10
    default_move_score = -0.1
    discount_factor = 0.9

    t3policy_dqn, t3config = load_model(model_dir, is_inference=False, num_input_nodes=num_input_nodes, num_hidden_layer_nodes=num_hidden_layer_nodes, num_output_nodes=num_output_nodes, relu_rate=relu_rate, dropout_rate=dropout_rate)
    t3policy_dqn.train()

    t3target_dqn = T3DQNet(num_input_nodes, num_hidden_layer_nodes, num_output_nodes, relu_rate=relu_rate, dropout_rate=dropout_rate)
    t3target_dqn.eval()

    # Server startup
    def run_webapp(): t3server.app.run(host=server_host, port=server_port)
    t3server.model_dir = model_dir
    t3server.agent = t3server.Agent(t3policy_dqn)
    server_thread = threading.Thread(target=run_webapp)
    server_thread.start()

    optimizer = optim.Adam(t3policy_dqn.parameters(), lr=learn_rate)
    if "optimizer_state" in t3config and t3config["optimizer_state"]:
        optimizer.load_state_dict(t3config["optimizer_state"])

    dataset = T3DQLDataset(data_dir, "training-(.*).txt")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    loss_fn = torch.nn.MSELoss()

    tb_log = SummaryWriter(logs_dir)

    # training loop
    init_epoch = t3config["epoch"] if "epoch" in t3config else 0
    loss = t3config["loss"] if "loss" in t3config else 0
    print(f"Starting training. init_epoch: {init_epoch}, max_epochs: {max_epochs}, loss: {loss}")
    for epoch in range(init_epoch, max_epochs):
        dataset.pre_step(epoch)

        tb_log.add_scalars('Stats', {
            'P1 Wins': dataset.stats["p1_wins"],
            'P2 Wins': dataset.stats["p2_wins"],
            'Draws': dataset.stats["draws"]
        }, epoch)

        for repeat in range(5):
            t3target_dqn.load_state_dict(t3policy_dqn.state_dict())

            for i, batch in enumerate(loader):
                curr_state_idx, next_state_idx, curr_choice, reward = batch

                # Calculate predicted Q-values
                curr_states = dataset.board_states[curr_state_idx]
                policy_q_values = t3policy_dqn(curr_states)
                predicted_q_values = policy_q_values.gather(1, curr_choice.unsqueeze(1)).squeeze(1)

                # Calculate target Q-values
                next_states = dataset.board_states[next_state_idx]
                next_q_values = t3target_dqn(next_states).detach()
                max_next_q_values, _ = next_q_values.max(dim=1)
                target_q_values = reward + discount_factor * -max_next_q_values
                target_q_values = target_q_values.float()

                # Compute loss
                loss = loss_fn(predicted_q_values, target_q_values)

                # Backward propagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_log.add_scalar('Loss/train', loss.item(), epoch * repeat * len(loader) + i)

            print(f"epoch: {epoch} repeat: {repeat} loss: {loss}")

        dataset.post_step()
        save_model_checkpoint(model_dir, t3policy_dqn, optimizer_state=optimizer.state_dict(), epoch=epoch, loss=loss)
        #requests.post(f"{server_base_url}/model/reload", json={})

    t3target_dqn.load_state_dict(t3policy_dqn.state_dict())
    save_model_checkpoint(model_dir, t3policy_dqn, optimizer_state=optimizer.state_dict(), epoch=max_epochs, loss=loss)

    tb_log.close()

    # Server shutdown
    request_shutdown(server_base_url)
    t3server.shutdown_event.wait(timeout=60)
    os.kill(os.getpid(), signal.SIGINT)
    server_thread.join(timeout=60)
