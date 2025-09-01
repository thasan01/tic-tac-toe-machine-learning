import os
import sys
import time
import re
import signal
import threading
import subprocess
import requests
import torch
from torch import optim
from torch.utils.data import IterableDataset, DataLoader
#from src.server.t3server import app as webapp, shutdown_event
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

def run_games(epoch, out_dir, session_template, max_sessions, exploration_rate):
    for i in range(max_sessions):
        session = session_template.format(epoch, i)
        subprocess.run([
            "node", "../game/build/tic-tac-toe.console.js",
            "random-agent random-agent",
            "--outdir", out_dir,
            "--sessionName", session,
            "--encoder", "OneHotEncoder",
            ])


class T3DQLDataset(IterableDataset):
    """
    An iterable dataset that reads video frames from a custom binary format.
    It scans a directory for files matching a pattern and yields clips of 16 frames.
    """

    def __init__(self, root_dir: str, file_expression: str = ".*"):
        self.root_dir = root_dir
        self.file_pattern = re.compile(file_expression)

    def __scan_dir(self):
        files_to_scan = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                if self.file_pattern.match(filename):
                    files_to_scan.append(os.path.join(root, filename))
        files_to_scan.sort()
        return files_to_scan

    def _load(self, files_to_scan):

        for filename in files_to_scan:
            with open(filename, "rb") as fp:
                print(f"Opening file: {filename}")
                yield 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError("Multi-process data loading is not supported. Set num_workers=0 in your DataLoader.")

        files_to_scan = self.__scan_dir()
        return self._load(files_to_scan)

def make_dql_tran_func(model):
    output_space = set(range(9))
    def train_step(feature, label, reward, next_input):
        pass

    return train_step

if __name__ == "__main__":
    # server params
    server_host = "127.0.0.1"
    server_port = 5000
    server_base_url = f"http://{server_host}:{server_port}"
    # model params
    data_dir = ""
    model_dir = "../data/model/t3"
    num_input_nodes=30
    num_hidden_layer_nodes=[128, 256, 512, 256, 128]
    num_output_nodes=9
    relu_rate=0.1
    dropout_rate=0.1
    # training params
    max_epochs = 100
    learn_rate = 1e-4

    t3policy_dqn, t3config = load_model(model_dir, num_input_nodes=num_input_nodes, num_hidden_layer_nodes=num_hidden_layer_nodes, num_output_nodes=num_output_nodes, relu_rate=relu_rate, dropout_rate=dropout_rate)
    t3target_dqn = T3DQNet(num_input_nodes, num_hidden_layer_nodes, num_output_nodes, relu_rate=relu_rate, dropout_rate=dropout_rate)
    t3policy_dqn.train()
    t3policy_dqn.eval()

    # Server startup
    def run_webapp(): t3server.app.run(host=server_host, port=server_port)
    t3server.agent = t3server.Agent(t3policy_dqn)
    server_thread = threading.Thread(target=run_webapp)
    server_thread.start()

    optimizer = optim.Adam(t3policy_dqn.parameters(), lr=learn_rate)
    if "optimizer_state" in t3config and t3config["optimizer_state"]:
        optimizer.load_state_dict(t3config["optimizer_state"])

    dataset = T3DQLDataset(data_dir)

    # training loop
    init_epoch = t3config["epoch"] if "epoch" in t3config else 0
    loss = t3config["loss"] if "loss" in t3config else 0
    print(f"Starting training. init_epoch: {init_epoch}, max_epochs: {max_epochs}, loss: {loss}")
    for epoch in range(init_epoch, max_epochs):
        t3target_dqn.load_state_dict(t3policy_dqn.state_dict())
        pass

    #save_model_checkpoint(model_dir, t3policy_dqn, optimizer_state=optimizer.state_dict(), epoch=max_epochs, loss=loss)

    # Server shutdown
    request_shutdown(server_base_url)
    t3server.shutdown_event.wait(timeout=60)
    os.kill(os.getpid(), signal.SIGINT)
    server_thread.join(timeout=60)
