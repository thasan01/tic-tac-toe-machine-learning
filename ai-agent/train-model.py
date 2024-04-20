import time
import subprocess
import t3dqn as t3
import sys
import requests
import json
import torch
import torch.optim as optim
from torch import nn
import pythonmonkey as pm
import glob
import os

api_base_url = "http://127.0.0.1:5000"
out_dir = "./data/training"
session_template = "training-{:06d}-{:06d}"
max_epochs = 1000
max_sessions = 5

# player_server_script = "player-server.py"
player_server_script = "no-server.py"
model_filename = "./data/model/t3-simple1.pt"


def wait_for_server(base_url):
    url = f"{base_url}/ping"
    for i in range(50):
        try:
            response = requests.get(url)
            body = response.json()
            print(f'Health API response: {body}')

            if body["alive"]:
                return
            else:
                break

        except Exception as ex:
            # wait for a bit
            print(f'server exception: {ex}')
            time.sleep(5)
    #
    sys.exit(-1)


def reload_model(base_url):
    url = f"{base_url}/model/reload"
    response = requests.post(url)
    body = response.json()
    print(f"Reload API Response: {body}")


def run_games(epoch, out_dir):
    for i in range(max_sessions):
        session = session_template.format(epoch, i)
        subprocess.run(["node", "../game/build/tic-tac-toe.console.js",
                        "--player1Type", "RLWebAgentPlayer",
                        "--player2Type", "RandomPlayer",
                        "--player1Profile", "rl-agent-1",
                        "--trueRandomRate", "0.5",
                        "--suppressOutput",
                        "--configDir", "../game/config",
                        "--outdir", out_dir,
                        "--sessionName", session,
                        "--encoder", "BitEncoder"
                        ])


def calculate_reward(action, is_winner, turns_left):
    if not action["isValid"]:
        return -10

    if is_winner and turns_left == 1:
        return 1
    elif not is_winner and turns_left == 2:
        return -1

    return 0


def create_memories(memories, epoch, in_dir):
    # memory is a list of [state, action, reward, next_state]
    for i in range(max_sessions):
        session = session_template.format(epoch, i)
        filename = in_dir + "/" + session.format(epoch, i) + ".txt"

        with open(filename) as file:
            parsed_json = json.load(file)

            if 'winner' in parsed_json:
                winner = parsed_json['winner']
            else:
                winner = 0

            history = parsed_json['history']
            valid_moves = list(filter(lambda turn: turn["isValid"], history))
            invalid_moves = list(filter(lambda turn: not turn["isValid"], history))

            max_turns = len(valid_moves)
            for curr_turn, action in enumerate(valid_moves):

                if "choice" not in action:
                    continue

                is_winner = action["player"] == winner
                turns_left = max_turns - curr_turn
                reward = calculate_reward(action, is_winner, turns_left)

                state = [action["player"], action["board"]]
                choice = action["choice"]
                if turns_left > 1:
                    next_action = valid_moves[curr_turn + 1]
                    next_state = [next_action["player"], next_action["board"]]
                else:
                    next_state = None
                memories.append([state, choice, reward, next_state])

            for action in invalid_moves:

                if "choice" not in action:
                    continue

                state = [action["player"], action["board"]]
                choice = action["choice"]
                reward = -10
                memories.append([state, choice, reward, None])
    return


def make_qlearning_train_step(policy_dqn, target_dqn, loss_fn, optimizer, discount_rate):
    def train_step(input, label, reward, next_input):

        # Sets model to TRAIN mode
        policy_dqn.train()
        target_dqn.eval()

        # Calculate Q Value
        if next_input is None:
            q_value = torch.tensor(reward)
        else:
            with torch.no_grad():
                q_value = reward + (discount_rate * target_dqn(next_input).max())

        # Makes predictions
        y = policy_dqn(input)
        yhat = target_dqn(input)

        # Update target_dqn output with q_value
        yhat[label] = q_value.item()

        # Computes loss
        loss = loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def create_list(js_proxy_list):
    ls = []
    for i in js_proxy_list:
        ls.append(i)
    return ls


def train(model, step, memories, decoder=None, board_size=0):
    model.train()
    for action in memories:
        player, state = action[0]
        decoded_state = create_list(decoder.decode(state, board_size))
        decoded_state.append(player)

        feature = torch.tensor(decoded_state, dtype=torch.float)
        label = action[1]

        reward = action[2]

        next_state = action[3]
        next_input = None
        if next_state is not None:
            next_player, next_board = next_state
            next_board = create_list(decoder.decode(next_board, board_size))
            next_board.append(next_player)
            next_input = torch.tensor(next_board)

        loss = step(input=feature, label=label, reward=reward, next_input=next_input)
        print(f"loss: {loss}")

    print(f"")
    return


def cleanup_files(file_dir, pattern):
    for f in glob.glob(f"{file_dir}/{pattern}"):
        os.remove(f)


def app():
    learn_rate = 0.01
    discount_rate = 0.9
    policy_dqn = t3.get_model(filename=model_filename)

    target_dqn = t3.get_model()
    target_dqn.load_state_dict(policy_dqn.state_dict())

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(policy_dqn.parameters(), lr=learn_rate)
    train_step = make_qlearning_train_step(policy_dqn, target_dqn, loss_fn, optimizer, discount_rate)

    board_size = 9
    encoder = pm.require("../game/src/bitencoder.js")

    wait_for_server(api_base_url)
    memories = []

    for epoch in range(max_epochs):
        run_games(epoch, out_dir)
        memories *= 0
        create_memories(memories, epoch, out_dir)

        train(model=policy_dqn, step=train_step, memories=memories, decoder=encoder, board_size=board_size)
        t3.save_model(policy_dqn, filename=model_filename, archive=False)
        cleanup_files(out_dir, "training-*.txt")
        reload_model(api_base_url)


# =========================
# Entry Point
# =========================
# with subprocess.Popen(["venv/Scripts/python", f"{player_server_script}"]) as proc:
#    app()
#    print("stopping proc")
#    proc.kill()

app()