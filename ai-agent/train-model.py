from statistics import mean
import time
import subprocess
import sys
import requests
import json
import torch
import torch.optim as optim
from torch import nn
import glob
import os
import t3dqn as t3
import t3stats
from randomutil import Random

api_base_url = "http://127.0.0.1:5000"
out_dir = "./data/training"
session_template = "training-{:06d}-{:06d}"
max_epochs = 400
max_sessions = 100

delete_training_files = True
# player_server_script = "player-server.py"
player_server_script = "no-server.py"
model_filename = "./data/model/t3-simple1.pt"
stats_filename = "./data/model/t3-stats.dat"

good_move_score = 1
bad_move_score = -1
invalid_move_score = -5
default_move_score = -0.1


def wait_for_server(base_url):
    url = f"{base_url}/ping"
    for i in range(50):
        try:
            response = requests.get(url)
            body = response.json()
            # print(f'Health API response: {body}')

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


def run_games(epoch, out_dir, exploration_rate):
    for i in range(max_sessions):
        session = session_template.format(epoch, i)
        subprocess.run(["node", "../game/build/tic-tac-toe.console.js",
                        # "--player1Type", "RandomPlayer",
                        "--player1Type", "RLWebAgentPlayer",
                        # "--player2Type", "RLWebAgentPlayer",
                        "--player2Type", "RandomPlayer",
                        "--player1Profile", "rl-agent-1",
                        # "--player2Profile", "rl-agent-1",
                        "--trueRandomRate", "0.0",
                        "--suppressOutput",
                        "--configDir", "../game/config",
                        "--outdir", out_dir,
                        "--sessionName", session,
                        "--encoder", "OneHotEncoder",
                        "--explorationRate", f"{exploration_rate}"
                        ])


def calculate_session_stats(stats, winner, status_msg):
    if winner == 0:
        if "draw" in status_msg:
            stats["game_draws"] += 1
    elif winner is None:
        if status_msg == "Player1 disqualified!":
            stats["p1_dq"] += 1
        elif status_msg == "Player2 disqualified!":
            stats["p2_dq"] += 1
    elif winner == 1:
        stats["p1_wins"] += 1
    elif winner == 2:
        stats["p2_wins"] += 1
    return


def onehot_encode_state(action):
    onehot_player = [0, 0, 0]
    onehot_player[action["player"]] = 1
    return action["board"] + onehot_player


def process_player_move(memories, is_winner, player_moves):
    num_moves = len(player_moves)
    for curr_turn, action in enumerate(player_moves):
        # error check
        if "choice" not in action:
            return

        state = [onehot_encode_state(action), action["options"]]
        choice = action["choice"]

        reward = default_move_score
        next_state = None

        # calculate_reward logic
        if not action["isValid"]:
            reward = invalid_move_score
        elif curr_turn + 1 >= num_moves:
            if is_winner:
                reward = good_move_score
            else:
                reward = bad_move_score
        else:
            next_action = player_moves[curr_turn + 1]
            next_state = [onehot_encode_state(next_action), next_action["options"]]

        memories.append([state, choice, reward, next_state])
    return


def create_memories(memories, epoch, in_dir):
    # memory is a list of [state, action, reward, next_state]

    epoch_stats = {"p1_wins": 0, "p2_wins": 0, "p1_dq": 0, "p2_dq": 0, "game_draws": 0}
    for i in range(max_sessions):
        session = session_template.format(epoch, i)
        filename = in_dir + "/" + session.format(epoch, i) + ".txt"

        with open(filename) as file:
            parsed_json = json.load(file)

            if 'winner' in parsed_json:
                winner = parsed_json['winner']
            else:
                winner = 0

            calculate_session_stats(epoch_stats, winner, parsed_json['status'])

            history = parsed_json['history']
            invalid_moves = list(filter(lambda turn: not turn["isValid"], history))
            valid_moves = list(filter(lambda turn: turn["isValid"], history))
            p1_moves = list(filter(lambda turn: turn["player"] == 1, valid_moves))
            p2_moves = list(filter(lambda turn: turn["player"] == 2, valid_moves))

            process_player_move(memories, (winner == 1), p1_moves)
            process_player_move(memories, (winner == 2), p2_moves)

            for action in invalid_moves:

                if "choice" not in action:
                    continue

                onehot_player = [0, 0, 0]
                onehot_player[action["player"]] = 1
                state = [action["board"] + onehot_player, action["options"]]
                choice = action["choice"]
                reward = invalid_move_score
                memories.append([state, choice, reward, None])

    return epoch_stats


def make_qlearning_train_step(policy_dqn, target_dqn, loss_fn, optimizer, discount_rate):
    output_space = set(range(9))

    def train_step(feature, label, reward, next_input):

        # Sets model to TRAIN mode
        target_dqn.eval()
        policy_dqn.train()

        x, options = feature

        # Calculate Q Value
        if next_input is None:
            q_value = torch.tensor(reward)
        else:
            with torch.no_grad():
                # next state's reward is subtracted from current state instead of added
                # because next state is the other player's turn. Therefore, if the other
                # player made a successful move, then that is bad for the current player
                next_x, next_options = next_input
                q_value = reward + (discount_rate * target_dqn(next_x).max())

        # Makes predictions
        y = policy_dqn(x)
        yhat = target_dqn(x)

        # Update target_dqn output with q_value
        yhat = yhat.clone()
        yhat[label] = q_value.item()
        # For all invalid options, set reward to -10
        for i in (output_space - set(options)):
            yhat[i] = invalid_move_score

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


def train(model, step, memories):
    model.train()
    losses = []
    for action in reversed(memories):
        state, options = action[0]
        feature = [torch.tensor(state, dtype=torch.float), options]
        # feature = torch.tensor(state, dtype=torch.float)
        label = action[1]

        reward = action[2]

        next_state = action[3]
        next_input = None
        if next_state is not None:
            next_state, next_choice = next_state
            next_input = [torch.tensor(next_state, dtype=torch.float), next_choice]

        loss = step(feature=feature, label=label, reward=reward, next_input=next_input)
        # only calculate losses for AI agent (player1)
        if state[28] == 1:
            losses.append(loss)

    return losses


def cleanup_files(file_dir, pattern):
    if delete_training_files:
        for f in glob.glob(f"{file_dir}/{pattern}"):
            os.remove(f)


def app():
    discovery_rate = 1.0
    decay_rate = 0.99

    learn_rate = 0.00005  # 0.01
    discount_rate = 0.9

    seed = 0
    random = Random(seed)
    model_args = {"random": random}
    policy_dqn = t3.get_model(filename=model_filename, input_args=model_args)
    target_dqn = t3.get_model(filename=None, input_args=model_args)

    loss_fn = nn.HuberLoss(delta=1.0)  # nn.MSELoss()
    optimizer = optim.Adam(policy_dqn.parameters(), lr=learn_rate)
    train_step = make_qlearning_train_step(policy_dqn, target_dqn, loss_fn, optimizer, discount_rate)

    wait_for_server(api_base_url)
    memories = []
    game_stats = t3stats.GameStats(max_epochs=max_epochs, max_sessions=max_sessions)

    for epoch in range(max_epochs):
        target_dqn.load_state_dict(policy_dqn.state_dict())
        run_games(epoch, out_dir, discovery_rate)

        memories *= 0
        epoch_stats = create_memories(memories, epoch, out_dir)
        epoch_stats["exploration_rate"] = discovery_rate

        losses = train(model=policy_dqn, step=train_step, memories=memories)
        epoch_stats["avg_loss"] = mean(losses)

        t3.save_model(policy_dqn, filename=model_filename, archive=False)
        cleanup_files(out_dir, "training-*.txt")

        game_stats.add_epoch_stats(epoch_stats)
        t3stats.save_stats(stats_filename, game_stats)

        reload_model(api_base_url)
        discovery_rate = discovery_rate * decay_rate


# =========================
# Entry Point
# =========================
# with subprocess.Popen(["venv/Scripts/python", f"{player_server_script}"]) as proc:
#    app()
#    print("stopping proc")
#    proc.kill()

app()
