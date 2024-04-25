from flask import Flask, request
import json
import random
import torch
import t3dqn as t3
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class Agent:
    def __init__(self):
        self.model_filename = "./data/model/t3-simple1.pt"
        self.model = None

    def reload(self):
        self.model = t3.get_model(filename=self.model_filename)
        self.model.eval()

    def decide(self, state, exploration_rate, options):
        if exploration_rate is not None and random.random() < exploration_rate:
            # select random action
            return options[random.randrange(len(options))]
        else:
            with torch.no_grad():
                x = torch.FloatTensor(state)
                return torch.argmax(self.model.forward(x)).item()


agent = Agent()
agent.reload()
app = Flask(__name__)  # Flask constructor


def predict(state, exploration_rate, options):
    return agent.decide(state, exploration_rate, options)


# A decorator used to tell the application
# which URL is associated function
@app.route('/ping')
def ping():
    return {'status': 'Service is up.', "alive": True}


@app.route('/model/reload', methods=['POST'])
def reload_model():
    agent.reload()
    return {"message": "Reloaded model"}


@app.route('/player/choice', methods=['POST'])
def player_choice():
    req_payload = request.get_json()

    exploration_rate = req_payload.get('explorationRate')
    options = req_payload.get('options')

    player_id = req_payload.get('playerId')
    board = req_payload.get('board')
    board.append(player_id)

    choice = predict(board, exploration_rate, options)
    return {"choice": choice, "playerId": player_id}


if __name__ == '__main__':
    app.run()
