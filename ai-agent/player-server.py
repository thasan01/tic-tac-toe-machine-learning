from flask import Flask, request
import json
import torch
import t3dqn as t3
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

model_filename = "./data/model/t3-simple.pt"
model = t3.get_model(filename=model_filename)
model.eval()


def predict(model, state):
    x = torch.FloatTensor(state)
    return torch.argmax(model.forward(x))


app = Flask(__name__)  # Flask constructor


# A decorator used to tell the application
# which URL is associated function
@app.route('/ping')
def ping():
    return {'status': 'Service is up.', "alive": True}


@app.route('/model/reload', methods=['POST'])
def reload_model():
    model = t3.get_model(filename=model_filename)
    model.eval()
    return {"message": "Reloaded model"}


@app.route('/player/choice', methods=['POST'])
def player_choice():
    req_payload = request.get_json()

    player_id = req_payload.get('playerId')
    board = req_payload.get('board')
    board.append(player_id)
    choice = predict(model, board)
    return {"choice": choice.item(), "playerId": player_id}


if __name__ == '__main__':
    app.run()
