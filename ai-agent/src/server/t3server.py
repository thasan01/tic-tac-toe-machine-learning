import threading
from flask import Flask, request
import src.model.t3dqn as t3
from t3encoder import encode
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)

app = Flask(__name__)  # Flask constructor
shutdown_event = threading.Event()
agent = None
model_dir = None

class Agent:
    def __init__(self, model):
        self.model = model

    def reload(self, model_dir:str):
        self.model, _ = t3.load_model(model_dir, is_inference=True)
        self.model.eval()

    def decide(self, state, exploration_rate, options):
        return self.model.inference(state, exploration_rate, options)


# A decorator used to tell the application
# which URL is associated function
@app.route('/ping')
def ping():
    return {'status': 'Service is up.', "alive": True}


@app.route('/model/reload', methods=['POST'])
def reload_model():
    agent.reload(model_dir)
    return {"message": "Reloaded model"}


@app.route('/player/choice', methods=['POST'])
def player_choice():
    req_payload = request.get_json()

    exploration_rate = req_payload.get('explorationRate')
    options = req_payload.get('options')

    player_id = req_payload.get('playerId')
    board = req_payload.get('board')

    choice = agent.decide(state=encode(board, player_id), exploration_rate=exploration_rate, options=options)
    return {"choice": choice, "playerId": player_id}

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_event.set()  # Set the shutdown event
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()  # Call the shutdown function to stop the server
    return {"status": "Server is shutting down."}

if __name__ == '__main__':
    model_dir = "./data/model/t3"
    agent = Agent(model_dir)
    agent.reload(model_dir)

    server_hostname = '127.0.0.1'
    server_port = 5000
    app.run(host=server_hostname, port=server_port)
