import torch
from torch import nn
import os
import os.path as path
from randomutil import Random

DEFAULT_MODEL_FILENAME = "t3-model.pth"
DEFAULT_TRAINING_FILENAME = "t3-training.pth"
DEFAULT_RELU_RATE = 0.1
DEFAULT_DROPOUT_RATE = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_space = set(range(9))

# Tic-Tac-Toe Deep Q Learning Network
class T3DQNet(nn.Module):
    def __init__(self,
                 num_input_nodes:int,
                 num_hidden_layer_nodes:list[int],
                 num_output_nodes:int,
                 random:Random=Random(),
                 relu_rate:float = DEFAULT_RELU_RATE,
                 dropout_rate:float = DEFAULT_DROPOUT_RATE):
        super(T3DQNet, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.num_hidden_layer_nodes = num_hidden_layer_nodes
        self.num_output_nodes = num_output_nodes
        self.random = random
        self.relu_rate = relu_rate
        self.dropout_rate = dropout_rate

        # This list will hold the layers of the neural network
        layers = []

        # Add the first hidden layer
        if num_hidden_layer_nodes:
            layers.append(nn.Linear(num_input_nodes, num_hidden_layer_nodes[0]))
            layers.append(nn.LeakyReLU(self.relu_rate))
            layers.append(nn.Dropout(self.dropout_rate))

            # Add subsequent hidden layers
            for i in range(len(num_hidden_layer_nodes) - 1):
                layers.append(nn.Linear(num_hidden_layer_nodes[i], num_hidden_layer_nodes[i + 1]))
                layers.append(nn.BatchNorm1d(num_hidden_layer_nodes[i + 1]))
                layers.append(nn.LeakyReLU(self.relu_rate))
                layers.append(nn.Dropout(self.dropout_rate))

            # Add the output layer
            layers.append(nn.Linear(num_hidden_layer_nodes[-1], num_output_nodes))
        else:  # No hidden layers
            layers.append(nn.Linear(num_input_nodes, num_output_nodes))

        # Combine all the layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # The forward pass simply passes the input through the sequential network
        x = self.network(x)
        return x

    def inference(self, state, exploration_rate, options=None):
        if exploration_rate is not None and self.random.fraction() < exploration_rate:
            # select random action
            return options[self.random.range(len(options))]
        else:
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32).to(device)
                # Reshape the tensor to match the expected batch dimension
                x = x.reshape(1, len(state))

                y = self.forward(x)
                # Remove the batch dimension before further processing
                y = y.squeeze(0)

                # Apply mask on the output if valid options exists
                if options is not None:
                    output_options_tensor = torch.tensor(list(output_space - set(options)), dtype=torch.long).to(device)
                    y.index_fill_(0, output_options_tensor, float("-inf"))

                return torch.argmax(y).item()


def load_model(model_dir:str, **kwargs):
    is_inference = kwargs.get("is_inference", False)
    model_filename = path.join(model_dir, DEFAULT_MODEL_FILENAME)
    if path.exists(model_filename):
        model_config = torch.load(model_filename)
        model = T3DQNet(model_config["num_input_nodes"],
                     model_config["num_hidden_layer_nodes"],
                     model_config["num_output_nodes"],
                     relu_rate=model_config["relu_rate"],
                     dropout_rate=model_config["dropout_rate"])

        if "model_state" in model_config:
            model.load_state_dict(model_config["model_state"])
    else:
        req_keys = ["num_input_nodes", "num_hidden_layer_nodes", "num_output_nodes"]
        if not all(key in kwargs for key in req_keys):
            raise KeyError(f"Missing one or more of the following required inputs: {req_keys}")

        model = T3DQNet(kwargs.get("num_input_nodes"),
                     kwargs.get("num_hidden_layer_nodes"),
                     kwargs.get("num_output_nodes"),
                     relu_rate=kwargs.get("relu_rate", DEFAULT_RELU_RATE),
                     dropout_rate=kwargs.get("dropout_rate", DEFAULT_DROPOUT_RATE))

    ret_config = {}
    if not is_inference:
        training_filename = path.join(model_dir, DEFAULT_TRAINING_FILENAME)
        if path.exists(training_filename):
            training_config = torch.load(training_filename)
            if "optimizer_state" in training_config: ret_config["optimizer_state"] = training_config["optimizer_state"]
            if "epoch" in training_config: ret_config["epoch"] = training_config["epoch"]
            if "loss" in training_config: ret_config["loss"] = training_config["loss"]
            if "exploration_rate" in training_config: ret_config["exploration_rate"] = training_config["exploration_rate"]
            if "exploration_decay" in training_config: ret_config["exploration_decay"] = training_config["exploration_decay"]
            if "experience_replay" in training_config: ret_config["experience_replay"] = training_config["experience_replay"]

    model = model.to(device)
    return model, ret_config

def save_model_checkpoint(model_dir:str, model:T3DQNet, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    model_config = {
        "num_input_nodes" : model.num_input_nodes,
        "num_hidden_layer_nodes": model.num_hidden_layer_nodes,
        "num_output_nodes" :model.num_output_nodes,
        "relu_rate" :model.relu_rate,
        "dropout_rate" : model.dropout_rate,
        "model_state": model.state_dict()
    }
    torch.save(model_config, path.join(model_dir, DEFAULT_MODEL_FILENAME))

    training_config = {
        "optimizer_state": kwargs.get("optimizer_state") if "optimizer_state" in kwargs else None,
        "epoch": kwargs.get("epoch", 0),
        "loss": kwargs.get("loss", None),
        "exploration_rate": kwargs.get("exploration_rate"),
        "exploration_decay": kwargs.get("exploration_decay"),
        "experience_replay": kwargs.get("experience_replay")
    }
    torch.save(training_config, path.join(model_dir, DEFAULT_TRAINING_FILENAME))
