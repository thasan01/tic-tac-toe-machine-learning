import torch
from torch import nn
import torch.nn.functional as F
import os
import time
import zipfile
from utils.random_util import Random

output_space = set(range(9))


# Tic-Tac-Toe Deep Q Network Model
class Model(nn.Module):
    def __init__(self, input_nodes, layer_sizes, output_nodes, dropout=0.1, random=Random()):
        super(Model, self).__init__()
        self.output_nodes = output_nodes
        sizes = [input_nodes] + layer_sizes + [output_nodes]
        self.fc_layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])
        self.dropout = nn.Dropout(dropout)
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.random = random

    def forward(self, x):
        for layer in self.fc_layers[:-1]:
            x = self.leakyReLU(layer(x))
            x = self.dropout(x)
        x = self.fc_layers[-1](x)
        return x

    def predict(self, state, exploration_rate, options=None):
        if exploration_rate is not None and self.random.fraction() < exploration_rate:
            # select random action
            return options[self.random.range(len(options))]
        else:
            with torch.no_grad():
                x = torch.FloatTensor(state)
                y = self.forward(x)
                # Apply mask on the output if valid options exists
                if options is not None:
                    for i in (output_space - set(options)):
                        y[i] = float("-inf")

                return torch.argmax(y).item()


def get_model(filename=None, input_args=None):

    if input_args is None:
        input_args = {}

    default_args = {"input_nodes": 30, "layer_sizes": [120, 840, 120], "output_nodes": 9}
    model_args = {**default_args, **input_args}

    if filename is not None:
        try:  # Attempt to load the model if it exists
            model = torch.load(filename, weights_only=False)
        except FileNotFoundError:
            # if it doesn't exist, then create  new one
            model = Model(**model_args)
    else:
        model = Model(**model_args)

    return model


def archive_file(filepath):
    # Check if file exists
    if os.path.isfile(filepath):
        # Get the current Unix timestamp
        timestamp = str(int(time.time()))

        # Split the file path into directory, filename and extension
        directory, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)

        # Create the archived filename with the timestamp
        archived_filename = f"{filename}_{timestamp}{extension}.zip"
        archived_filepath = os.path.join(directory, archived_filename)

        # Archive and compress the file
        with zipfile.ZipFile(archived_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(filepath, arcname=os.path.basename(filepath))


def save_model(model, filename, archive=False):
    if archive:
        archive_file(filename)

    torch.save(model, filename)
