import torch
from torch import nn
import torch.nn.functional as F
import os
import time
import zipfile


# Tic-Tac-Toe Deep Q Network Model
class Model(nn.Module):
    def __init__(self, input_nodes, hidden_layer1_nodes, output_nodes):
        super().__init__()
        # fully connected network
        self.fc1 = nn.Linear(input_nodes, hidden_layer1_nodes)
        self.fc2 = nn.Linear(hidden_layer1_nodes, hidden_layer1_nodes)
        self.fc3 = nn.Linear(hidden_layer1_nodes, hidden_layer1_nodes)
        self.fc4 = nn.Linear(hidden_layer1_nodes, hidden_layer1_nodes)
        self.out = nn.Linear(hidden_layer1_nodes, output_nodes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))  # Apply rectified linear unit (ReLU) activation
        x = self.out(x)  # Calculate output
        return x


def get_model(filename=None):
    model = None

    if filename is not None:
        try:  # Attempt to load the model if it exists
            model = torch.load(filename)
        except FileNotFoundError:
            # if it doesn't exist, then create  new one
            model = Model(10, 18, 9)
    else:
        model = Model(10, 18, 9)

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
