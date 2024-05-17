# AI Agent Module

This module contains the python scripts to create, train and deploy the RL based AI Agent

## Project Setup

To build the project, first create a Python Virtual Environment (venv). Use an IDE like Pycharm or run the following commands:
```
cd ai-agent
python -m venv ./venv
```

Then activate the venv:

Windows:
```
cd ai-agent
"venv/Scripts/activate"
```

Linux:
```
cd ai-agent
source venv/bin/activate
```

Finally, install the dependencies:
```
pip install -r requirements.txt
```


## Train Model

To train the model, active the venv and run the following command:
```
python train-model.py
```

Alternatively, download the latest [pretrained model file]([https://github.com/thasan01/tic-tac-toe-machine-learning/releases/download/1.0/t3-trained-model.zip](https://github.com/thasan01/tic-tac-toe-machine-learning/releases/download/1.0/t3-trained-model-v0.2.zip)) and place it in the location: **ai-agent/data/model**
