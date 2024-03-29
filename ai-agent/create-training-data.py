import subprocess
import json

max_sessions = 1
out_dir = "./data/training"
filename_template = "training-{:06d}"

'''
for i in range(max_sessions):
    session = filename_template.format(i)
    subprocess.run(["node", "../game/build/tic-tac-toe.console.js",
                    "--player1Type", "RandomPlayer",
                    "--player2Type", "RandomPlayer",
                    "--trueRandom",
                    "--outdir", out_dir,
                    "--sessionName", session]
                   )
'''

for i in range(max_sessions):
    filename = out_dir + "/" + filename_template.format(i) + ".txt"
    print(f"Processing {filename}")

    with open(filename) as file:
        parsed_json = json.load(file)
        winner = parsed_json['winner']
        history = parsed_json['history']

        for action in history:
            print(f"action: {action}")

