const renderer = require("./renderer/console-renderer");
const game = require("./game");
const states = require("./states/game-states");
const { save } = require("./states/save-result");

const argv = require("minimist")(process.argv.slice(2));

let encoder = undefined;
switch (argv.encoder) {
  case "BitEncoder":
    encoder = require("./bitencoder");
    break;
  case "OneHotEncoder":
    encoder = require("./onehotencoder");
    break;
}

let config = {};
if (argv.configDir) {
  process.env["NODE_CONFIG_DIR"] = argv.configDir;
  config = require("config");
}
let inputArgs = { ...config, ...argv };

function loadPlayer(type, args, playerId) {
  if (type === "HumanConsolePlayer")
    //
    Player = require("./player/console-human-player");
  else if (type === "RandomPlayer")
    //
    Player = require("./player/random-player");
  else if (type === "RLWebAgentPlayer")
    //
    Player = require("./player/web-player");
  else throw new Error(`Invalid player type: ${type}`);

  return new Player(args, playerId);
}

let init = {
  logic: async ({ session }) => {
    return new Promise(async (resolve, reject) => {
      session.board = new Array(9).fill(0);
      let { players } = session;
      let {
        player1Type,
        player2Type,
        sessionName,
        outdir,
        suppressOutput,
        invalidChoiceThreshold = 5,
        sameInvalidChoiceThreshold = 2,
      } = inputArgs;

      try {
        players.push(loadPlayer(player1Type, inputArgs, 1));
        players.push(loadPlayer(player2Type, inputArgs, 2));

        players[0].register();
        players[1].register();

        session.sessionName = sessionName;
        session.outdir = outdir;
        session.activePlayer = 0;
        session.suppressOutput = suppressOutput;
        session.invalidChoiceThreshold = invalidChoiceThreshold;
        session.sameInvalidChoiceThreshold = sameInvalidChoiceThreshold;

        resolve();
      } catch (err) {
        reject(err);
      }
    });
  },

  transitions: { turn: () => true },
};

states["init"] = init;
states["save"] = save;

delete states["turn"].transitions["end"];
states["turn"].transitions["save"] = ({ gameover }) => gameover;

game({ renderer, encoder, states, initialState: "init" });
