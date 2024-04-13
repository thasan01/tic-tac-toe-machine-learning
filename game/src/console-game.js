const renderer = require("./renderer/console-renderer");
const game = require("./game");
const states = require("./states/game-states");
const { save } = require("./states/save-result");

const argv = require("minimist")(process.argv.slice(2));
const encoder =
  argv.encoder === "BitEncoder" ? require("./bitencoder") : undefined;

function loadPlayer(type, args = {}) {
  if (type === "HumanConsolePlayer")
    Player = require("./player/console-human-player");
  else if (type === "RandomPlayer") Player = require("./player/random-player");
  else throw new Error(`Invalid player type: ${type}`);

  return new Player(args);
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
      } = argv;

      try {
        players.push(loadPlayer(player1Type, argv));
        players.push(loadPlayer(player2Type, argv));

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
