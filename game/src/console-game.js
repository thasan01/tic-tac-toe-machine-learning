const renderer = require("./renderer/console-renderer");
const game = require("./game");
const states = require("./states/game-states");
const { save } = require("./states/save-result");
const encoder = require("./bitencoder");

function loadPlayer(type, args = {}) {
  if (type === "HumanConsolePlayer")
    Player = require("./player/console-human-player");
  else if (type === "RandomPlayer") Player = require("./player/random-player");
  else throw new Error(`Invalid player type: ${type}`);

  console.log("player args:", args);
  return new Player(args);
}

let init = {
  logic: async ({ session }) => {
    return new Promise(async (resolve, reject) => {
      session.board = new Array(9).fill(0);
      let { players } = session;

      try {
        let argv = require("minimist")(process.argv.slice(2));
        console.log("argv: ", argv);

        let { player1Type, player2Type, sessionName, outdir } = argv;
        players.push(loadPlayer(player1Type, argv));
        players.push(loadPlayer(player2Type, argv));

        session.sessionName = sessionName;
        session.outdir = outdir;
        session.activePlayer = 0;

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
