// Entry point for the CLI game. Parses arguments, wires up players and states,
// then launches either a single game or a concurrent batch of games.
const renderer = require("./renderer/console-renderer");
const { game, gameBatch } = require("./game");
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

// Config file values act as defaults; CLI args override them.
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
  else if(type == "HeuristicPlayer")
    //
    Player = require("./player/heuristic-player");
  else throw new Error(`Invalid player type: ${type}`);

  return new Player(args, playerId);
}

// Returns a fresh init state object bound to the given session parameters.
// Called once per session so that batch runs each get their own players,
// sessionName, and board — rather than all sessions sharing a single closure.
function makeInit(params) {
  return {
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
        } = params;

        try {
          players.push(loadPlayer(player1Type, params, 1));
          players.push(loadPlayer(player2Type, params, 2));

          // register() pings the model server; throws ServerConnectionError if unreachable.
          await players[0].register();
          await players[1].register();

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
}

// Redirect the end-of-game path through the save state so results are
// written to disk before the final board render. The original turn→end
// shortcut is removed so every completed game passes through save first.
states["save"] = save;
delete states["turn"].transitions["end"];
states["turn"].transitions["save"] = ({ gameover }) => gameover;

const numSessions = parseInt(inputArgs.sessions, 10) || 1;
const concurrency = parseInt(inputArgs.concurrency, 10) || Infinity;

if (numSessions <= 1) {
  const singleStates = { ...states, init: makeInit(inputArgs) };
  game({ renderer, encoder, states: singleStates, initialState: "init" });
} else {
  // Batch mode: run N sessions concurrently. Each gets a suffixed sessionName
  // (e.g. training-000001-000000 through training-000001-000099) and its own
  // init state, but shares the stateless turn/save/end objects safely.
  const configs = Array.from({ length: numSessions }, (_, i) => {
    const sessionName = `${inputArgs.sessionName}-${String(i).padStart(6, "0")}`;
    const sessionParams = { ...inputArgs, sessionName };
    const sessionStates = { ...states, init: makeInit(sessionParams) };
    return { renderer, encoder, states: sessionStates, initialState: "init" };
  });
  gameBatch(configs, { concurrency }).catch((err) => {
    console.error("Batch game error:", err);
    process.exit(1);
  });
}
