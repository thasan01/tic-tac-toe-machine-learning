const game = require("./game");
const states = require("./states/game-states");

let init = {
  logic: async ({ session }) => {
    return new Promise(async (resolve, reject) => {
      session.board = new Array(9).fill(0);
      let { players } = session;

      try {
        players.push(null);
        players.push(null);

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

console.log({ game, states });
