const Player = require("./player");

class RandomPlayer extends Player {
  constructor({ trueRandom = false } = { trueRandom: false }) {
    super();
    this.trueRandom = trueRandom;
    console.log("random player init: ", { trueRandom });
  }

  choose(board, options) {
    let len = this.trueRandom ? board.length : options.length;
    let index = Math.floor(Math.random() * len);
    return options[index];
  }
}

module.exports = RandomPlayer;
