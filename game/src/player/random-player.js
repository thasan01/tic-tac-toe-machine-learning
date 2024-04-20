const Player = require("./player");

class RandomPlayer extends Player {
  constructor({ trueRandomRate = 0.0 } = { trueRandomRate: 0.0 }) {
    super();
    this.trueRandomRate = trueRandomRate;
  }

  register() {
    return true;
  }

  choose(board, options) {
    let rnum = Math.random();
    return rnum < this.trueRandomRate
      ? Math.round(rnum * board.length)
      : options[Math.round(rnum * options.length)];
  }
}

module.exports = RandomPlayer;
