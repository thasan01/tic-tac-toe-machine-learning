const prompts = require("prompts");
const Player = require("./player");

class ConsoleHumanPlayer extends Player {
  //
  async register() {
    return true;
  }

  //
  async choose(board, options) {
    const response = await prompts({
      type: "number",
      name: "value",
      message: "Enter an index of the row/column:",
      validate: (value) =>
        options.indexOf(value) < 0
          ? `Invalid value, avaliable options are: ${options}`
          : true,
    });

    let { value: choice } = response;
    return choice;
  }
}

module.exports = ConsoleHumanPlayer;
