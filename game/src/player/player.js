class Player {
  async register() {
    throw new Error("Method 'register()' must be implemented.");
  }

  async choose(board, options) {
    throw new Error("Method 'choose()' must be implemented.");
  }
}

module.exports = Player;
