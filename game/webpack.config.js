const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = [
  {
    name: "console",
    entry: "./src/console-game.js",
    target: "node",
    output: {
      path: path.resolve(__dirname, "build"),
      filename: "tic-tac-toe.console.js",
    },
  },

  {
    name: "web",
    entry: "./src/web-game.js",
    target: "node",
    output: {
      path: path.resolve(__dirname, "build"),
      filename: "tic-tac-toe.web.js",
    },
  },
];
