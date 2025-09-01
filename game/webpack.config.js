const path = require("path");
const nodeExternals = require("webpack-node-externals");

module.exports = [
  // This configuration is correct for a Node.js console app. No changes needed here.
  {
    name: "console",
    mode: "production",
    entry: "./src/console-game.js",
    target: "node",
    externals: [nodeExternals()],
    output: {
      path: path.resolve(__dirname, "build"),
      filename: "tic-tac-toe.console.js",
    },
  },

  // This configuration is for the web browser. It needs to be corrected.
  {
    name: "web",
    mode: "production",
    entry: "./src/web-game.js",
    target: "web", 
    output: {
      path: path.resolve(__dirname, "build"),
      filename: "tic-tac-toe.web.js",
    },
  },
];