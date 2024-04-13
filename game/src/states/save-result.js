const fs = require("node:fs");

let save = {
  logic: async ({ session }) => {
    return new Promise(async (resolve, reject) => {
      let { outdir, sessionName } = session;

      if (!outdir || !sessionName)
        return reject(
          new Error(
            "Unable to write to file because outdir or session name are invalid."
          )
        );

      let filename = `${outdir}/${sessionName}.txt`;

      try {
        let allValidMoves = session.history.filter((e) => e.isValid);
        let p1 = allValidMoves.filter((e) => e.player === 0).length;
        let p2 = allValidMoves.filter((e) => e.player === 1).length;

        let content = {
          winner: session.winner,
          status: session.status,
          history: session.history,
          validMoves: [p1, p2],
        };
        fs.writeFileSync(filename, JSON.stringify(content));
      } catch (err) {
        return reject(err);
      }
      resolve();
    });
  },
  transitions: {
    end: () => true,
  },
};

module.exports = { save };
