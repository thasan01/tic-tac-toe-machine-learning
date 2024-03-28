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
        let content = {
          winner: session.winner,
          status: session.status,
          history: session.history,
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
