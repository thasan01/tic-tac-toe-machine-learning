function processTransitions(trans, session) {
  if (Array.isArray(trans)) {
  } else if (typeof trans === "object") {
    for (let [key, value] of Object.entries(trans)) {
      if (value(session)) return key;
    }
  }
  return null;
}

let defaultEncoder = { encode: (array) => [...array], decode: (bits) => bits };

let game = async ({
  renderer,
  encoder = defaultEncoder,
  states,
  initialState,
}) => {
  let currentState = states[initialState];

  var session = {
    currentState,
    board: null,
    players: [],
    history: [],
  };

  let fatalError = false;

  while (currentState.transitions) {
    let promise = currentState.logic({ renderer, encoder, session });
    let newState = null;
    let breakLoop = false;

    await promise
      .then(() => {
        newState = processTransitions(currentState.transitions, session);
      })
      .catch((err) => {
        console.error(err instanceof Error ? err.message : String(err));
        fatalError = true;
      })
      .finally(() => {
        if (newState != null) currentState = states[newState];
        else breakLoop = true;
      });

    if (breakLoop) break;
  }

  if (!fatalError) {
    currentState.logic({ renderer, encoder, session });
  }
};

async function gameBatch(configs, { concurrency = Infinity } = {}) {
  if (concurrency === Infinity || concurrency >= configs.length) {
    return Promise.all(configs.map((cfg) => game(cfg)));
  }

  const results = new Array(configs.length);
  let nextIndex = 0;

  async function worker() {
    while (nextIndex < configs.length) {
      const i = nextIndex++;
      results[i] = await game(configs[i]);
    }
  }

  await Promise.all(Array.from({ length: concurrency }, worker));
  return results;
}

module.exports = { game, gameBatch };
