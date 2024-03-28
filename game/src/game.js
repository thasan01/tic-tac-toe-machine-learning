function processTransitions(trans, session) {
  if (Array.isArray(trans)) {
  } else if (typeof trans === "object") {
    for (let [key, value] of Object.entries(trans)) {
      if (value(session)) return key;
    }
  }
  return null;
}

let defaultEncoder = { encode: (array) => array, decode: (bits) => bits };

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
  };

  while (currentState.transitions) {
    let promise = currentState.logic({ renderer, encoder, session });
    let newState = null;

    await promise
      .then((resp) => {
        let trans = currentState.transitions;
        newState = processTransitions(trans, session);
      })
      .catch((resp) => {
        console.log("Encountered Error: ", resp);
      })
      .finally(() => {
        if (newState != null) currentState = states[newState];
        //Current State has transitions defined but none of them were acivated,
        //so setting transitions to null to break out of the loop.
        else currentState.transitions = null;
      });
  }
  currentState.logic({ renderer, encoder, session });
};

module.exports = game;
