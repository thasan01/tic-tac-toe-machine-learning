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

  let newState = null;
  while (currentState.transitions) {
    let promise = currentState.logic({ renderer, encoder, session });

    await promise
      .then((resp) => {
        let trans = currentState.transitions;
        newState = processTransitions(trans, session);
      })
      .catch((err) => {
        console.error("Encountered Error: ", err.message);   
      })
      .finally(() => {
        if (newState != null) {
          currentState = states[newState];
        }

        //Current State has transitions defined but none of them were acivated,
        //so setting transitions to null to break out of the loop.
        else currentState.transitions = null;
      });
  }

  if (newState === "end"){
    currentState.logic({ renderer, encoder, session })
      .catch(err => console.error("Encountered Error: ", err.message));   
  }

};

module.exports = game;
