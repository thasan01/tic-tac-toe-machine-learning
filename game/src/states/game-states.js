function resetInvalidRecord({ player, choice } = { player: -1, choice: -1 }) {
  return {
    player,
    choice,
    choiceCounter: 0,
    sameChoiceCounter: 0,
    disqualified: false,
  };
}

function validateChoice(board, choice, session) {
  let invalid = board[choice] != 0;
  if (invalid) {
    const { invalidChoiceThreshold, sameInvalidChoiceThreshold } = session;
    let player = session.activePlayer;
    let rec = session.invalidRecord;

    //If invalid record is not defined or a different player made an invalid choice
    //then create a new record for the player / choice
    if (!rec || rec.player !== player) {
      session.invalidRecord = resetInvalidRecord({ player, choice });
    }
    //
    if (rec.choice === choice) rec.sameChoiceCounter++;

    rec.choiceCounter++;

    if (
      rec.choiceCounter >= invalidChoiceThreshold ||
      rec.sameChoiceCounter >= sameInvalidChoiceThreshold
    )
      rec.disqualified = true;
  }

  return invalid;
}

function getOptions(board) {
  return board
    .map((item, idx) => (item === 0 ? idx : -1))
    .filter((item) => item > -1);
}

function isEndGame(board) {
  let x = -1;
  let t = board.map((i) => (i === 0 ? --x : i));

  let flag = t[0] === t[1] && t[1] === t[2]; //1st row check
  flag |= t[3] === t[4] && t[4] === t[5]; //2nd row check
  flag |= t[6] === t[7] && t[7] === t[8]; //3rd row check

  flag |= t[0] === t[3] && t[3] === t[6]; //1st column check
  flag |= t[1] === t[4] && t[4] === t[7]; //2nd column check
  flag |= t[2] === t[5] && t[5] === t[8]; //3rd column check

  flag |= t[0] === t[4] && t[4] === t[8]; //left - right diagonal check
  flag |= t[2] === t[4] && t[4] === t[6]; //right - left diagonal check

  return flag;
}

//====================
// Define Game States
//====================

let turn = {
  logic: async ({ renderer, encoder, session }) => {
    return new Promise(async (resolve, reject) => {
      let { players, activePlayer, board, suppressOutput } = session;
      let player = players[activePlayer];

      let options = getOptions(board);
      let choice = null;

      if (!suppressOutput) renderer(session);

      //If no option is available, then its a draw
      if (!Array.isArray(options) || !options.length) {
        session.gameover = true;
        session.status = `Game is draw!`;
        resolve();
      } else {
        session.invalidRecord = resetInvalidRecord();
        let isInvalid = false;
        do {
          choice = await player.choose([...board], options);
          isInvalid = validateChoice(board, choice, session);

          if (session.invalidRecord.disqualified) {
            session.status = `Player[${activePlayer}] disqualified!`;
            session.winner = null;
            session.gameover = true;
            return resolve();
          }

          session.history.push({
            player: activePlayer,
            board: encoder.encode(board),
            choice,
            isValid: !isInvalid,
          });
        } while (isInvalid);

        //commit the choice
        let playerId = activePlayer + 1;
        session.board[choice] = playerId;
        session.activePlayer = playerId % players.length;
        session.gameover = isEndGame(board);

        if (session.gameover) {
          session.status = `Player[${playerId}] won!`;
          session.winner = activePlayer;
        }

        resolve();
      }
    });
  },
  transitions: {
    turn: ({ gameover }) => !gameover,
    end: ({ gameover }) => gameover,
  },
};

let end = {
  logic: async ({ renderer, session }) => {
    if (!session.suppressOutput) renderer(session);
  },
};

module.exports = { turn, end };
