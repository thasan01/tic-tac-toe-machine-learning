function renderBoard(board) {
  line = "";
  border = "------------";
  sign = [" ", "X", "O"];

  console.log(border);
  for (let i = 0; i < board.length; i++) {
    if (i % 3 == 0) {
      if (line !== "") {
        console.log(line);
        console.log(border);
      }
      line = "| ";
    }
    line = line + sign[board[i]] + " | ";
  }
  console.log(line);
  console.log(border);
}

function render(session) {
  let { activePlayer, board, status, gameover } = session;

  if (gameover) {
    console.log(status);
  } else {
    let playerId = activePlayer + 1;
    console.log(`Entering Player[${playerId}]'s turn`);
  }

  renderBoard(board);
  console.log("");
}

module.exports = render;
