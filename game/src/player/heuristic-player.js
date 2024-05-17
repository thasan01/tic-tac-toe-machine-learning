const Player = require("./player");

const streaks = [
    [0,1,2],[3,4,5],[6,7,8],  // horizontal streaks
    [0,3,6],[1,4,7],[2,5,8], // vertical streaks
    [0,4,8], [2,4,6] //cross streaks
];

function findEmptySpot(board, streak){
    for (let i in streak)

        if( board[streak[i]]===0 )
            return streak[i];

    throw new Error(`Logic Error: could not find free spot in board: ${board} with streak: ${streak}`);
}

function checkStreak(board, streak, playerId, otherId){
    bucket = [0,0,0]
    for (let i in streak){
        bucket[board[streak[i]]]++
    }

    let win = [false, -1];
    let lose = [false, -1];

    //streak has exactly 1 free space
    if(bucket[0] == 1){
      if(bucket[playerId] == 2)
        win = [true, findEmptySpot(board, streak)];
      else if(bucket[otherId] == 2)
        lose = [true, findEmptySpot(board, streak)];
    }

    return [win, lose];
}


class HeuristicPlayer extends Player {

  constructor(args, playerId) {
    super();
    this.playerId = playerId;
    this.otherId = (playerId == 1 ? 2 : 1);
  }

  register() {
    return true;
  }

  choose(board, options) {

    let win = [false, -1];
    let lose = [false, -1]

    for(let i in streaks){

        let [_win, _lose] = checkStreak(board, streaks[i], this.playerId, this.otherId);
    
        if(_win[0])
            win = _win;
    
        if(_lose[0])
            lose = _lose;
    }
    
    //If player is about to win, then make the winning move
    if(win[0])
        return win[1];
    
    //If player is about to lose, then prevent losing
    else if(lose[0])
        return lose[1];

    //else make a random valid move
    return options[Math.floor(Math.random() * options.length)];    
  }
}


module.exports = HeuristicPlayer;
