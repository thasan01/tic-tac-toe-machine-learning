# Tic-Tac-Toe Machine Learning Example

This repository contains Javascript implementation of Tic Tac Toe game that leverages a Python based of player agent.

# The Game

There are two types of games: web and console. The web game is not been implemented yet.

## Console Game

The console based game is invoked through NodeJS CLI.

To build the games, run the following command from the root project folder:
```
npm run install --prefix ./game
npm run build --prefix ./game
```

To run the game, run the following command from the root project folder.
For an AI agent vs Random agent: 
```
node ./game/build/tic-tac-toe.console.js --player1Type RLWebAgentPlayer --player1Profile rl-agent-1 --explorationRate 0.0 --player2Type RandomPlayer --trueRandomRate 0.25 --sessionName game1 --configDir game/config
```

For an AI agent vs Human player: 
```
node ./game/build/tic-tac-toe.console.js --player1Type RLWebAgentPlayer --player1Profile rl-agent-1 --explorationRate 0.0 --player2Type HumanConsolePlayer --sessionName game1 --configDir game/config
```


## Input Arguments

Following are list of input arguments can be passed into the game.

<table>
  <tr>
    <td>Name</td>
    <td>Description</td>
    <td>Required</td>
    <td>Allowed Values</td>
  </tr>

  <tr>
    <td>player1Type</td>
    <td>Specifies the type of player for Player 1</td>
    <td>Yes</td>
    <td><ul> <li>HumanConsolePlayer</li> <li>HumanWebPlayer</li> <li>RandomPlayer</li> <li>RLWebAgentPlayer</li> </ul></td>
  </tr>

  <tr>
    <td>player2Type</td>
    <td>Specifies the type of player for Player 2</td>
    <td>Yes</td>
    <td><ul> <li>HumanConsolePlayer</li> <li>HumanWebPlayer</li> <li>RandomPlayer</li> <li>RLWebAgentPlayer</li> </ul></td>
  </tr>

  <tr>
    <td>sessionName</td>
    <td>An unique identifier for the game session. It is used in the output file name. If `sessionName` is provided, then `outdir` must also be provided.<br/> Output file = ${outdir}/${sessionName}.txt</td>
    <td>No</td>
    <td>*</td>
  </tr>

  <tr>
    <td>outdir</td>
    <td>The file path (relative or absolute) where to game result is saved. If `outdir` is provided, then `sessionName` must also be provided. <br/> Output file = ${outdir}/${sessionName}.txt</td>
    <td>No</td>
    <td>*</td>
  </tr>

  <tr>
    <td>configDir</td>
    <td>The file path (relative or absolute) where to game's config file. The config file contains default values of common properties. They can be overridden by passing them from command line. If calling from the project root location, the value is: <b>game/config</b></td>
    <td>Yes</td>
    <td>*</td>          
  </tr>

  <tr>
    <td>encoder</td>
    <td>This param specifies whether to encode the game board state.</td>
    <td>No</td>
    <td><ul><li>None</li><li>BitEncoder</li><li>OneHotEncoder</li></ul></td>
  </tr>

  <tr>
    <td>trueRandomRate</td>
    <td>The default behavior for a Random Player Agent is to select an option from the remaining available options provided by the game. If this parameter provides a chance for the agent to select a random option from the board instead. In other words, this flag will enable the agent to make invalid moves. This is used to punish the AI agent during the training process. If this parameter is not provided, the default value is 0.</td>
    <td>No</td>
    <td>Real number between 0 and 1</td>          
  </tr>

  <tr>
    <td>suppressOutput</td>
    <td>If this flag is provided, the game will not call the renderer class, to output the state of the board per turn. This is used during the training process to reduce I/O overhead and increase performance.</td>
    <td>No</td>
    <td>N/A</td>          
  </tr>

  <tr>
    <td>invalidChoiceThreshold</td>
    <td>Number of times an agent can make an invalid value before being disqualified. The default value is 5.</td>
    <td>No</td>
    <td>Integer</td>          
  </tr>

  <tr>
    <td>sameInvalidChoiceThreshold</td>
    <td>Number of times an agent can the same invalid value before being disqualified. The default value is 2.</td>
    <td>No</td>
    <td>Integer</td>          
  </tr>

  <tr>
    <td>explorationRate</td>
    <td>This rate is used by the RLWebAgentPlayer during the prediction step. It controls the logic for the agent use the neural network to predict a choice (exploitation) or randomly select a choice (exploration). The higher the number, the more likely the agent will use exploration. If no value is provided the agent will fall back to the default value of 0.</td>
    <td>No</td>
    <td>Real number between 0 and 1</td>          
  </tr>


</table>
