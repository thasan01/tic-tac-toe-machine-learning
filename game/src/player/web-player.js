const Player = require("./player");
//Endpoint Context urls
const pingUrl = "/ping";
const choiceUrl = "/player/choice";
const reloadUrl = "/model/reload";
//
const jsonHeaders = {
  Accept: "application/json",
  "Content-Type": "application/json",
};

function getProfile(args, playerId) {
  return args && args.profiles ? args.profiles[playerId] : null;
}

class WebPlayer extends Player {
  constructor(args, playerId, profileId) {
    super();
    let profile = getProfile(args, profileId);

    if (!profile)
      throw new Error(`Unable to find profile for player[${playerId}]`);

    this.playerId = playerId;
    this.baseUrl = profile.baseUrl;
    this.explorationRate = args.explorationRate || profile.explorationRate || 0.0;
  }

  async register() {
    const response = await fetch(`${this.baseUrl}${pingUrl}`);
    let body = await response.json();
    return body.alive;
  }

  async choose(board, options) {
    let requestPayload = {
      playerId: this.playerId,
      board,
      options,
      explorationRate: this.explorationRate,
    };

    const response = await fetch(`${this.baseUrl}${choiceUrl}`, {
      method: "post",
      headers: jsonHeaders,
      body: JSON.stringify(requestPayload),
    });

    let body = await response.json();
    return body.choice;
  }
}

module.exports = WebPlayer;
