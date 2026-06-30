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

class ServerConnectionError extends Error {
  constructor(url, cause) {
    super(`Cannot connect to model server at ${url} — is the server running? (${cause.message})`);
    this.name = "ServerConnectionError";
    this.url = url;
    this.cause = cause;
  }
}

class RequestQueue {
  constructor() {
    this._queue = [];
    this._running = false;
  }

  enqueue(fn) {
    return new Promise((resolve, reject) => {
      this._queue.push({ fn, resolve, reject });
      this._drain();
    });
  }

  _drain() {
    if (this._running || this._queue.length === 0) return;
    this._running = true;
    const { fn, resolve, reject } = this._queue.shift();
    fn()
      .then(resolve, reject)
      .finally(() => {
        this._running = false;
        this._drain();
      });
  }
}

const sharedQueue = new RequestQueue();

function getProfile(args, playerId) {
  let key = `player${playerId}Profile`;
  return args && args.profiles ? args.profiles[args[key]] : null;
}

class WebPlayer extends Player {
  constructor(args, playerId) {
    super();
    let profile = getProfile(args, playerId);

    if (!profile)
      throw new Error(`Unable to find profile for player[${playerId}]`);

    this.playerId = playerId;
    this.baseUrl = profile.baseUrl;
    this.explorationRate = args.explorationRate;
  }

  async register() {
    try {
      const response = await fetch(`${this.baseUrl}${pingUrl}`);
      let body = await response.json();
      return body.alive;
    } catch (err) {
      throw new ServerConnectionError(this.baseUrl, err);
    }
  }

  async choose(board, options) {
    const requestPayload = {
      playerId: this.playerId,
      board,
      options,
      explorationRate: this.explorationRate,
    };

    return sharedQueue.enqueue(async () => {
      let response;
      try {
        response = await fetch(`${this.baseUrl}${choiceUrl}`, {
          method: "post",
          headers: jsonHeaders,
          body: JSON.stringify(requestPayload),
        });
      } catch (err) {
        throw new ServerConnectionError(this.baseUrl, err);
      }
      const body = await response.json();
      return body.choice;
    });
  }
}

module.exports = WebPlayer;
module.exports.ServerConnectionError = ServerConnectionError;
