import { WebSocketServer, WebSocket } from 'ws';

const BINANCE = 'wss://stream.binance.com:9443/ws';
const PAIRS = ['btcusdt', 'paxgusdt', 'xautusdt'];

export function initLiveStream(server) {
  const wss = new WebSocketServer({ server, path: '/ws' });
  const clients = new Set();
  const latestPrices = {};

  function broadcast(msg) {
    const payload = JSON.stringify(msg);
    for (const client of clients) {
      if (client.readyState === WebSocket.OPEN) client.send(payload);
    }
  }

  function connectFeed(pair, streams, handler) {
    const url = `${BINANCE}/${streams.map(s => `${pair}@${s}`).join('/')}`;
    const ws = new WebSocket(url);
    ws.on('message', raw => {
      try {
        const data = JSON.parse(raw.toString());
        handler(data);
      } catch (_) {}
    });
    ws.on('close', () => setTimeout(() => connectFeed(pair, streams, handler), 3000));
    ws.on('error', () => {});
    return ws;
  }

  // Trade + ticker for each pair
  for (const pair of PAIRS) {
    connectFeed(pair, ['trade', 'ticker'], data => {
      let msg = null;
      if (data.e === 'trade') {
        msg = { type: 'trade', pair: data.s, price: parseFloat(data.p), qty: parseFloat(data.q), time: data.T, side: data.m ? 'sell' : 'buy' };
      } else if (data.e === '24hrTicker') {
        msg = { type: 'ticker', pair: data.s, price: parseFloat(data.c), change: parseFloat(data.P), high: parseFloat(data.h), low: parseFloat(data.l), vol: parseFloat(data.q), time: data.E };
      }
      if (msg) { latestPrices[data.s] = msg; broadcast(msg); }
    });

    // Depth (order book) for each pair
    connectFeed(pair, ['depth20@100ms'], data => {
      if (data.e === 'depthUpdate') {
        broadcast({ type: 'depth', pair: data.s, bids: data.b, asks: data.a });
      }
    });
  }

  wss.on('connection', (ws) => {
    clients.add(ws);
    Object.values(latestPrices).forEach(p => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(p)); });
    ws.on('close', () => clients.delete(ws));
    ws.on('error', () => clients.delete(ws));
  });

  return wss;
}
