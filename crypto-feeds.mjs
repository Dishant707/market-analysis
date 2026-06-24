const BASE = 'https://api.binance.com/api/v3/ticker/price';

const PAIRS = ['BTCUSDT', 'PAXGUSDT', 'XAUTUSDT'];

async function fetchPrice(symbol) {
  const res = await fetch(`${BASE}?symbol=${symbol}`);
  if (!res.ok) throw new Error(`${symbol}: ${res.status}`);
  const data = await res.json();
  return { symbol, price: parseFloat(data.price) };
}

function format(symbol, price) {
  const label = symbol.replace('USDT', '/USDT');
  const padded = price.toFixed(2);
  return `${label}: $${padded}`;
}

async function poll() {
  const results = await Promise.allSettled(PAIRS.map(fetchPrice));
  const lines = [];
  for (const r of results) {
    if (r.status === 'fulfilled') {
      lines.push(format(r.value.symbol, r.value.price));
    } else {
      lines.push(`[error] ${r.reason.message}`);
    }
  }
  console.clear();
  console.log(`\n  Live Crypto Feeds  (${new Date().toLocaleTimeString()})\n`);
  console.log(`  ${lines.join('\n  ')}\n`);
}

console.log('Starting live feeds (Ctrl+C to stop)...');
poll();
setInterval(poll, 5000);
