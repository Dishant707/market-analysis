// ─── Multi-source Data Fetcher ───
// Binance for crypto · Yahoo Finance for TradFi

const YAHOO = 'https://query1.finance.yahoo.com/v8/finance/chart';

// Asset universe
export const ASSETS = {
  // Crypto
  BTC:   { source: 'binance', symbol: 'BTCUSDT',  type: 'crypto' },
  ETH:   { source: 'binance', symbol: 'ETHUSDT',  type: 'crypto' },
  SOL:   { source: 'binance', symbol: 'SOLUSDT',  type: 'crypto' },
  XAUT:  { source: 'binance', symbol: 'XAUTUSDT', type: 'crypto' },

  // Indices
  SP500:   { source: 'yahoo', symbol: '^GSPC',   type: 'index' },
  NASDAQ:  { source: 'yahoo', symbol: '^IXIC',   type: 'index' },
  DOW:     { source: 'yahoo', symbol: '^DJI',    type: 'index' },
  VIX:     { source: 'yahoo', symbol: '^VIX',    type: 'index' },

  // Commodities
  GOLD:    { source: 'yahoo', symbol: 'GC=F',    type: 'commodity' },
  SILVER:  { source: 'yahoo', symbol: 'SI=F',    type: 'commodity' },
  OIL:     { source: 'yahoo', symbol: 'CL=F',    type: 'commodity' },

  // Forex
  DXY:     { source: 'yahoo', symbol: 'DX-Y.NYB', type: 'forex' },
  EURUSD:  { source: 'yahoo', symbol: 'EURUSD=X', type: 'forex' },
  USDJPY:  { source: 'yahoo', symbol: 'USDJPY=X', type: 'forex' },
};

export function round(n, d = 2) {
  return Math.round(n * 10 ** d) / 10 ** d;
}

async function fetchBinance(symbol, interval = '1d', limit = 500) {
  const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Binance ${symbol}: ${res.status}`);
  const data = await res.json();
  return data.map(k => ({
    t: k[0], o: +k[1], h: +k[2], l: +k[3], c: +k[4], v: +k[5]
  }));
}

async function fetchYahoo(symbol, interval = '1d', range = '6mo') {
  const url = `${YAHOO}/${encodeURIComponent(symbol)}?interval=${interval}&range=${range}`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  if (!res.ok) throw new Error(`Yahoo ${symbol}: ${res.status}`);
  const data = await res.json();
  const r = data.chart.result[0];
  const q = r.indicators.quote[0];

  const timestamps = r.timestamp || [];
  const closes = q.close || [];
  const volumes = q.volume || [];

  const candles = [];
  for (let i = 0; i < timestamps.length; i++) {
    const c = closes[i];
    if (c === null || c === undefined) continue;
    candles.push({
      t: timestamps[i] * 1000,
      o: c,
      h: c,
      l: c,
      c,
      v: (volumes[i] || 0),
    });
  }
  return candles;
}

// Fetch historical data for one asset
export async function fetchAsset(name, interval = '1d', limit = 500) {
  const cfg = ASSETS[name];
  if (!cfg) throw new Error(`Unknown asset: ${name}`);
  const raw = cfg.source === 'binance'
    ? await fetchBinance(cfg.symbol, interval, limit)
    : await fetchYahoo(cfg.symbol, interval);
  return { name, ...cfg, candles: raw };
}

// Fetch multiple assets in parallel, align by date
export async function fetchAligned(assets, interval = '1d', limit = 500) {
  const results = await Promise.allSettled(
    assets.map(a => fetchAsset(a, interval, limit))
  );

  const valid = [];
  for (const r of results) {
    if (r.status === 'fulfilled') valid.push(r.value);
  }

  if (valid.length < 2) throw new Error('Need at least 2 assets with data');

  // Align by timestamp — find common dates
  const dateSets = valid.map(a => new Set(a.candles.map(c => dateKey(c.t))));
  const common = [...dateSets[0]].filter(d => dateSets.every(s => s.has(d))).sort();

  const aligned = valid.map(a => {
    const map = new Map(a.candles.map(c => [dateKey(c.t), c.c]));
    return {
      name: a.name,
      type: a.type,
      symbol: a.symbol,
      prices: common.map(d => map.get(d)),
      timestamps: common.map(d => {
        const c = a.candles.find(c => dateKey(c.t) === d);
        return c ? c.t : null;
      }),
    };
  });

  return { assets: aligned, commonDates: common, count: common.length };
}

function dateKey(ts) {
  return new Date(ts).toISOString().slice(0, 10);
}
