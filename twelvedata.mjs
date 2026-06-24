// ──────────────────────────────────────────────────
//  TWELVEDATA.MJS — Unified Market Data Fetcher
//  Primary source for all assets (crypto, gold, FX)
//  Fallback to Binance if rate limited
// ──────────────────────────────────────────────────

const TD = 'https://api.twelvedata.com';
const API_KEY = process.env.TWELVEDATA_API_KEY || 'bb21d50e60f749579ad37fb98e5bedb3';
const BINANCE = 'https://api.binance.com/api/v3';

// ─── Asset Mapping ─────────────────────────────
const ASSETS = {
  'BTC':    { td: 'BTC/USD',     bn: 'BTCUSDT',   name: 'Bitcoin' },
  'XAUT':   { td: 'XAU/USD',     bn: 'XAUTUSDT',  name: 'Gold (XAUT)' },
  'ETH':    { td: 'ETH/USD',     bn: 'ETHUSDT',   name: 'Ethereum' },
  'GOLD':   { td: 'XAU/USD',     bn: null,        name: 'Gold Spot' },
  'SILVER': { td: 'XAG/USD',     bn: null,        name: 'Silver' },
  'CRUDE':  { td: 'CL',          bn: null,        name: 'Crude Oil' },
  'DXY':    { td: 'DXY',         bn: null,        name: 'Dollar Index' },
  'SPX':    { td: 'SPX',         bn: null,        name: 'S&P 500' },
};

// ─── Fetch Klines from Twelve Data ─────────────
export async function fetchKlines(symbol, interval = '1h', limit = 200) {
  const cfg = ASSETS[symbol.replace('USDT', '')];
  const tdSymbol = cfg?.td || symbol;
  const intervalMap = { '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1day' };
  const tdInterval = intervalMap[interval] || '1h';

  // Try Twelve Data first
  try {
    const url = `${TD}/time_series?symbol=${tdSymbol}&interval=${tdInterval}&outputsize=${limit}&apikey=${API_KEY}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`TD ${res.status}`);
    const data = await res.json();

    if (data.status === 'error') throw new Error(data.message);

    const values = data.values || [];
    // Twelve Data returns newest first, Binance expects oldest first
    return values.reverse().map(v => ({
      t: new Date(v.datetime).getTime(),
      o: parseFloat(v.open),
      h: parseFloat(v.high),
      l: parseFloat(v.low),
      c: parseFloat(v.close),
      v: parseFloat(v.volume || 0),
    }));
  } catch (e) {
    // Fallback to Binance for crypto pairs
    if (cfg?.bn) {
      try {
        const bnUrl = `${BINANCE}/klines?symbol=${cfg.bn}&interval=${interval}&limit=${limit}`;
        const res = await fetch(bnUrl);
        if (!res.ok) throw new Error(`Binance ${res.status}`);
        const data = await res.json();
        return data.map(k => ({
          t: k[0], o: +k[1], h: +k[2], l: +k[3], c: +k[4], v: +k[5],
        }));
      } catch (_) {
        throw new Error(`TD failed: ${e.message} | Binance fallback also failed`);
      }
    }
    throw e;
  }
}

// ─── Fetch Live Ticker ──────────────────────────
export async function fetchTicker(symbol) {
  const cfg = ASSETS[symbol.replace('USDT', '')];
  const tdSymbol = cfg?.td || symbol;

  try {
    const url = `${TD}/quote?symbol=${tdSymbol}&apikey=${API_KEY}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`TD ${res.status}`);
    const data = await res.json();
    if (data.status === 'error') throw new Error(data.message);

    return {
      price: parseFloat(data.close || data.price || 0),
      chg: parseFloat(data.change || 0),
      chgPct: parseFloat(data.percent_change || 0),
      high: parseFloat(data.high || 0),
      low: parseFloat(data.low || 0),
      vol: parseFloat(data.volume || 0),
      open: parseFloat(data.open || 0),
    };
  } catch (e) {
    // Fallback to Binance ticker
    if (cfg?.bn) {
      try {
        const bnUrl = `${BINANCE}/ticker/24hr?symbol=${cfg.bn}`;
        const res = await fetch(bnUrl);
        if (!res.ok) throw new Error(`Binance ${res.status}`);
        const d = await res.json();
        return {
          price: parseFloat(d.lastPrice),
          chg: parseFloat(d.priceChange),
          chgPct: parseFloat(d.priceChangePercent),
          high: parseFloat(d.highPrice),
          low: parseFloat(d.lowPrice),
          vol: parseFloat(d.quoteVolume),
        };
      } catch (_) {
        throw new Error(`TD failed: ${e.message}`);
      }
    }
    throw e;
  }
}

// ─── Fetch Multiple Tickers at Once ────────────
export async function fetchAllTickers(symbols = ['BTC', 'XAUT', 'GOLD', 'DXY']) {
  const results = {};
  for (const s of symbols) {
    try {
      results[s] = await fetchTicker(s);
    } catch (e) {
      results[s] = { error: e.message };
    }
  }
  return results;
}

function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

export const ASSET_LIST = Object.keys(ASSETS);
