// ──────────────────────────────────────────────────
//  COMMODITIES.MJS — Real-time Commodity Data
//  Gold, Silver, Oil, Copper, Natural Gas
//  Yahoo Finance (free, no auth) → Telegram
// ──────────────────────────────────────────────────

const YAHOO = 'https://query1.finance.yahoo.com/v8/finance/chart';

const COMMODITIES = {
  GOLD:    { symbol: 'GC=F',   name: 'Gold',     unit: '$/oz',    key: true },
  SILVER:  { symbol: 'SI=F',   name: 'Silver',   unit: '$/oz',    key: false },
  CRUDE:   { symbol: 'CL=F',   name: 'Crude Oil', unit: '$/bbl',  key: true },
  COPPER:  { symbol: 'HG=F',   name: 'Copper',   unit: '$/lb',    key: false },
  NGAS:    { symbol: 'NG=F',   name: 'Nat Gas',  unit: '$/mmbtu', key: false },
  PLATINUM:{ symbol: 'PL=F',   name: 'Platinum', unit: '$/oz',    key: false },
  DXY:     { symbol: 'DX-Y.NYB', name: 'Dollar Index', unit: 'pts', key: true },
};

async function fetchYahoo(symbol, interval = '1d', range = '3mo') {
  const url = `${YAHOO}/${encodeURIComponent(symbol)}?interval=${interval}&range=${range}`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  if (!res.ok) throw new Error(`Yahoo ${symbol}: ${res.status}`);
  const data = await res.json();
  const result = data.chart?.result?.[0];
  if (!result) throw new Error(`No data for ${symbol}`);
  const q = result.indicators?.quote?.[0];
  if (!q) throw new Error(`No quotes for ${symbol}`);
  const timestamps = result.timestamp || [];
  const closes = q.close || [];
  const volumes = q.volume || [];
  const opens = q.open || [];
  const highs = q.high || [];
  const lows = q.low || [];

  const candles = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (closes[i] === null) continue;
    candles.push({
      t: timestamps[i] * 1000,
      o: +opens[i] || +closes[i],
      h: +highs[i] || +closes[i],
      l: +lows[i] || +closes[i],
      c: +closes[i],
      v: +(volumes[i] || 0),
    });
  }
  return candles;
}

function calcRSI(data, period = 14) {
  const r = []; let avgG = 0, avgL = 0;
  for (let i = 0; i < data.length; i++) {
    if (i < period) { r.push(null); continue; }
    if (i === period) {
      let g = 0, l = 0;
      for (let j = i - period + 1; j <= i; j++) {
        const d = data[j] - data[j - 1];
        if (d > 0) g += d; else l -= d;
      }
      avgG = g / period; avgL = l / period;
    } else {
      const d = data[i] - data[i - 1];
      avgG = ((avgG * (period - 1)) + (d > 0 ? d : 0)) / period;
      avgL = ((avgL * (period - 1)) + (d < 0 ? -d : 0)) / period;
    }
    if (avgL === 0) { r.push(100); continue; }
    r.push(100 - 100 / (1 + avgG / avgL));
  }
  return r;
}

export async function analyzeCommodity(key) {
  const cfg = COMMODITIES[key];
  if (!cfg) return { error: `Unknown commodity: ${key}` };

  try {
    const candles = await fetchYahoo(cfg.symbol, '1d', '3mo');
    if (candles.length < 20) return { error: `Insufficient data: ${key}` };

    const closes = candles.map(k => k.c);
    const price = closes[closes.length - 1];
    const prev = closes[closes.length - 2];
    const weekAgo = closes[Math.max(0, closes.length - 8)];
    const monthAgo = closes[Math.max(0, closes.length - 31)];

    const rsi = calcRSI(closes, 14);
    const rsiVal = rsi[rsi.length - 1];
    const rsiSignal = rsiVal > 70 ? 'OB' : rsiVal < 30 ? 'OS' :
                      rsiVal > 55 ? 'BULL' : rsiVal < 45 ? 'BEAR' : 'NEUT';

    const high3m = Math.max(...closes);
    const low3m = Math.min(...closes);
    const rangePos = (price - low3m) / (high3m - low3m || 1) * 100;

    const returns = [];
    for (let i = 1; i < closes.length; i++) returns.push((closes[i] - closes[i - 1]) / closes[i - 1] * 100);
    const volatility = Math.sqrt(returns.reduce((s, r) => s + (r - returns.reduce((a, b) => a + b, 0) / returns.length) ** 2, 0) / returns.length);

    const chg24h = round((price - prev) / prev * 100, 2);
    const chg7d = round((price - weekAgo) / weekAgo * 100, 2);
    const chg30d = monthAgo ? round((price - monthAgo) / monthAgo * 100, 2) : null;

    let trend = 'NEUTRAL';
    if (chg7d > 3) trend = 'UPTREND';
    else if (chg7d < -3) trend = 'DOWNTREND';

    return {
      key,
      name: cfg.name,
      symbol: cfg.symbol,
      unit: cfg.unit,
      price: round(price, 2),
      chg24h,
      chg7d,
      chg30d,
      rsi: round(rsiVal, 1),
      rsiSignal,
      volatility: round(volatility, 2),
      trend,
      rangePos: round(rangePos, 1),
      high3m: round(high3m, 2),
      low3m: round(low3m, 2),
    };
  } catch (e) {
    return { error: e.message, key };
  }
}

export async function analyzeAllCommodities() {
  const keys = Object.keys(COMMODITIES).filter(k => COMMODITIES[k].key);
  const results = [];
  for (const k of keys) {
    const r = await analyzeCommodity(k);
    results.push(r);
  }
  return results;
}

export function formatCommodityAlert(results) {
  let msg = `\n⛽ COMMODITIES | ${new Date().toLocaleTimeString()}\n`;
  msg += `───────────────────────────\n`;

  for (const r of results) {
    if (r.error) continue;
    const arrow = r.trend === 'UPTREND' ? '🟢' : r.trend === 'DOWNTREND' ? '🔴' : '⚪';
    msg += `${arrow} ${r.name.padEnd(12)} $${r.price} | ${r.chg24h > 0 ? '+' : ''}${r.chg24h}% | RSI:${r.rsi} ${r.rsiSignal}\n`;
  }

  msg += `───────────────────────────`;
  return msg;
}

function round(n, d = 2) {
  return Math.round(n * 10 ** d) / 10 ** d;
}

// CLI
async function main() {
  const results = await analyzeAllCommodities();
  console.log(JSON.stringify(results, null, 2));
  console.log('\n' + formatCommodityAlert(results));
}

const isMain = process.argv[1]?.includes('commodities.mjs');
if (isMain) main();
