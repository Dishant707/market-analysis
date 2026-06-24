// ─────────────────────────────────────────────
//  DEEP CRYPTO MARKET ANALYSIS
//  Multi-timeframe · Order Book · Structure
//  Volume · Statistics · Fibonacci · Divergence
// ─────────────────────────────────────────────

const BASE = 'https://api.binance.com';
const KLINE_URL = `${BASE}/api/v3/klines`;
const BOOK_URL  = `${BASE}/api/v3/depth`;
const TICKER_URL = `${BASE}/api/v3/ticker/24hr`;

const PAIRS = ['BTCUSDT', 'PAXGUSDT', 'XAUTUSDT'];
const TIMEFRAMES = ['15m', '1h', '4h', '1d'];
const TF_NAMES = { '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D' };

// ─── Helpers ──────────────────────────────────

function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

function green(s) { return `\x1b[32m${s}\x1b[0m`; }
function red(s)   { return `\x1b[31m${s}\x1b[0m`; }
function yellow(s){ return `\x1b[33m${s}\x1b[0m`; }
function cyan(s)  { return `\x1b[36m${s}\x1b[0m`; }
function dim(s)   { return `\x1b[2m${s}\x1b[0m`; }
function bold(s)  { return `\x1b[1m${s}\x1b[0m`; }

// ─── Data Fetching ────────────────────────────

async function fetchKlines(symbol, interval, limit = 150) {
  const url = `${KLINE_URL}?symbol=${symbol}&interval=${interval}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${symbol} ${interval}: ${res.status}`);
  const raw = await res.json();
  return raw.map(k => ({
    t: k[0], open: +k[1], high: +k[2], low: +k[3], close: +k[4], vol: +k[5]
  }));
}

async function fetchOrderBook(symbol, limit = 100) {
  const url = `${BOOK_URL}?symbol=${symbol}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`orderbook ${symbol}: ${res.status}`);
  return await res.json();
}

async function fetchTicker24h(symbol) {
  const url = `${TICKER_URL}?symbol=${symbol}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ticker ${symbol}: ${res.status}`);
  return await res.json();
}

// ─── Technical Indicators ─────────────────────

function calcSMA(data, period) {
  const r = []; let sum = 0;
  for (let i = 0; i < data.length; i++) {
    sum += data[i];
    if (i >= period - 1) { r.push(sum / period); sum -= data[i - period + 1]; }
    else r.push(null);
  }
  return r;
}

function calcEMA(data, period) {
  const r = []; const mult = 2 / (period + 1);
  let prev = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) { r.push(null); continue; }
    if (i === period - 1) { r.push(prev); continue; }
    prev = (data[i] - prev) * mult + prev;
    r.push(prev);
  }
  return r;
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

function calcMACD(data) {
  const e12 = calcEMA(data, 12);
  const e26 = calcEMA(data, 26);
  const macd = []; const sig = []; const hist = [];
  for (let i = 0; i < data.length; i++) {
    if (e12[i] === null || e26[i] === null) { macd.push(null); sig.push(null); hist.push(null); continue; }
    macd.push(e12[i] - e26[i]);
  }
  const validMacd = macd.filter(v => v !== null);
  const sigEma = calcEMA(validMacd, 9);
  let sigIdx = 0;
  for (let i = 0; i < macd.length; i++) {
    if (macd[i] === null) { sig.push(null); hist.push(null); continue; }
    const s = sigEma[sigIdx++];
    sig.push(s);
    hist.push(macd[i] - s);
  }
  return { macd, sig, hist };
}

function calcBB(data, period = 20, mult = 2) {
  const mid = calcSMA(data, period);
  const upper = []; const lower = [];
  for (let i = 0; i < data.length; i++) {
    if (mid[i] === null) { upper.push(null); lower.push(null); continue; }
    let sq = 0;
    for (let j = i - period + 1; j <= i; j++) sq += (data[j] - mid[i]) ** 2;
    const sd = Math.sqrt(sq / period);
    upper.push(mid[i] + mult * sd);
    lower.push(mid[i] - mult * sd);
  }
  return { upper, mid, lower };
}

function calcATR(klines, period = 14) {
  const tr = []; const r = [];
  for (let i = 0; i < klines.length; i++) {
    if (i === 0) { tr.push(klines[i].high - klines[i].low); continue; }
    const hl = klines[i].high - klines[i].low;
    const hc = Math.abs(klines[i].high - klines[i - 1].close);
    const lc = Math.abs(klines[i].low - klines[i - 1].close);
    tr.push(Math.max(hl, hc, lc));
  }
  let prev = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = 0; i < tr.length; i++) {
    if (i < period - 1) { r.push(null); continue; }
    if (i === period - 1) { r.push(prev); continue; }
    prev = ((prev * (period - 1)) + tr[i]) / period;
    r.push(prev);
  }
  return r;
}

function calcVWAP(klines) {
  let cumPV = 0, cumVol = 0;
  return klines.map(k => {
    const typical = (k.high + k.low + k.close) / 3;
    cumPV += typical * k.vol;
    cumVol += k.vol;
    return cumVol ? cumPV / cumVol : null;
  });
}

// ─── Order Book Analysis ──────────────────────

function analyzeOrderBook(book) {
  const bids = book.bids.map(b => [+b[0], +b[1]]);
  const asks = book.asks.map(a => [+a[0], +a[1]]);

  const bestBid = bids[0][0];
  const bestAsk = asks[0][0];
  const spread = bestAsk - bestBid;
  const spreadPct = (spread / bestAsk) * 100;

  const bidVol = bids.reduce((s, b) => s + b[0] * b[1], 0);
  const askVol = asks.reduce((s, a) => s + a[0] * a[1], 0);
  const imbalance = bidVol > 0 && askVol > 0 ? ((bidVol - askVol) / (bidVol + askVol)) * 100 : 0;

  // detect walls (concentration of volume)
  const totalBidQty = bids.reduce((s, b) => s + b[1], 0);
  const totalAskQty = asks.reduce((s, a) => s + a[1], 0);
  const wallThreshold = 0.15;

  const bidWall = bids.find(b => b[1] >= totalBidQty * wallThreshold);
  const askWall = asks.find(a => a[1] >= totalAskQty * wallThreshold);

  return { bestBid, bestAsk, spread, spreadPct, imbalance, bidVol, askVol, bidWall, askWall, totalBidQty, totalAskQty };
}

// ─── Market Structure ─────────────────────────

function findSwingPoints(klines, window = 5) {
  const swings = { highs: [], lows: [] };
  for (let i = window; i < klines.length - window; i++) {
    let isHigh = true, isLow = true;
    for (let j = i - window; j <= i + window; j++) {
      if (j === i) continue;
      if (klines[j].high >= klines[i].high) isHigh = false;
      if (klines[j].low <= klines[i].low) isLow = false;
    }
    if (isHigh) swings.highs.push({ idx: i, price: klines[i].high, time: klines[i].t });
    if (isLow) swings.lows.push({ idx: i, price: klines[i].low, time: klines[i].t });
  }
  return swings;
}

function analyzeTrend(klines) {
  const closes = klines.map(k => k.close);
  const len = closes.length;
  const ema20 = calcEMA(closes, 20);
  const ema50 = calcEMA(closes, 50);
  const last = closes[len - 1];

  const above20 = last >= ema20[ema20.length - 1];
  const above50 = last >= ema50[ema50.length - 1];
  const emaAlign = ema20[ema20.length - 1] > ema50[ema50.length - 1];

  const swings = findSwingPoints(klines);
  const recentHighs = swings.highs.slice(-3);
  const recentLows = swings.lows.slice(-3);

  let trend = 'neutral';
  let strength = 0;

  if (above20 && above50 && emaAlign) { trend = 'uptrend'; strength = 2; }
  else if (!above20 && !above50 && !emaAlign) { trend = 'downtrend'; strength = 2; }
  else if (above20 && !above50) { trend = 'early uptrend'; strength = 1; }
  else if (!above20 && above50) { trend = 'early downtrend'; strength = 1; }

  const hh = recentHighs.length >= 2 && recentHighs[recentHighs.length - 1].price > recentHighs[recentHighs.length - 2].price;
  const hl = recentLows.length >= 2 && recentLows[recentLows.length - 1].price > recentLows[recentLows.length - 2].price;
  const lh = recentHighs.length >= 2 && recentHighs[recentHighs.length - 1].price < recentHighs[recentHighs.length - 2].price;
  const ll = recentLows.length >= 2 && recentLows[recentLows.length - 1].price < recentLows[recentLows.length - 2].price;

  if (hh && hl) { trend = 'strong uptrend'; strength = 3; }
  else if (lh && ll) { trend = 'strong downtrend'; strength = 3; }

  return { trend, strength, above20, above50, emaAlign, recentHighs, recentLows };
}

// ─── Volume Analysis ──────────────────────────

function analyzeVolume(klines) {
  const vols = klines.map(k => k.vol);
  const avgVol = vols.reduce((a, b) => a + b, 0) / vols.length;
  const lastVol = vols[vols.length - 1];
  const volRatio = lastVol / avgVol;
  const prevVol = vols[vols.length - 2];
  const volSpike = prevVol > 0 ? lastVol / prevVol : 0;

  const close = klines.map(k => k.close);
  const vwap = calcVWAP(klines);
  const lastVwap = vwap[vwap.length - 1];
  const vwapDist = lastVwap ? ((close[close.length - 1] - lastVwap) / lastVwap) * 100 : null;

  return { avgVol, lastVol, volRatio, volSpike, vwap: lastVwap, vwapDist };
}

// ─── Statistics ───────────────────────────────

function calcStats(klines) {
  const closes = klines.map(k => k.close);
  const returns = [];
  for (let i = 1; i < closes.length; i++) {
    returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
  }
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
  const stdDev = Math.sqrt(variance);

  const last = closes[closes.length - 1];
  const avgPrice = closes.reduce((a, b) => a + b, 0) / closes.length;
  const priceStd = Math.sqrt(closes.reduce((a, b) => a + (b - avgPrice) ** 2, 0) / closes.length);
  const zScore = priceStd ? (last - avgPrice) / priceStd : 0;

  const annualFactor = { '15m': 35040, '1h': 8760, '4h': 2190, '1d': 365 };
  const high = Math.max(...closes);
  const low = Math.min(...closes);
  const range = high - low;

  return { mean, stdDev, zScore, high, low, range, last, avgPrice, annualFactor };
}

// ─── Fibonacci Levels ─────────────────────────

function calcFibonacci(high, low) {
  const diff = high - low;
  return {
    '0.0%': high,
    '23.6%': high - diff * 0.236,
    '38.2%': high - diff * 0.382,
    '50.0%': high - diff * 0.5,
    '61.8%': high - diff * 0.618,
    '78.6%': high - diff * 0.786,
    '100%': low,
    '127.2%': low - diff * 0.272,
    '161.8%': low - diff * 0.618,
  };
}

// ─── Divergence Detection ─────────────────────

function detectDivergence(price, indicator, window = 14) {
  const last = price.length - 1;
  const lookback = Math.min(window, last);

  let maxIdx = last, minIdx = last;
  for (let i = last - lookback; i <= last; i++) {
    if (price[i] > price[maxIdx]) maxIdx = i;
    if (price[i] < price[minIdx]) minIdx = i;
  }

  let indMaxIdx = last, indMinIdx = last;
  for (let i = Math.max(0, last - lookback); i <= last; i++) {
    if (indicator[i] !== null && (indicator[i] > indicator[indMaxIdx] || indicator[i] === null)) {
      if (indicator[i] !== null) indMaxIdx = i;
    }
    if (indicator[i] !== null && (indicator[i] < indicator[indMinIdx] || indicator[i] === null)) {
      if (indicator[i] !== null) indMinIdx = i;
    }
  }

  let type = 'none';
  let strength = 0;

  // Bearish divergence: price makes higher high, RSI makes lower high
  if (price[maxIdx] > price[last - lookback] && indicator[indMaxIdx] < indicator[Math.max(0, last - lookback)]) {
    type = 'bearish'; strength = 2;
  }
  // Bullish divergence: price makes lower low, RSI makes higher low
  if (price[minIdx] < price[last - lookback] && indicator[indMinIdx] > indicator[Math.max(0, last - lookback)]) {
    type = 'bullish'; strength = 2;
  }
  // Hidden bearish: price makes lower high, RSI makes higher high
  if (price[maxIdx] < price[last - lookback] && indicator[indMaxIdx] > indicator[Math.max(0, last - lookback)]) {
    type = 'hidden bearish'; strength = 1;
  }
  // Hidden bullish: price makes higher low, RSI makes lower low
  if (price[minIdx] > price[last - lookback] && indicator[indMinIdx] < indicator[Math.max(0, last - lookback)]) {
    type = 'hidden bullish'; strength = 1;
  }

  return { type, strength };
}

// ─── Single Timeframe Analysis ────────────────

function analyzeTF(klines, tf) {
  const closes = klines.map(k => k.close);
  const last = closes[closes.length - 1];
  const prev = closes[closes.length - 2];

  const rsi = calcRSI(closes, 14);
  const macd = calcMACD(closes);
  const bb = calcBB(closes, 20, 2);
  const atr = calcATR(klines, 14);
  const ema20 = calcEMA(closes, 20);
  const ema50 = calcEMA(closes, 50);

  const lastRsi = rsi[rsi.length - 1];
  const lastMacd = macd.macd[macd.macd.length - 1];
  const lastSig = macd.sig[macd.sig.length - 1];
  const lastHist = macd.hist[macd.hist.length - 1];
  const lastAtr = atr[atr.length - 1];
  const lastBBu = bb.upper[bb.upper.length - 1];
  const lastBBl = bb.lower[bb.lower.length - 1];
  const lastBBm = bb.mid[bb.mid.length - 1];
  const lastEma20 = ema20[ema20.length - 1];
  const lastEma50 = ema50[ema50.length - 1];

  const vol = analyzeVolume(klines);
  const trend = analyzeTrend(klines);
  const stats = calcStats(klines);

  // divergence
  const rsiDiv = detectDivergence(closes, rsi, 28);
  const macdDiv = detectDivergence(closes, macd.macd, 28);

  // scoring
  let score = 0; let maxScore = 0;

  // RSI
  maxScore += 2;
  if (lastRsi !== null) {
    if (lastRsi < 30) score += 2;
    else if (lastRsi < 40) score += 1;
    else if (lastRsi > 70) score -= 2;
    else if (lastRsi > 60) score -= 1;
    if (lastRsi > 50) score += 0.5; else score -= 0.5;
  }

  // Trend
  maxScore += 3;
  score += trend.strength * (trend.trend.includes('up') ? 1 : trend.trend.includes('down') ? -1 : 0);

  // EMAs
  maxScore += 2;
  if (trend.above20) score += 1; else score -= 1;
  if (trend.above50) score += 1; else score -= 1;

  // MACD hist
  maxScore += 2;
  if (lastHist !== null) {
    if (lastHist > 0) score += 1; else score -= 1;
    const prevH = macd.hist[macd.hist.length - 2];
    if (prevH !== null && lastHist > prevH) score += 1;
    else if (prevH !== null) score -= 1;
  }

  // BB position
  maxScore += 2;
  if (last <= lastBBl) score += 2;
  else if (last <= lastBBm) score += 0.5;
  else if (last >= lastBBu) score -= 2;
  else if (last >= lastBBm) score -= 0.5;

  // Divergence
  maxScore += 2;
  if (rsiDiv.type === 'bullish') score += 1.5;
  else if (rsiDiv.type === 'bearish') score -= 1.5;
  if (macdDiv.type === 'bullish') score += 0.5;
  else if (macdDiv.type === 'bearish') score -= 0.5;

  // Volume
  maxScore += 1;
  if (vol.volRatio > 1.5) score += 1;
  else if (vol.volRatio < 0.5) score -= 0.5;

  // z-score (mean reversion)
  maxScore += 1;
  if (stats.zScore < -2) score += 1;
  else if (stats.zScore > 2) score -= 1;

  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
  const normalized = Math.max(-100, Math.min(100, pct));

  let signal;
  if (normalized >= 40) signal = 'STRONG BUY';
  else if (normalized >= 15) signal = 'BUY';
  else if (normalized <= -40) signal = 'STRONG SELL';
  else if (normalized <= -15) signal = 'SELL';
  else signal = 'NEUTRAL';

  return {
    tf, last, prev, lastRsi, lastMacd, lastSig, lastHist,
    lastAtr, lastBBu, lastBBl, lastBBm, lastEma20, lastEma50,
    vol, trend, stats, rsiDiv, macdDiv,
    score: normalized, signal
  };
}

// ─── Display ──────────────────────────────────

function displayTF(r) {
  const tfLabel = bold(`[${TF_NAMES[r.tf]}]`);
  const chg = ((r.last - r.prev) / r.prev) * 100;
  const chgStr = chg >= 0 ? green(`+${round(chg, 2)}%`) : red(`${round(chg, 2)}%`);

  const rsiStr = r.lastRsi !== null ? round(r.lastRsi, 1) : '--';
  const rsiColor = r.lastRsi >= 70 ? red : r.lastRsi <= 30 ? green : (s) => s;

  const histVal = r.lastHist !== null ? round(r.lastHist, 2) : '--';
  const histStr = r.lastHist > 0 ? green(`+${histVal}`) : r.lastHist < 0 ? red(`${histVal}`) : `${histVal}`;

  const macdCross = r.lastMacd !== null && r.lastSig !== null
    ? (r.lastMacd > r.lastSig ? green('↑') : red('↓')) : '?';

  const bbp = r.last !== null && r.lastBBu !== null && r.lastBBl !== null
    ? round(((r.last - r.lastBBl) / (r.lastBBu - r.lastBBl)) * 100, 0) : '--';

  const divStr = r.rsiDiv.type !== 'none'
    ? ` ${yellow(`[RSI ${r.rsiDiv.type} div]`)}` : '';
  const macdDivStr = r.macdDiv.type !== 'none'
    ? ` ${yellow(`[MACD ${r.macdDiv.type} div]`)}` : '';

  const volStr = r.vol.volRatio > 1.5 ? green(`${round(r.vol.volRatio, 1)}x`) :
    r.vol.volRatio > 1.2 ? yellow(`${round(r.vol.volRatio, 1)}x`) :
    dim(`${round(r.vol.volRatio, 1)}x`);

  const trendIcon = r.trend.trend.includes('uptrend') ? green('↗') :
    r.trend.trend.includes('downtrend') ? red('↘') : yellow('→');

  const sigColor = r.signal === 'STRONG BUY' ? green :
    r.signal === 'BUY' ? green :
    r.signal === 'STRONG SELL' ? red :
    r.signal === 'SELL' ? red : yellow;

  console.log(`  ${tfLabel} $${round(r.last, 2)}  ${chgStr}  ${trendIcon}  RSI: ${rsiColor(rsiStr)}  MACD: ${histStr}${macdCross}  BB: ${bbp}%  Vol: ${volStr}${divStr}${macdDivStr}`);
  console.log(`        ${dim(`EMA20: ${round(r.lastEma20, 2)}  EMA50: ${round(r.lastEma50, 2)}  ATR: ${round(r.lastAtr, 2)}  Signal: ${sigColor(bold(r.signal))} (${round(r.score, 0)}%)`)}`);
}

function displayOrderBook(pair, book) {
  const o = analyzeOrderBook(book);
  const imbStr = o.imbalance > 5 ? green(`+${round(o.imbalance, 1)}%`) :
    o.imbalance < -5 ? red(`${round(o.imbalance, 1)}%`) : `${round(o.imbalance, 1)}%`;

  let wallStr = '';
  if (o.bidWall) wallStr += ` ${green(`Bid wall $${round(o.bidWall[0], 2)} (${round((o.bidWall[1] / o.totalBidQty) * 100, 1)}%)`)}`;
  if (o.askWall) wallStr += ` ${red(`Ask wall $${round(o.askWall[0], 2)} (${round((o.askWall[1] / o.totalAskQty) * 100, 1)}%)`)}`;

  console.log(`  ${bold('Order Book')}  Spread: $${round(o.spread, 2)} (${round(o.spreadPct, 4)}%)  Imbalance: ${imbStr}${wallStr}`);
  console.log(`        ${dim(`Best Bid: $${round(o.bestBid, 2)}  Best Ask: $${round(o.bestAsk, 2)}  Bid Vol: ${round(o.bidVol, 0)}  Ask Vol: ${round(o.askVol, 0)}`)}`);
}

function displayStats(r) {
  const stats = r.stats;
  const zStr = stats.zScore > 2 ? red(`+${round(stats.zScore, 2)}σ`) :
    stats.zScore < -2 ? green(`${round(stats.zScore, 2)}σ`) : `${round(stats.zScore, 2)}σ`;
  const rangePct = stats.avgPrice > 0 ? (stats.range / stats.avgPrice) * 100 : 0;

  const volRegime = stats.stdDev > 0.02 ? red('HIGH') :
    stats.stdDev < 0.005 ? green('LOW') : yellow('NORMAL');

  console.log(`  ${bold('Statistics')}   Z-Score: ${zStr}  Range: $${round(stats.low, 2)}–$${round(stats.high, 2)} (${round(rangePct, 1)}%)  Vol: ${volRegime}`);
}

function displayFibonacci(r) {
  const stats = r.stats;
  const fib = calcFibonacci(stats.high, stats.low);
  const last = r.last;
  const lines = Object.entries(fib)
      .filter(([k]) => !k.startsWith('161'))
    .map(([k, v]) => {
      const dist = round(((last - v) / v) * 100, 2);
      const marker = Math.abs(dist) < 0.5 ? green(' ◄ PRICE') :
        Math.abs(dist) < 1.5 ? yellow(' ◄ near') : '';
      const pctStr = ` (${dist >= 0 ? '+' : ''}${round(dist, 2)}%)`;
      const colored = dist >= 0 ? pctStr : dim(pctStr);
      return `    ${k.padStart(5)}  $${String(round(v, 2)).padStart(12)}${colored}${marker}`;
    }).join('\n');
  console.log(`  ${bold('Fibonacci')}\n${lines}`);
}

// ─── Composite Score ──────────────────────────

function compositeSignal(results) {
  const weights = { '15m': 0.15, '1h': 0.35, '4h': 0.30, '1d': 0.20 };
  let total = 0, wSum = 0;
  for (const r of results) {
    total += r.score * weights[r.tf];
    wSum += weights[r.tf];
  }
  const composite = total / wSum;

  let signal, color;
  if (composite >= 40) { signal = 'STRONG BUY'; color = green; }
  else if (composite >= 15) { signal = 'BUY'; color = green; }
  else if (composite <= -40) { signal = 'STRONG SELL'; color = red; }
  else if (composite <= -15) { signal = 'SELL'; color = red; }
  else { signal = 'NEUTRAL'; color = yellow; }

  return { composite: round(composite, 1), signal, color };
}

// ─── Main ─────────────────────────────────────

async function deepDive(pair) {
  const [analysis] = await Promise.all([
    (async () => {
      const tfResults = [];
      const klinesMap = {};
      for (const tf of TIMEFRAMES) {
        const klines = await fetchKlines(pair, tf);
        klinesMap[tf] = klines;
        tfResults.push(analyzeTF(klines, tf));
      }
      return { tfResults, klinesMap };
    })(),
  ]);

  const { tfResults, klinesMap } = analysis;

  const label = pair.replace('USDT', '/USDT');
  const lastPrice = tfResults.find(r => r.tf === '1h').last;
  const ticker = await fetchTicker24h(pair);
  const high24 = +ticker.highPrice;
  const low24 = +ticker.lowPrice;
  const vol24 = +ticker.quoteVolume;

  console.log(`\n${bold(`╔══════════════════════════════════════════════════════════════════╗`)}`);
  console.log(`${bold(`║`)}  ${cyan(bold(label))}   $${round(lastPrice, 2)}  24h: $${round(low24, 2)}–$${round(high24, 2)}  Vol: $${round(vol24 / 1e6, 1)}M  ${bold(`║`)}`);
  console.log(`${bold(`╚══════════════════════════════════════════════════════════════════╝`)}`);

  for (const r of tfResults) {
    displayTF(r);
  }

  try {
    const book = await fetchOrderBook(pair);
    displayOrderBook(pair, book);
  } catch (e) { console.log(`  ${red('Order book: error')}`); }

  // stats + fib from 1d
  const d1Result = tfResults.find(r => r.tf === '1d');
  if (d1Result) {
    displayStats(d1Result);
    displayFibonacci(d1Result);
  }

  // composite
  const comp = compositeSignal(tfResults);
  console.log(`\n  ${bold('═══ COMPOSITE SIGNAL ═══')}  ${comp.color(bold(`${comp.signal} (${comp.composite}%)`))}  ${bold('═══')}`);
}

async function main() {
  console.clear();
  const start = Date.now();
  for (const pair of PAIRS) {
    try {
      await deepDive(pair);
    } catch (e) {
      console.error(`\n  ${red(`${pair}: ${e.message}`)}`);
    }
  }
  console.log(`\n  ${dim(`Completed in ${((Date.now() - start) / 1000).toFixed(1)}s`)}`);
}

main();
