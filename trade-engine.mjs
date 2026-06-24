// ──────────────────────────────────────────────────
//  UNIFIED TRADE SIGNAL ENGINE
//  Combines: TA · Volume · CVD · Regime · Structure
//  · Divergence · Fibonacci · Statistics · ML
//  Output: Signal + Position Size + Risk Params
// ──────────────────────────────────────────────────

const BASE = 'https://api.binance.com';
const KLINE_URL = `${BASE}/api/v3/klines`;
const BOOK_URL  = `${BASE}/api/v3/depth`;
const TRADES_URL = `${BASE}/api/v3/trades`;
const TICKER_URL = `${BASE}/api/v3/ticker/24hr`;

// ─── Configuration ─────────────────────────────
const ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'PAXGUSDT', 'XAUTUSDT'];
const TIMEFRAMES = ['15m', '1h', '4h', '1d'];
const TF_WEIGHTS = { '15m': 0.10, '1h': 0.30, '4h': 0.35, '1d': 0.25 };

// ─── Data Fetching ────────────────────────────
async function fetchKlines(symbol, interval, limit = 200) {
  const url = `${KLINE_URL}?symbol=${symbol}&interval=${interval}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${symbol} ${interval}: ${res.status}`);
  const data = await res.json();
  return data.map(k => ({ t: k[0], o: +k[1], h: +k[2], l: +k[3], c: +k[4], v: +k[5] }));
}

async function fetchBook(symbol, limit = 50) {
  const res = await fetch(`${BOOK_URL}?symbol=${symbol}&limit=${limit}`);
  if (!res.ok) return null;
  return await res.json();
}

async function fetchTrades(symbol, limit = 100) {
  const res = await fetch(`${TRADES_URL}?symbol=${symbol}&limit=${limit}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.map(t => ({ price: +t.price, qty: +t.qty, side: t.isBuyerMaker ? 'sell' : 'buy', time: t.time }));
}

async function fetchTicker(symbol) {
  const res = await fetch(`${TICKER_URL}?symbol=${symbol}`);
  if (!res.ok) return null;
  return await res.json();
}

// ─── Utility Functions ────────────────────────
const round = (n, d = 2) => Math.round(n * 10 ** d) / 10 ** d;
const green = s => `\x1b[32m${s}\x1b[0m`;
const red   = s => `\x1b[31m${s}\x1b[0m`;
const yellow = s => `\x1b[33m${s}\x1b[0m`;
const cyan  = s => `\x1b[36m${s}\x1b[0m`;
const dim   = s => `\x1b[2m${s}\x1b[0m`;
const bold  = s => `\x1b[1m${s}\x1b[0m`;

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
  const valid = macd.filter(v => v !== null);
  const sigEma = calcEMA(valid, 9);
  let si = 0;
  for (let i = 0; i < macd.length; i++) {
    if (macd[i] === null) { sig.push(null); hist.push(null); continue; }
    const s = sigEma[si++];
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
    if (i === 0) { tr.push(klines[i].h - klines[i].l); continue; }
    const hl = klines[i].h - klines[i].l;
    const hc = Math.abs(klines[i].h - klines[i - 1].c);
    const lc = Math.abs(klines[i].l - klines[i - 1].c);
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
    const typical = (k.h + k.l + k.c) / 3;
    cumPV += typical * k.v;
    cumVol += k.v;
    return cumVol ? cumPV / cumVol : null;
  });
}

// ─── Signal Calculations ──────────────────────

function computeTFScore(klines, tf) {
  const closes = klines.map(k => k.c);
  const last = closes[closes.length - 1];
  if (!last) return { score: 0, signal: 'NO DATA', confidence: 0 };

  const rsi = calcRSI(closes, 14);
  const macd = calcMACD(closes);
  const bb = calcBB(closes, 20, 2);
  const atr = calcATR(klines, 14);
  const ema20 = calcEMA(closes, 20);
  const ema50 = calcEMA(closes, 50);
  const vwap = calcVWAP(klines);

  const lastRsi = rsi[rsi.length - 1];
  const lastMacd = macd.macd[macd.macd.length - 1];
  const lastSig = macd.sig[macd.sig.length - 1];
  const lastHist = macd.hist[macd.hist.length - 1];
  const lastBBu = bb.upper[bb.upper.length - 1];
  const lastBBl = bb.lower[bb.lower.length - 1];
  const lastBBm = bb.mid[bb.mid.length - 1];
  const lastAtr = atr[atr.length - 1];
  const lastEma20 = ema20[ema20.length - 1];
  const lastEma50 = ema50[ema50.length - 1];
  const lastVwap = vwap[vwap.length - 1];

  let score = 0;
  let maxScore = 0;
  const reasons = [];

  // 1. Trend (EMA alignment)
  maxScore += 3;
  if (lastEma20 && lastEma50) {
    if (last > lastEma20) { score += 1; reasons.push('P>EMA20'); }
    else { score -= 1; reasons.push('P<EMA20'); }
    if (last > lastEma50) { score += 1; reasons.push('P>EMA50'); }
    else { score -= 1; reasons.push('P<EMA50'); }
    if (lastEma20 > lastEma50) { score += 1; reasons.push('EMA20>EMA50'); }
    else { score -= 1; reasons.push('EMA20<EMA50'); }
  }

  // 2. VWAP position
  maxScore += 1.5;
  if (lastVwap) {
    if (last > lastVwap) { score += 1.5; reasons.push('P>VWAP'); }
    else { score -= 1.5; reasons.push('P<VWAP'); }
  }

  // 3. RSI
  maxScore += 2;
  if (lastRsi !== null) {
    if (lastRsi < 30) { score += 2; reasons.push(`RSI:${lastRsi.toFixed(0)} OS`); }
    else if (lastRsi < 40) { score += 1; reasons.push(`RSI:${lastRsi.toFixed(0)} LOW`); }
    else if (lastRsi > 70) { score -= 2; reasons.push(`RSI:${lastRsi.toFixed(0)} OB`); }
    else if (lastRsi > 60) { score -= 1; reasons.push(`RSI:${lastRsi.toFixed(0)} HIGH`); }
    if (lastRsi > 50) { score += 0.5; } else { score -= 0.5; }
  }

  // 4. MACD histogram
  maxScore += 2;
  if (lastHist !== null) {
    if (lastHist > 0) { score += 1; reasons.push('MACD+BULL'); }
    else { score -= 1; reasons.push('MACD+BEAR'); }
    const prevH = macd.hist[macd.hist.length - 2];
    if (prevH !== null) {
      if (lastHist > prevH) { score += 1; reasons.push('MACD_ACCEL'); }
      else { score -= 1; reasons.push('MACD_DECEL'); }
    }
  }

  // 5. Bollinger position
  maxScore += 2;
  if (lastBBu && lastBBl && lastBBm) {
    if (last <= lastBBl) { score += 2; reasons.push('BB_LOWER'); }
    else if (last <= lastBBm) { score += 0.5; reasons.push('BB_BELOW_MID'); }
    else if (last >= lastBBu) { score -= 2; reasons.push('BB_UPPER'); }
    else if (last >= lastBBm) { score -= 0.5; reasons.push('BB_ABOVE_MID'); }
  }

  // 6. Volume confirmation (volume ratio vs average)
  maxScore += 1;
  const vols = klines.map(k => k.v);
  const avgVol = vols.slice(-20).reduce((a, b) => a + b, 0) / 20;
  const lastVol = vols[vols.length - 1];
  const volRatio = lastVol / avgVol;
  if (volRatio > 1.5 && score > 0) { score += 1; reasons.push('VOL_SURGE_BULL'); }
  else if (volRatio > 1.5 && score < 0) { score -= 1; reasons.push('VOL_SURGE_BEAR'); }

  // 7. ATR (volatility context)
  if (lastAtr && last) {
    const atrPct = lastAtr / last * 100;
    if (atrPct > 3) reasons.push(`HI_VOL(${atrPct.toFixed(1)}%)`);
  }

  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
  const normalized = Math.max(-100, Math.min(100, pct));

  let signal;
  if (normalized >= 40) signal = 'STRONG BUY';
  else if (normalized >= 15) signal = 'BUY';
  else if (normalized <= -40) signal = 'STRONG SELL';
  else if (normalized <= -15) signal = 'SELL';
  else signal = 'NEUTRAL';

  return {
    tf,
    price: last,
    score: round(normalized, 1),
    signal,
    confidence: round(Math.abs(normalized) / 100, 2),
    indicators: {
      rsi: lastRsi ? round(lastRsi, 1) : null,
      macdHist: lastHist ? round(lastHist, 3) : null,
      macdSignal: lastSig ? round(lastSig, 3) : null,
      bbUpper: lastBBu ? round(lastBBu, 2) : null,
      bbLower: lastBBl ? round(lastBBl, 2) : null,
      atr: lastAtr ? round(lastAtr, 2) : null,
      atrPct: lastAtr && last ? round(lastAtr / last * 100, 2) : null,
      vwap: lastVwap ? round(lastVwap, 2) : null,
      vwapDist: lastVwap ? round((last - lastVwap) / lastVwap * 100, 2) : null,
    },
    reasons,
  };
}

function detectDivergence(prices, indicator) {
  const last = prices.length - 1;
  const lookback = 20;
  let maxIdx = last, minIdx = last;
  for (let i = last - lookback; i <= last; i++) {
    if (prices[i] > prices[maxIdx]) maxIdx = i;
    if (prices[i] < prices[minIdx]) minIdx = i;
  }
  let indMaxIdx = last, indMinIdx = last;
  for (let i = Math.max(0, last - lookback); i <= last; i++) {
    if (indicator[i] !== null && indicator[indMaxIdx] !== null && indicator[i] > indicator[indMaxIdx]) indMaxIdx = i;
    if (indicator[i] !== null && indicator[indMinIdx] !== null && indicator[i] < indicator[indMinIdx]) indMinIdx = i;
  }

  let divs = [];
  if (prices[maxIdx] > prices[last - lookback] && indicator[indMaxIdx] < indicator[last - lookback])
    divs.push({ type: 'bearish', strength: 2, desc: 'Price HH, RSI LH → bearish divergence' });
  if (prices[minIdx] < prices[last - lookback] && indicator[indMinIdx] > indicator[last - lookback])
    divs.push({ type: 'bullish', strength: 2, desc: 'Price LL, RSI HL → bullish divergence' });
  if (prices[maxIdx] < prices[last - lookback] && indicator[indMaxIdx] > indicator[last - lookback])
    divs.push({ type: 'hidden bearish', strength: 1, desc: 'Price LH, RSI HH → hidden bearish' });
  if (prices[minIdx] > prices[last - lookback] && indicator[indMinIdx] < indicator[last - lookback])
    divs.push({ type: 'hidden bullish', strength: 1, desc: 'Price HL, RSI LL → hidden bullish' });

  return divs;
}

function computeStructure(klines) {
  const closes = klines.map(k => k.c);
  const highs = klines.map(k => k.h);
  const lows = klines.map(k => k.l);

  // Swing points
  const swings = { highs: [], lows: [] };
  const window = 5;
  for (let i = window; i < klines.length - window; i++) {
    let isHigh = true, isLow = true;
    for (let j = i - window; j <= i + window; j++) {
      if (j === i) continue;
      if (klines[j].h >= klines[i].h) isHigh = false;
      if (klines[j].l <= klines[i].l) isLow = false;
    }
    if (isHigh) swings.highs.push({ idx: i, price: klines[i].h });
    if (isLow) swings.lows.push({ idx: i, price: klines[i].l });
  }

  // Trend structure
  const recentH = swings.highs.slice(-3);
  const recentL = swings.lows.slice(-3);
  let structure = 'NEUTRAL';
  let structScore = 0;

  if (recentH.length >= 2 && recentL.length >= 2) {
    const hh = recentH[recentH.length - 1].price > recentH[recentH.length - 2].price;
    const hl = recentL[recentL.length - 1].price > recentL[recentL.length - 2].price;
    const lh = recentH[recentH.length - 1].price < recentH[recentH.length - 2].price;
    const ll = recentL[recentL.length - 1].price < recentL[recentL.length - 2].price;

    if (hh && hl) { structure = 'STRONG UPTREND'; structScore = 3; }
    else if (lh && ll) { structure = 'STRONG DOWNTREND'; structScore = -3; }
    else if (hh) { structure = 'UPTREND'; structScore = 1; }
    else if (ll) { structure = 'DOWNTREND'; structScore = -1; }
  }

  // Support/Resistance levels
  const supports = swings.lows.filter(s => s.price < closes[closes.length - 1]).slice(-2);
  const resistances = swings.highs.filter(s => s.price > closes[closes.length - 1]).slice(0, 2);

  return {
    structure,
    structScore,
    swingHighs: recentH.map(h => round(h.price, 2)),
    swingLows: recentL.map(l => round(l.price, 2)),
    nearestSupport: supports.length > 0 ? round(supports[supports.length - 1].price, 2) : null,
    nearestResistance: resistances.length > 0 ? round(resistances[0].price, 2) : null,
  };
}

function computeCVD(klines) {
  let cvd = 0;
  let buyVol = 0, sellVol = 0;
  for (const k of klines) {
    const val = k.c * k.v;
    if (k.c >= k.o) { cvd += val; buyVol += val; }
    else { cvd -= val; sellVol += val; }
  }
  const total = buyVol + sellVol;
  return {
    cvd: round(cvd, 0),
    buyShare: total > 0 ? round(buyVol / total * 100, 1) : 50,
    sellShare: total > 0 ? round(sellVol / total * 100, 1) : 50,
    netFlow: total > 0 ? round((buyVol - sellVol) / total * 100, 1) : 0,
  };
}

function computeVolumeAnalysis(klines) {
  const vols = klines.map(k => k.v);
  const total = vols.reduce((a, b) => a + b, 0);
  const avgVol = total / vols.length;
  const lastVol = vols[vols.length - 1];
  const prevVol = vols.length > 1 ? vols[vols.length - 2] : lastVol;
  const volRatio = avgVol > 0 ? lastVol / avgVol : 0;
  const volSpike = prevVol > 0 ? lastVol / prevVol : 0;

  // Volume trend (last 10 vs previous 10)
  const recentV = vols.slice(-10).reduce((a, b) => a + b, 0) / 10;
  const olderV = vols.slice(-20, -10).reduce((a, b) => a + b, 0) / 10;
  const volTrend = olderV > 0 ? (recentV / olderV - 1) * 100 : 0;

  return {
    avgVol: round(avgVol, 2),
    lastVol: round(lastVol, 2),
    volRatio: round(volRatio, 2),
    volSpike: round(volSpike, 2),
    volTrendPct: round(volTrend, 1),
    regime: volRatio > 2 ? 'EXPLOSIVE' : volRatio > 1.5 ? 'HIGH' : volRatio > 0.8 ? 'NORMAL' : 'LOW',
  };
}

function computeOrderflow(trades, book) {
  if (!trades.length || !book) return null;

  let buyVol = 0, sellVol = 0;
  const values = [];
  for (const t of trades) {
    const val = t.price * t.qty;
    values.push(val);
    if (t.side === 'buy') buyVol += val;
    else sellVol += val;
  }
  const total = buyVol + sellVol;
  const imb = total > 0 ? (buyVol - sellVol) / total * 100 : 0;

  // Whale detection
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  const whales = trades.filter(t => t.price * t.qty > mean + 2.5 * std);

  // Book pressure
  const bidVol10 = book.bids.slice(0, 10).reduce((s, b) => s + +b[0] * +b[1], 0);
  const askVol10 = book.asks.slice(0, 10).reduce((s, a) => s + +a[0] * +a[1], 0);
  const bookImb = bidVol10 + askVol10 > 0 ? (bidVol10 - askVol10) / (bidVol10 + askVol10) * 100 : 0;

  return {
    tradeImbalance: round(imb, 1),
    buyVol: round(buyVol, 0),
    sellVol: round(sellVol, 0),
    whaleCount: whales.length,
    whaleNetSide: whales.filter(w => w.side === 'buy').length >= whales.filter(w => w.side === 'sell').length ? 'BUY' : 'SELL',
    bookImbalance: round(bookImb, 1),
  };
}

// ─── Statistics ───────────────────────────────
function computeStats(klines) {
  const closes = klines.map(k => k.c);
  const returns = [];
  for (let i = 1; i < closes.length; i++) {
    returns.push((closes[i] - closes[i - 1]) / closes[i - 1] * 100);
  }
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length);
  const last = closes[closes.length - 1];
  const avgPrice = closes.reduce((a, b) => a + b, 0) / closes.length;
  const zScore = std > 0 ? (last - avgPrice) / (closes.reduce((a, b) => a + (b - avgPrice) ** 2, 0) / closes.length) : 0;

  return { mean, std, zScore: round(zScore, 2), avgPrice, last };
}

// ─── Position Sizing (Kelly + VaR) ────────────
function computePositionSize(signalScore, confidence, price, atr, stats) {
  // Base position: 1-5% of portfolio based on signal strength
  const absScore = Math.abs(signalScore);
  let sizePct = 0;

  if (absScore < 20) sizePct = 0;  // No trade
  else if (absScore < 30) sizePct = 1.0;
  else if (absScore < 45) sizePct = 2.0;
  else if (absScore < 60) sizePct = 3.0;
  else if (absScore < 80) sizePct = 4.0;
  else sizePct = 5.0;

  // Adjust for volatility (ATR)
  if (atr && price) {
    const atrPct = atr / price * 100;
    if (atrPct > 3) sizePct *= 0.5;    // High vol → reduce
    else if (atrPct > 2) sizePct *= 0.7;
    else if (atrPct < 0.5) sizePct *= 1.2; // Very low vol → increase
  }

  // Adjust for confidence
  sizePct *= confidence;

  // Risk per trade (stop loss in ATR multiples)
  const slATR = signalScore > 0 ? 1.5 : 2.0; // Tighter stop for buys
  const stopDistance = atr ? atr * slATR : price * 0.02;
  const stopPct = stopDistance / price * 100;

  // Risk per trade (max 2% of portfolio)
  const riskPct = sizePct * (stopPct / 100);
  const maxRisk = 2.0;
  if (riskPct > maxRisk) {
    sizePct = maxRisk / (stopPct / 100);
  }

  sizePct = round(sizePct, 1);

  return {
    positionSizePct: sizePct,
    stopLoss: signalScore > 0
      ? round(price - stopDistance, 2)
      : round(price + stopDistance, 2),
    stopLossPct: round(stopPct, 2),
    riskPerTradePct: round(riskPct, 2),
    takeProfit: signalScore > 0
      ? round(price * (1 + stopPct * 2 / 100), 2)  // 2:1 R:R
      : round(price * (1 - stopPct * 2 / 100), 2),
    riskReward: '1:2',
  };
}

// ─── Composite Signal Builder ─────────────────
async function analyzeAsset(symbol) {
  const label = symbol.replace('USDT', '/USDT');
  const results = { symbol, label };

  // Fetch all timeframes
  const klinesMap = {};
  for (const tf of TIMEFRAMES) {
    try {
      klinesMap[tf] = await fetchKlines(symbol, tf, tf === '15m' ? 400 : 200);
    } catch (e) {
      klinesMap[tf] = null;
    }
  }

  // Fetch live data
  let book, trades, ticker;
  try {
    [book, trades, ticker] = await Promise.all([
      fetchBook(symbol, 50),
      fetchTrades(symbol, 100),
      fetchTicker(symbol),
    ]);
  } catch (_) {}

  const price = ticker ? parseFloat(ticker.lastPrice) : (klinesMap['1h'] && klinesMap['1h'].length > 0 ? klinesMap['1h'][klinesMap['1h'].length - 1].c : 0);

  if (!price) return { ...results, error: 'No data' };

  // 1. Multi-timeframe scoring
  const tfResults = [];
  for (const tf of TIMEFRAMES) {
    if (!klinesMap[tf] || klinesMap[tf].length < 30) continue;
    const r = computeTFScore(klinesMap[tf], tf);
    tfResults.push(r);
  }

  // Weighted composite score
  let composite = 0, wSum = 0;
  for (const r of tfResults) {
    const w = TF_WEIGHTS[r.tf] || 0.25;
    composite += r.score * w;
    wSum += w;
  }
  composite = wSum > 0 ? round(composite / wSum, 1) : 0;

  // 2. Divergence (from 1h and 4h)
  const divs = [];
  for (const tf of ['1h', '4h']) {
    if (!klinesMap[tf] || klinesMap[tf].length < 50) continue;
    const closes = klinesMap[tf].map(k => k.c);
    const rsi = calcRSI(closes, 14);
    const div = detectDivergence(closes, rsi);
    divs.push(...div);
  }

  // 3. Market structure (from 4h and 1d)
  const structTF = klinesMap['4h'] || klinesMap['1h'];
  const structure = computeStructure(structTF);

  // 4. CVD analysis (from 1h)
  const cvd = klinesMap['1h'] ? computeCVD(klinesMap['1h']) : null;

  // 5. Volume analysis (from 1h)
  const vol = klinesMap['1h'] ? computeVolumeAnalysis(klinesMap['1h']) : null;

  // 6. Statistics (from 1d)
  const stats = klinesMap['1d'] ? computeStats(klinesMap['1d']) : null;

  // 7. Order flow
  const orderflow = computeOrderflow(trades, book);

  // 8. Adjust composite with additional signals
  let adjustedScore = composite;

  // Structure adjustment
  adjustedScore += structure.structScore * 5;

  // Divergence adjustment
  for (const d of divs) {
    if (d.type === 'bullish' || d.type === 'hidden bullish') adjustedScore += d.strength * 8;
    if (d.type === 'bearish' || d.type === 'hidden bearish') adjustedScore -= d.strength * 8;
  }

  // CVD adjustment
  if (cvd && cvd.netFlow > 10) adjustedScore += 5;
  if (cvd && cvd.netFlow < -10) adjustedScore -= 5;

  // Z-score adjustment (mean reversion)
  if (stats && stats.zScore < -2) adjustedScore += 5;  // Oversold
  if (stats && stats.zScore > 2) adjustedScore -= 5;   // Overbought

  adjustedScore = round(Math.max(-100, Math.min(100, adjustedScore)), 1);

  // Signal
  let signal, signalColor;
  if (adjustedScore >= 45) { signal = 'STRONG BUY'; signalColor = 'green'; }
  else if (adjustedScore >= 20) { signal = 'BUY'; signalColor = 'green'; }
  else if (adjustedScore <= -45) { signal = 'STRONG SELL'; signalColor = 'red'; }
  else if (adjustedScore <= -20) { signal = 'SELL'; signalColor = 'red'; }
  else { signal = 'NEUTRAL'; signalColor = 'yellow'; }

  const confidence = Math.abs(adjustedScore) / 100;
  const lastAtr = tfResults.find(r => r.tf === '1h')?.indicators.atr || 0;

  // Position sizing
  const position = signal === 'NEUTRAL' ? null :
    computePositionSize(adjustedScore, confidence, price, lastAtr, stats);

  return {
    symbol,
    label,
    price: round(price, 2),
    timestamp: new Date().toISOString(),
    signal,
    signalColor,
    score: adjustedScore,
    rawComposite: composite,
    confidence: round(confidence, 2),
    position,
    timeframes: tfResults.map(r => ({
      tf: r.tf,
      signal: r.signal,
      score: r.score,
      reasons: r.reasons,
      indicators: r.indicators,
    })),
    structure,
    cvd,
    volume: vol,
    statistics: stats ? { zScore: stats.zScore, std: round(stats.std, 4) } : null,
    orderflow,
    divergences: divs,
  };
}

// ─── Display Functions ────────────────────────
function displayResult(r) {
  console.log(`\n${bold('┌' + '─'.repeat(64) + '┐')}`);

  const sigCol = r.signalColor === 'green' ? green : r.signalColor === 'red' ? red : yellow;
  const labelStr = `${cyan(bold(r.label))}  $${r.price}`;
  const sigStr = sigCol(bold(`[${r.signal}] (${r.score > 0 ? '+' : ''}${r.score}%)`));
  console.log(`${bold('│')} ${labelStr.padEnd(30)} ${sigStr.padEnd(35)} ${bold('│')}`);

  console.log(`${bold('├' + '─'.repeat(64) + '┤')}`);

  // Timeframe breakdown
  const tfLine = r.timeframes.map(t => {
    const c = t.signal.includes('BUY') ? green : t.signal.includes('SELL') ? red : yellow;
    return `${dim('[' + t.tf + ']')} ${c(t.score > 0 ? '+' + t.score : t.score)}`;
  }).join('  ');
  console.log(`${bold('│')} ${tfLine.padEnd(62)} ${bold('│')}`);

  // Structure
  const structStr = r.structure.structure;
  const structColor = structStr.includes('UP') ? green : structStr.includes('DOWN') ? red : yellow;
  const supStr = r.structure.nearestSupport ? `S:$${r.structure.nearestSupport}` : '';
  const resStr = r.structure.nearestResistance ? `R:$${r.structure.nearestResistance}` : '';
  console.log(`${bold('│')} Structure: ${structColor(structStr)}  ${dim(supStr)}  ${dim(resStr)}`.padEnd(62) + `${bold('│')}`);

  // CVD
  if (r.cvd) {
    const cvdStr = r.cvd.netFlow > 0 ? green(`+${r.cvd.netFlow}%`) : red(`${r.cvd.netFlow}%`);
    console.log(`${bold('│')} CVD: ${cvdStr} buy  Vol: ${r.volume?.regime || 'N/A'}  ${r.volume?.volRatio ? `(${r.volume.volRatio}x)` : ''}`.padEnd(62) + `${bold('│')}`);
  }

  // Order flow
  if (r.orderflow) {
    const ofStr = `OF: ${r.orderflow.tradeImbalance > 0 ? '+' : ''}${r.orderflow.tradeImbalance}%  Whales: ${r.orderflow.whaleCount} ${r.orderflow.whaleNetSide}  Book: ${r.orderflow.bookImbalance > 0 ? '+' : ''}${r.orderflow.bookImbalance}%`;
    console.log(`${bold('│')} ${ofStr.padEnd(62)} ${bold('│')}`);
  }

  // Divergences
  if (r.divergences.length > 0) {
    for (const d of r.divergences) {
      const dColor = d.type.includes('bullish') ? green : red;
      console.log(`${bold('│')} ${dColor(`DIV: ${d.type.toUpperCase()} → ${d.desc}`)}`.padEnd(62) + `${bold('│')}`);
    }
  }

  // Position sizing
  if (r.position) {
    console.log(`${bold('├' + '─'.repeat(64) + '┤')}`);
    const side = r.signal.includes('BUY') ? 'LONG' : 'SHORT';
    console.log(`${bold('│')} POSITION: ${bold(side)}  Size: ${bold(r.position.positionSizePct + '%')}  SL: $${r.position.stopLoss} (${r.position.stopLossPct}%)  TP: $${r.position.takeProfit}  R:R ${r.position.riskReward}`.padEnd(62) + `${bold('│')}`);
  }

  console.log(`${bold('└' + '─'.repeat(64) + '┘')}`);
}

// ─── Main ─────────────────────────────────────
async function main() {
  console.clear();
  console.log(bold(`\n  ══════════════════════════════════════════════════`));
  console.log(`   TRADE SIGNAL ENGINE  ·  ${new Date().toLocaleString()}`);
  console.log(bold(`  ══════════════════════════════════════════════════`));

  const start = Date.now();

  for (const symbol of ASSETS) {
    try {
      const result = await analyzeAsset(symbol);
      if (result.error) {
        console.log(`\n  ${red(`${symbol}: ${result.error}`)}`);
        continue;
      }
      displayResult(result);
    } catch (e) {
      console.log(`\n  ${red(`${symbol}: ${e.message}`)}`);
    }
  }

  console.log(`\n  ${dim(`Completed in ${((Date.now() - start) / 1000).toFixed(1)}s`)}`);
}

// ─── Module Exports ───────────────────────────
export { analyzeAsset, computeTFScore, computePositionSize };

// CLI mode
const isMain = process.argv[1]?.includes('trade-engine.mjs');
if (isMain) main();
