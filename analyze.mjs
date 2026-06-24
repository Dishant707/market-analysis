const KLINE_URL = 'https://api.binance.com/api/v3/klines';

const PAIRS = ['BTCUSDT', 'PAXGUSDT', 'XAUTUSDT'];

async function fetchKlines(symbol, interval = '1h', limit = 100) {
  const url = `${KLINE_URL}?symbol=${symbol}&interval=${interval}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${symbol}: ${res.status}`);
  const raw = await res.json();
  return raw.map(k => ({
    open: parseFloat(k[1]),
    high: parseFloat(k[2]),
    low: parseFloat(k[3]),
    close: parseFloat(k[4]),
    volume: parseFloat(k[5]),
  }));
}

function sma(data, period) {
  const result = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) { result.push(null); continue; }
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += data[j];
    result.push(sum / period);
  }
  return result;
}

function ema(data, period) {
  const result = [];
  const multiplier = 2 / (period + 1);
  let prev = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) { result.push(null); continue; }
    if (i === period - 1) { result.push(prev); continue; }
    prev = (data[i] - prev) * multiplier + prev;
    result.push(prev);
  }
  return result;
}

function rsi(data, period = 14) {
  const result = [];
  let gains = 0, losses = 0;
  for (let i = 0; i < data.length; i++) {
    if (i < period) { result.push(null); continue; }
    if (i === period) {
      gains = 0; losses = 0;
      for (let j = i - period + 1; j <= i; j++) {
        const diff = data[j] - data[j - 1];
        if (diff > 0) gains += diff; else losses -= diff;
      }
      gains /= period; losses /= period;
    } else {
      const diff = data[i] - data[i - 1];
      gains = ((gains * (period - 1)) + (diff > 0 ? diff : 0)) / period;
      losses = ((losses * (period - 1)) + (diff < 0 ? -diff : 0)) / period;
    }
    if (losses === 0) { result.push(100); continue; }
    const rs = gains / losses;
    result.push(100 - (100 / (1 + rs)));
  }
  return result;
}

function macd(data) {
  const ema12 = ema(data, 12);
  const ema26 = ema(data, 26);
  const macdLine = [];
  const signal = [];
  const histogram = [];
  for (let i = 0; i < data.length; i++) {
    if (ema12[i] === null || ema26[i] === null) {
      macdLine.push(null);
      signal.push(null);
      histogram.push(null);
      continue;
    }
    macdLine.push(ema12[i] - ema26[i]);
  }
  const sigArr = [];
  for (let i = 0; i < macdLine.length; i++) {
    if (macdLine[i] === null) { sigArr.push(null); continue; }
    const valid = macdLine.slice(1, i + 1).filter(v => v !== null);
    if (valid.length < 9) { sigArr.push(ema(macdLine.filter(v => v !== null), 9).pop() ?? null); continue; }
    sigArr.push(ema(valid, 9).pop() ?? null);
  }
  for (let i = 0; i < macdLine.length; i++) {
    if (macdLine[i] === null || sigArr[i] === null) histogram.push(null);
    else histogram.push(macdLine[i] - sigArr[i]);
  }
  return { macdLine, signal: sigArr, histogram };
}

function bollinger(data, period = 20, std = 2) {
  const mid = sma(data, period);
  const upper = [];
  const lower = [];
  for (let i = 0; i < data.length; i++) {
    if (mid[i] === null) { upper.push(null); lower.push(null); continue; }
    let sumSq = 0;
    for (let j = i - period + 1; j <= i; j++) sumSq += (data[j] - mid[i]) ** 2;
    const sd = Math.sqrt(sumSq / period);
    upper.push(mid[i] + std * sd);
    lower.push(mid[i] - std * sd);
  }
  return { upper, mid, lower };
}

function atr(high, low, close, period = 14) {
  const tr = [];
  for (let i = 0; i < close.length; i++) {
    if (i === 0) { tr.push(high[i] - low[i]); continue; }
    const hl = high[i] - low[i];
    const hc = Math.abs(high[i] - close[i - 1]);
    const lc = Math.abs(low[i] - close[i - 1]);
    tr.push(Math.max(hl, hc, lc));
  }
  const result = [];
  let prev = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = 0; i < tr.length; i++) {
    if (i < period - 1) { result.push(null); continue; }
    if (i === period - 1) { result.push(prev); continue; }
    prev = ((prev * (period - 1)) + tr[i]) / period;
    result.push(prev);
  }
  return result;
}

function emojiForRsi(rsiVal) {
  if (rsiVal === null) return '';
  if (rsiVal >= 70) return ' \x1b[31m⚠ OVERBOUGHT\x1b[0m';
  if (rsiVal <= 30) return ' \x1b[32m✅ OVERSOLD\x1b[0m';
  return '';
}

function emojiForBB(close, upper, lower) {
  if (close === null || upper === null || lower === null) return '';
  if (close >= upper) return ' \x1b[31m▲ TOUCHING UPPER\x1b[0m';
  if (close <= lower) return ' \x1b[32m▼ TOUCHING LOWER\x1b[0m';
  return '';
}

function emojiForMACD(hist) {
  if (hist === null) return '';
  if (hist > 0) return ' \x1b[32m↑ BULLISH\x1b[0m';
  return ' \x1b[31m↓ BEARISH\x1b[0m';
}

function formatChange(pct) {
  if (pct > 0) return `\x1b[32m+${pct.toFixed(2)}%\x1b[0m`;
  return `\x1b[31m${pct.toFixed(2)}%\x1b[0m`;
}

async function analyze(symbol, interval = '1h', limit = 100) {
  const klines = await fetchKlines(symbol, interval, limit);
  const closes = klines.map(k => k.close);
  const highs = klines.map(k => k.high);
  const lows = klines.map(k => k.low);

  const last = closes[closes.length - 1];
  const prev = closes[closes.length - 2];
  const change24 = ((last - klines[0].close) / klines[0].close) * 100;
  const change1 = ((last - prev) / prev) * 100;

  const rsiVals = rsi(closes, 14);
  const bb = bollinger(closes, 20, 2);
  const macdVals = macd(closes);
  const atrVals = atr(highs, lows, closes, 14);
  const ema20 = ema(closes, 20);
  const ema50 = ema(closes, 50);

  const lastRsi = rsiVals[rsiVals.length - 1];
  const lastAtr = atrVals[atrVals.length - 1];
  const lastMacd = macdVals.macdLine[macdVals.macdLine.length - 1];
  const lastSignal = macdVals.signal[macdVals.signal.length - 1];
  const lastHist = macdVals.histogram[macdVals.histogram.length - 1];
  const lastBBu = bb.upper[bb.upper.length - 1];
  const lastBBl = bb.lower[bb.lower.length - 1];
  const lastBBm = bb.mid[bb.mid.length - 1];
  const lastEma20 = ema20[ema20.length - 1];
  const lastEma50 = ema50[ema50.length - 1];

  const label = symbol.replace('USDT', '/USDT');

  console.log(`\n  ┌─ ${label}  (1h · ${klines.length} candles)`);
  console.log(`  ├─ Price: $${last.toFixed(2)}`);
  console.log(`  ├─ 1h Change: ${formatChange(change1)}`);
  console.log(`  ├─ 24h Change: ${formatChange(change24)}`);
  console.log(`  ├─────────────────────────────────────`);
  console.log(`  ├─ RSI(14): ${lastRsi?.toFixed(2)}${emojiForRsi(lastRsi)}`);
  console.log(`  ├─ EMA(20): ${lastEma20?.toFixed(2)}  ${last >= lastEma20 ? '\x1b[32m▲ Price Above\x1b[0m' : '\x1b[31m▼ Price Below\x1b[0m'}`);
  console.log(`  ├─ EMA(50): ${lastEma50?.toFixed(2)}  ${last >= lastEma50 ? '\x1b[32m▲ Price Above\x1b[0m' : '\x1b[31m▼ Price Below\x1b[0m'}`);
  console.log(`  ├─ BB Upper: $${lastBBu?.toFixed(2)}${emojiForBB(last, lastBBu, lastBBl)}`);
  console.log(`  ├─ BB Mid:   $${lastBBm?.toFixed(2)}`);
  console.log(`  ├─ BB Lower: $${lastBBl?.toFixed(2)}`);
  console.log(`  ├─ MACD: ${lastMacd?.toFixed(2)}  Signal: ${lastSignal?.toFixed(2)}  Hist: ${lastHist?.toFixed(2)}${emojiForMACD(lastHist)}`);
  console.log(`  ├─ ATR(14): $${lastAtr?.toFixed(2)}  (volatility)`);
  console.log(`  └─ ${getSignal(lastRsi, last, lastEma20, lastEma50, lastHist, lastBBu, lastBBl)}`);
  console.log('');
}

function getSignal(rsiVal, price, ema20, ema50, hist, bbu, bbl) {
  let bullish = 0, bearish = 0;

  if (rsiVal !== null && rsiVal < 30) bullish += 2;
  if (rsiVal !== null && rsiVal > 70) bearish += 2;
  if (rsiVal !== null && rsiVal > 50) bullish += 1; else bearish += 1;
  if (price >= ema20) bullish += 1; else bearish += 1;
  if (price >= ema50) bullish += 1; else bearish += 1;
  if (hist !== null && hist > 0) bullish += 2; else if (hist !== null) bearish += 2;
  if (price >= bbu) bearish += 1;
  if (price <= bbl) bullish += 1;

  let signal;
  if (bullish >= bearish + 3) signal = '\x1b[32mSTRONG BUY 🟢\x1b[0m';
  else if (bullish > bearish) signal = '\x1b[32mBUY 🟢\x1b[0m';
  else if (bearish >= bullish + 3) signal = '\x1b[31mSTRONG SELL 🔴\x1b[0m';
  else if (bearish > bullish) signal = '\x1b[31mSELL 🔴\x1b[0m';
  else signal = '\x1b[33mNEUTRAL ⚠️\x1b[0m';

  return `Signal: ${signal}`;
}

async function main() {
  console.clear();
  console.log(`\x1b[1m  ════════════════════════════════════`);
  console.log(`   CRYPTO MARKET ANALYSIS (1h)          `);
  console.log(`   ${new Date().toLocaleString()}         `);
  console.log(`  ════════════════════════════════════\x1b[0m`);
  for (const pair of PAIRS) {
    try {
      await analyze(pair);
    } catch (e) {
      console.error(`  ${pair}: ${e.message}`);
    }
  }
}

main();
