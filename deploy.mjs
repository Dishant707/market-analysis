// ──────────────────────────────────────────────────
//  DEPLOY.MJS — Fast Deploy + WhatsApp Alerts
//  Starts server + SSH tunnel + signal alerts
// ──────────────────────────────────────────────────

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { WebSocket } from 'ws';
import { fetchKlines, fetchTicker } from './twelvedata.mjs';
import { analyzeOptionSelling, formatDeltaAlert } from './delta.mjs';
import { analyzeAllCommodities, formatCommodityAlert } from './commodities.mjs';
import { computeEdgeLevels, isNearEdgeLevel, formatEdgeAlert } from './edge-levels.mjs';
import { computeUnifiedSignal } from './signal-model.mjs';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
const PORT = process.env.PORT || 3000;

// ─── Load Config ──────────────────────────────
function loadEnv() {
  const cfg = {};
  try {
    const env = fs.existsSync('.env') ? fs.readFileSync('.env', 'utf8') : '';
    for (const line of env.split('\n')) {
      const m = line.trim().match(/^([^#]+?)=(.+)$/);
      if (m) cfg[m[1].trim()] = m[2].trim();
    }
  } catch (_) {}
  return {
    telegramToken: cfg.TELEGRAM_TOKEN || '',
    telegramChatId: cfg.TELEGRAM_CHAT_ID || '',
  };
}

const cfg = loadEnv();

// ─── Telegram Alert ───────────────────────────
async function sendAlert(message, priority = 'normal') {
  if (!cfg.telegramToken || !cfg.telegramChatId) return;
  try {
    const prefix = priority === 'high' ? '🔴 ' : priority === 'medium' ? '🟡 ' : '🟢 ';
    const text = encodeURIComponent(`${prefix}${message}`);
    const url = `https://api.telegram.org/bot${cfg.telegramToken}/sendMessage?chat_id=${cfg.telegramChatId}&text=${text}`;
    await fetch(url);
  } catch (_) {}
}

// ─── Public Tunnel via localhost.run ──────────
async function startTunnel() {
  // Skip SSH tunnel on cloud platforms (Railway, Render) — they provide URLs
  if (process.env.RAILWAY_SERVICE_ID || process.env.RENDER) return null;

  return new Promise((resolve) => {
    const ssh = spawn('ssh', [
      '-o', 'StrictHostKeyChecking=no',
      '-o', 'ServerAliveInterval=60',
      '-R', `80:localhost:${PORT}`,
      'nokey@localhost.run',
    ]);

    let resolved = false;
    const timeout = setTimeout(() => { if (!resolved) { resolved = true; resolve(null); } }, 15000);

    ssh.stdout.on('data', (d) => {
      const out = d.toString();
      const match = out.match(/https?:\/\/[^\s]+\.lhr\.life/);
      if (match && !resolved) {
        resolved = true;
        clearTimeout(timeout);
        resolve(match[0]);
      }
    });

    ssh.stderr.on('data', () => {});
    ssh.on('error', () => { if (!resolved) { resolved = true; clearTimeout(timeout); resolve(null); } });
  });
}

// ─── Persistent Rust Engine ──────────────────
let engine = null, buf = '', pending = [], ready = false;

function startEngine() {
  try {
    if (engine) return;
    if (!fs.existsSync('./rust-engine/target/release/modern_engine')) {
      console.log('  Engine binary not found — running without Rust analysis');
      return;
    }
    engine = spawn('./rust-engine/target/release/modern_engine', [], { cwd: __dirname, stdio: ['pipe', 'pipe', 'pipe'] });
  buf = ''; ready = false;
  engine.stdout.on('data', d => {
    buf += d.toString();
    const lines = buf.split('\n');
    buf = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const p = JSON.parse(line);
        if (p.type === 'ready') { ready = true; console.log('  Engine ready'); continue; }
        const pend = pending.shift();
        if (pend) pend.resolve(p);
      } catch (_) {}
    }
  });
  engine.stderr.on('data', () => {});
  engine.on('close', () => { engine = null; ready = false; for (const p of pending) p.reject(new Error('died')); pending = []; setTimeout(startEngine, 3000); });
  engine.on('error', () => { engine = null; ready = false; });
  } catch (_) { console.log('  Engine start failed — running without Rust analysis'); }
}

function rustCall(action, data) {
  return new Promise((resolve, reject) => {
    if (!engine || !ready) {
      const p = spawn('./rust-engine/target/release/modern_engine', [], { cwd: __dirname, stdio: ['pipe', 'pipe', 'pipe'] });
      let o = '', e = '';
      p.stdout.on('data', d => o += d);
      p.stderr.on('data', d => e += d);
      p.on('close', () => {
        try {
          const lines = o.trim().split('\n').filter(l => l.trim());
          resolve(JSON.parse(lines[lines.length - 1]));
        } catch (_) { reject(new Error(e || 'parse error')); }
      });
      p.stdin.end(JSON.stringify({ action, data }) + '\n');
      return;
    }
    pending.push({ resolve, reject });
    engine.stdin.write(JSON.stringify({ action, data }) + '\n');
  });
}

startEngine();

// ─── Data Fetchers ────────────────────────────
// klines now use Twelve Data (imported above)
// Binance fallbacks for order book + trades (not rate-limited)

async function fetchBook(symbol, limit = 50) {
  const res = await fetch(`https://api.binance.com/api/v3/depth?symbol=${symbol}&limit=${limit}`);
  if (!res.ok) return null;
  return await res.json();
}

async function fetchTrades(symbol, limit = 100) {
  const res = await fetch(`https://api.binance.com/api/v3/trades?symbol=${symbol}&limit=${limit}`);
  if (!res.ok) return [];
  return (await res.json()).map(t => ({ price: +t.price, qty: +t.qty, side: t.isBuyerMaker ? 'sell' : 'buy', time: t.time }));
}

// ─── API Endpoints ────────────────────────────

app.get('/api/analyze/:pair', async (req, res) => {
  try {
    const pair = req.params.pair.toUpperCase();
    const interval = req.query.interval || '1h';
    const limit = parseInt(req.query.limit) || 200;
    const [klines, book, trades] = await Promise.all([
      fetchKlines(pair, interval, limit), fetchBook(pair, 50), fetchTrades(pair, 100),
    ]);
    if (klines.length < 20) return res.status(400).json({ error: 'insufficient data' });
    const price = klines[klines.length - 1].c;
    const closes = klines.map(k => k.c);
    const highs = klines.map(k => k.h);
    const lows = klines.map(k => k.l);
    const volumes = klines.map(k => k.v);
    const start = Date.now();
    const [orderflow, regimes, features, bayesian] = await Promise.all([
      rustCall('orderflow', { trades, book_bids: book.bids, book_asks: book.asks, klines, price }),
      rustCall('regimes', { closes, highs, lows, volumes, num_regimes: 3 }),
      rustCall('features', { closes, highs, lows, volumes }),
      rustCall('bayesian', { closes, price, targets: [], horizon_hours: 24 }),
    ]);
    res.json({ pair, price, interval, candles: klines.length, elapsed: Date.now() - start,
      orderflow: orderflow.orderflow, regimes: regimes.regimes, features: features.features, bayesian: bayesian.bayesian });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/orderflow/:pair', async (req, res) => {
  const pair = req.params.pair.toUpperCase();
  const [klines, book, trades] = await Promise.all([fetchKlines(pair, '1h', 100), fetchBook(pair, 50), fetchTrades(pair, 100)]);
  const price = klines.length > 0 ? klines[klines.length - 1].c : 0;
  const result = await rustCall('orderflow', { trades, book_bids: book.bids, book_asks: book.asks, klines, price });
  res.json({ pair, price, ...result.orderflow });
});

app.get('/api/regimes/:pair', async (req, res) => {
  const pair = req.params.pair.toUpperCase();
  const klines = await fetchKlines(pair, '1h', 200);
  const closes = klines.map(k => k.c), highs = klines.map(k => k.h), lows = klines.map(k => k.l), volumes = klines.map(k => k.v);
  const result = await rustCall('regimes', { closes, highs, lows, volumes, num_regimes: 3 });
  res.json({ pair, price: closes[closes.length - 1], ...result.regimes });
});

app.get('/api/features/:pair', async (req, res) => {
  const pair = req.params.pair.toUpperCase();
  const klines = await fetchKlines(pair, '1h', 200);
  const closes = klines.map(k => k.c), highs = klines.map(k => k.h), lows = klines.map(k => k.l), volumes = klines.map(k => k.v);
  const result = await rustCall('features', { closes, highs, lows, volumes });
  res.json({ pair, price: closes[closes.length - 1], ...result.features });
});

app.get('/api/bayesian/:pair', async (req, res) => {
  const pair = req.params.pair.toUpperCase();
  const klines = await fetchKlines(pair, '1h', 200);
  const closes = klines.map(k => k.c);
  const price = closes[closes.length - 1];
  const targets = req.query.targets ? req.query.targets.split(',').map(Number) : Array.from({ length: 21 }, (_, i) => price * (0.9 + i * 0.01));
  const result = await rustCall('bayesian', { closes, price, targets, horizon_hours: 24 });
  res.json({ pair, price, ...result.bayesian });
});

app.get('/api/options/:pair', async (req, res) => {
  const pair = req.params.pair.toUpperCase();
  const days = parseInt(req.query.days) || 7;
  const premiumPct = parseFloat(req.query.premium) || 2.0;
  const klines = await fetchKlines(pair, '1d', 200);
  const closes = klines.map(k => k.c);
  const price = closes[closes.length - 1];
  const strikes = [];
  for (let i = -15; i <= 15; i++) strikes.push(price * (1 + i / 100));
  const step = price > 10000 ? 1000 : price > 1000 ? 100 : 10;
  const base = Math.round(price / step) * step;
  for (let i = -10; i <= 10; i++) {
    const s = base + i * step;
    if (!strikes.some(x => Math.abs(x - s) / s < 0.005)) strikes.push(s);
  }
  strikes.sort((a, b) => a - b);
  const result = await rustCall('options', { closes, price, strikes, days_to_expiry: days, premium_percent: premiumPct });
  res.json({ pair, price, daysToExpiry: days, ...result.options });
});

app.post('/api/alert', express.json(), async (req, res) => {
  const { message, symbol, signal, price, score } = req.body || {};
  const text = message || `${signal} signal on ${symbol} @ $${price} (${score > 0 ? '+' : ''}${score}%)`;
  await sendAlert(text, signal?.includes('STRONG') ? 'high' : 'medium');
  res.json({ sent: true, text });
});

app.use(express.static(path.join(__dirname, 'public')));

// ─── Helpers ──────────────────────────────────
function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

// ─── Real-Time Monitor ────────────────────────

const MONITORED = ['BTCUSDT', 'XAUTUSDT'];
const BINANCE_WS = 'wss://stream.binance.com:9443/ws';

// State
let livePrices = {};
let lastSignals = {};
let keyLevels = {};
let edgeLevels = {};
let indicators = {};
let alertCooldown = {};
let lastAlertMsg = {}; // dedupe

// ─── WebSocket: Live Prices ───────────────────
function connectLivePrices() {
  const streams = MONITORED.map(s => `${s.toLowerCase()}@ticker`).join('/');
  const ws = new WebSocket(`${BINANCE_WS}/${streams}`);

  ws.on('message', raw => {
    try {
      const d = JSON.parse(raw.toString());
      if (d.e === '24hrTicker') {
        livePrices[d.s] = {
          price: parseFloat(d.c), chg24h: parseFloat(d.P),
          high24h: parseFloat(d.h), low24h: parseFloat(d.l), vol: parseFloat(d.q),
        };
      }
    } catch (_) {}
  });

  ws.on('close', () => setTimeout(connectLivePrices, 3000));
  ws.on('error', () => {});
}

// ─── Multi-TF S/R Level Detection ─────────────
function findLevels(klines, tfLabel) {
  const highs = klines.map(k => k.h);
  const lows = klines.map(k => k.l);
  const w = Math.max(3, Math.floor(klines.length / 40)); // adaptive window

  const swings = { high: [], low: [] };
  for (let i = w; i < klines.length - w; i++) {
    let isH = true, isL = true;
    for (let j = i - w; j <= i + w; j++) {
      if (j === i) continue;
      if (klines[j].h >= klines[i].h) isH = false;
      if (klines[j].l <= klines[i].l) isL = false;
    }
    if (isH) swings.high.push({ price: klines[i].h, idx: i });
    if (isL) swings.low.push({ price: klines[i].l, idx: i });
  }

  function merge(arr) {
    if (!arr.length) return [];
    arr.sort((a, b) => a.price - b.price);
    const r = [];
    let cur = [arr[0]];
    for (let i = 1; i < arr.length; i++) {
      if (Math.abs(arr[i].price - cur[0].price) / cur[0].price < 0.008) cur.push(arr[i]);
      else { r.push({ price: cur.reduce((s, l) => s + l.price, 0) / cur.length, touches: cur.length }); cur = [arr[i]]; }
    }
    r.push({ price: cur.reduce((s, l) => s + l.price, 0) / cur.length, touches: cur.length });
    return r.map(x => ({ ...x, price: round(x.price, 2), tf: tfLabel }));
  }

  const supports = merge(swings.low);
  const resistances = merge(swings.high);
  return { supports, resistances, tf: tfLabel };
}

function strengthStars(touches) {
  if (touches >= 7) return '★★★★★';
  if (touches >= 5) return '★★★★☆';
  if (touches >= 3) return '★★★☆☆';
  if (touches >= 2) return '★★☆☆☆';
  return '★☆☆☆☆';
}

// ─── Full Analysis: refresh all levels ────────
async function refreshAnalysis() {
  for (const pair of MONITORED) {
    try {
      // Fetch multiple timeframes
      const [k1h, k4h, k1d] = await Promise.all([
        fetchKlines(pair, '1h', 200),
        fetchKlines(pair, '4h', 200),
        fetchKlines(pair, '1d', 200),
      ]);

      const price = k1h[k1h.length - 1]?.c || 0;
      if (!price) continue;

      // Multi-timeframe levels
      const hourly = findLevels(k1h, 'H');
      const daily4h = findLevels(k4h, 'D');
      const daily = findLevels(k1d, 'D');
      const weekly = findLevels(k1d.slice(-50), 'W');
      const monthly = findLevels(k1d, 'M');

      // Combine all S/R (deduplicate by proximity)
      const allSupports = [...hourly.supports, ...daily.supports, ...weekly.supports, ...monthly.supports]
        .filter(l => l.price < price)
        .sort((a, b) => b.price - a.price);
      const allResistances = [...hourly.resistances, ...daily.resistances, ...weekly.resistances, ...monthly.resistances]
        .filter(l => l.price > price)
        .sort((a, b) => a.price - b.price);

      function dedupe(arr, maxCount) {
        const r = [];
        for (const l of arr) {
          if (r.some(x => Math.abs(x.price - l.price) / l.price < 0.005)) continue;
          r.push({ ...l, distPct: round((l.price / price - 1) * 100, 2), stars: strengthStars(l.touches) });
        }
        return r.slice(0, maxCount);
      }

      // VWAP & EMA
      const closes = k1h.map(k => k.c);
      let cumPV = 0, cumVol = 0;
      for (const k of k1h) { cumPV += (k.h + k.l + k.c) / 3 * k.v; cumVol += k.v; }
      const vwap = cumVol > 0 ? cumPV / cumVol : price;
      const ema20 = calcEMA20(closes);

      // Range detection
      const recent20 = k1h.slice(-20);
      const rHigh = Math.max(...recent20.map(k => k.h));
      const rLow = Math.min(...recent20.map(k => k.l));
      const rangePct = (rHigh - rLow) / rLow * 100;
      const isChoppy = rangePct < 5 && rangePct > 0.5;

      keyLevels[pair] = {
        supports: dedupe(allSupports, 3),
        resistances: dedupe(allResistances, 3),
        vwap: round(vwap, 2),
        ema20: round(ema20[ema20.length - 1], 2),
        rangePct: round(rangePct, 2),
        rangeHigh: round(rHigh, 2),
        rangeLow: round(rLow, 2),
        isChoppy,
      };

      // Compute edge levels (high-impact key levels)
      try {
        edgeLevels[pair] = await computeEdgeLevels(pair);
        console.log(`  Edge levels computed for ${pair}`);
      } catch (e) {
        console.log(`  Edge levels failed for ${pair}: ${e.message}`);
        edgeLevels[pair] = null;
      }

      // Fetch full analysis for signal
      const res = await fetch(`http://localhost:${PORT}/api/analyze/${pair}`).then(r => r.json());
      if (!res || res.error) continue;

      const rsi = res.features?.features?.positive_ratio || 50;
      const regime = res.regimes?.regimeLabel || 'NORMAL';
      const probUp = res.bayesian?.probUp1d || 50;
      const probDown = res.bayesian?.probDown1d || 50;
      const imb = res.orderflow?.tradeImbalance || 0;
      const volRatio = res.features?.features?.volume_trend_10v10 || 1;

      let score = 0;
      if (regime.includes('TRENDING')) score += 8;
      score += (probUp - 50) * 0.6;
      score += imb * 0.2;

      let signal = 'NEUTRAL';
      if (score >= 18) signal = 'BUY';
      else if (score <= -18) signal = 'SELL';

      const prev = lastSignals[pair];
      lastSignals[pair] = { signal, score: round(score, 1), price, time: Date.now() };

      indicators[pair] = { rsi: round(rsi, 1), regime, probUp, probDown, imb: round(imb, 1), volRatio: round(volRatio, 2) };

      // Signal change → Telegram alert
      if (prev && prev.signal !== signal) {
        await sendSignalAlert(pair, signal, score, price);
      }
    } catch (_) {}
  }
}

// ─── Send signal change alert ─────────────────
async function sendSignalAlert(pair, signal, score, price) {
  const lv = keyLevels[pair];
  const ind = indicators[pair];
  const label = pair.replace('USDT', '');
  const arrow = signal === 'BUY' ? '🟢' : '🔴';
  const px = livePrices[pair] || {};
  const chg = (px.chg24h || 0).toFixed(1);
  const lo = px.low24h || price;
  const hi = px.high24h || price;

  let s1 = lv?.supports[0] ? `${lv.supports[0].price} ${lv.supports[0].tf}${lv.supports[0].stars}` : '—';
  let r1 = lv?.resistances[0] ? `${lv.resistances[0].price} ${lv.resistances[0].tf}${lv.resistances[0].stars}` : '—';
  let range = lv?.isChoppy ? ` | RANGE ${lv.rangePct}%` : '';

  let msg = `\n${arrow} ${label} $${price}  ${signal} [${score > 0 ? '+' : ''}${score}]  ${chg}%\n`;
  msg += `───────────────────────────\n`;
  msg += `RSI:${ind.rsi} | ${ind.regime} | Flow:${ind.imb > 0 ? '+' : ''}${ind.imb}% | ↑${ind.probUp}%↓${ind.probDown}%\n`;
  msg += `S:${s1} | R:${r1}${range}\n`;
  msg += `24h: $${lo} – $${hi}\n`;

  if (lv?.isChoppy) {
    msg += `OPTION: Put<${round(lv.rangeLow * 0.98, 0)} Call>${round(lv.rangeHigh * 1.02, 0)} | Prem~2%\n`;
  }

  msg += `───────────────────────────`;
  await sendAlert(msg, 'high');
}

// ─── EMA20 helper ─────────────────────────────
function calcEMA20(data) {
  const r = []; const mult = 2 / 21; let prev = 0;
  for (let i = 0; i < data.length; i++) {
    if (i < 19) { r.push(null); continue; }
    if (i === 19) { prev = data.slice(0, 20).reduce((a, b) => a + b, 0) / 20; r.push(prev); continue; }
    prev = (data[i] - prev) * mult + prev; r.push(prev);
  }
  return r;
}

// ─── 10-Second Unified Signal ──────────────────
async function priceCheck() {
  const now = Date.now();
  for (const pair of MONITORED) {
    const px = livePrices[pair];
    if (!px || !edgeLevels[pair]) continue;

    try {
      const result = await computeUnifiedSignal(pair, { edgeLevels: edgeLevels[pair] });
      if (result.error) continue;

      const price = px.price;
      const label = pair.replace('USDT', '');
      const chgStr = px.chg24h >= 0 ? `+${px.chg24h.toFixed(2)}%` : `${px.chg24h.toFixed(2)}%`;

      // Console: always show
      console.log(`${new Date().toLocaleTimeString()} | ${label.padEnd(6)} $${price.toFixed(2)} ${chgStr}  ${result.signal} ${result.score > 0 ? '+' : ''}${result.score}  RSI:${result.indicators?.rsi || '--'}`);

      // Telegram: ONE consolidated alert every 60 seconds
      const cooldown = alertCooldown[pair];
      if (cooldown && now - cooldown < 60000) continue;

      const prevSummary = lastAlertMsg[pair];
      if (result.summary && result.summary !== prevSummary) {
        await sendAlert(result.summary, result.signal !== 'NEUTRAL' ? 'high' : 'medium');
        alertCooldown[pair] = now;
        lastAlertMsg[pair] = result.summary;
      }
    } catch (e) {
      // silent retry next cycle
    }
  }
}

// ─── Start ────────────────────────────────────

async function main() {
  const url = await startTunnel();
  connectLivePrices();

  app.listen(PORT, () => {
    console.log(`\n  ═══════════════════════════════════════════════`);
    console.log(`  Market Analysis Server — DEPLOYED`);
    console.log(`  Local:    http://localhost:${PORT}`);
    if (url) console.log(`  Public:   ${url}`);
    else console.log(`  Public:   (tunnel failed — server running locally)`);
    console.log(`  Telegram:  ${cfg.telegramToken ? 'ENABLED — @Yehbebot' : 'DISABLED'}`);
    console.log(`  ═══════════════════════════════════════════════\n`);
  });

  // Wait for data, then start monitoring
  setTimeout(async () => {
    console.log('  Loading initial analysis...');
    await refreshAnalysis();
    console.log('  Monitor active — checking every 10s\n');
  }, 5000);

  // 10-second price + S/R checks
  setInterval(priceCheck, 10000);

  // 5-minute full analysis refresh
  setInterval(refreshAnalysis, 5 * 60 * 1000);

  // Delta Exchange options check every 30 min
  async function deltaCheck() {
    console.log('  Running Delta options check...');
    for (const asset of ['BTC', 'ETH']) {
      try {
        const analysis = await analyzeOptionSelling(asset);
        if (analysis.error) continue;
        const msg = formatDeltaAlert(analysis);
        await sendAlert(msg, 'medium');
      } catch (e) { console.log(`  Delta ${asset}: ${e.message}`); }
    }
  }
  setTimeout(deltaCheck, 15000);
  setInterval(deltaCheck, 10 * 60 * 1000);

  // Commodity check every 15 minutes
  async function commodityCheck() {
    console.log('  Running commodity check...');
    try {
      const results = await analyzeAllCommodities();
      const msg = formatCommodityAlert(results);
      await sendAlert(msg, 'normal');
    } catch (e) { console.log(`  Commodities: ${e.message}`); }
  }
  setTimeout(commodityCheck, 25000);
  setInterval(commodityCheck, 15 * 60 * 1000);

  process.on('SIGINT', () => {
    console.log('\n  Shutting down...');
    process.exit(0);
  });
}

main();
