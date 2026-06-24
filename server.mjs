import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
const PORT = 3000;

// ─── Persistent Rust Engine ──────────────────
let engine = null;
let buf = '';
let pending = [];
let ready = false;
let rid = 0;

function startEngine() {
  if (engine) return;
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
}

function rustCall(action, data) {
  return new Promise((resolve, reject) => {
    if (!engine || !ready) {
      // Fallback to one-shot
      const p = spawn('./rust-engine/target/release/modern_engine', [], { cwd: __dirname, stdio: ['pipe', 'pipe', 'pipe'] });
      let o = '', e = '';
      p.stdout.on('data', d => o += d);
      p.stderr.on('data', d => e += d);
      p.on('close', () => {
        try {
          const lines = o.trim().split('\n').filter(l => l.trim());
          const last = JSON.parse(lines[lines.length - 1]);
          resolve(last);
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

// ─── API Endpoints ──────────────────────────

// Live market data (Binance)
async function fetchKlines(symbol, interval = '1h', limit = 200) {
  const res = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`);
  if (!res.ok) throw new Error(`${symbol}: ${res.status}`);
  const data = await res.json();
  return data.map(k => ({ t: k[0], o: +k[1], h: +k[2], l: +k[3], c: +k[4], v: +k[5] }));
}

async function fetchBook(symbol, limit = 50) {
  const res = await fetch(`https://api.binance.com/api/v3/depth?symbol=${symbol}&limit=${limit}`);
  if (!res.ok) throw new Error(`${symbol}: ${res.status}`);
  return await res.json();
}

async function fetchTrades(symbol, limit = 100) {
  const res = await fetch(`https://api.binance.com/api/v3/trades?symbol=${symbol}&limit=${limit}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.map(t => ({ price: +t.price, qty: +t.qty, side: t.isBuyerMaker ? 'sell' : 'buy', time: t.time }));
}

// GET /api/analyze/:pair — Full modern analysis
app.get('/api/analyze/:pair', async (req, res) => {
  try {
    const pair = req.params.pair.toUpperCase();
    const interval = req.query.interval || '1h';
    const limit = parseInt(req.query.limit) || 200;

    const [klines, book, trades] = await Promise.all([
      fetchKlines(pair, interval, limit),
      fetchBook(pair, 50),
      fetchTrades(pair, 100),
    ]);

    if (klines.length < 20) return res.status(400).json({ error: 'insufficient data' });

    const price = klines[klines.length - 1].c;
    const closes = klines.map(k => k.c);
    const highs = klines.map(k => k.h);
    const lows = klines.map(k => k.l);
    const volumes = klines.map(k => k.v);

    const start = Date.now();

    // Run ALL analyses in parallel
    const [orderflow, regimes, features, bayesian] = await Promise.all([
      rustCall('orderflow', {
        trades, book_bids: book.bids, book_asks: book.asks, klines, price,
      }),
      rustCall('regimes', { closes, highs, lows, volumes, num_regimes: 3 }),
      rustCall('features', { closes, highs, lows, volumes }),
      rustCall('bayesian', { closes, price, targets: [], horizon_hours: 24 }),
    ]);

    res.json({
      pair, price, interval, candles: klines.length, elapsed: Date.now() - start,
      orderflow: orderflow.orderflow,
      regimes: regimes.regimes,
      features: features.features,
      bayesian: bayesian.bayesian,
    });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// GET /api/orderflow/:pair — Just order flow
app.get('/api/orderflow/:pair', async (req, res) => {
  try {
    const pair = req.params.pair.toUpperCase();
    const [klines, book, trades] = await Promise.all([
      fetchKlines(pair, '1h', 100), fetchBook(pair, 50), fetchTrades(pair, 100),
    ]);
    const price = klines.length > 0 ? klines[klines.length - 1].c : 0;
    const result = await rustCall('orderflow', { trades, book_bids: book.bids, book_asks: book.asks, klines, price });
    res.json({ pair, price, ...result.orderflow });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// GET /api/regimes/:pair — Regime detection
app.get('/api/regimes/:pair', async (req, res) => {
  try {
    const pair = req.params.pair.toUpperCase();
    const klines = await fetchKlines(pair, '1h', 200);
    const closes = klines.map(k => k.c);
    const highs = klines.map(k => k.h);
    const lows = klines.map(k => k.l);
    const volumes = klines.map(k => k.v);
    const result = await rustCall('regimes', { closes, highs, lows, volumes, num_regimes: 3 });
    res.json({ pair, price: closes[closes.length - 1], ...result.regimes });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// GET /api/features/:pair — Feature extraction
app.get('/api/features/:pair', async (req, res) => {
  try {
    const pair = req.params.pair.toUpperCase();
    const klines = await fetchKlines(pair, '1h', 200);
    const closes = klines.map(k => k.c);
    const highs = klines.map(k => k.h);
    const lows = klines.map(k => k.l);
    const volumes = klines.map(k => k.v);
    const result = await rustCall('features', { closes, highs, lows, volumes });
    res.json({ pair, price: closes[closes.length - 1], ...result.features });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// GET /api/bayesian/:pair — Bayesian probability
app.get('/api/bayesian/:pair', async (req, res) => {
  try {
    const pair = req.params.pair.toUpperCase();
    const klines = await fetchKlines(pair, '1h', 200);
    const closes = klines.map(k => k.c);
    const price = closes[closes.length - 1];
    const targets = req.query.targets
      ? req.query.targets.split(',').map(Number)
      : Array.from({ length: 21 }, (_, i) => price * (0.9 + i * 0.01));
    const result = await rustCall('bayesian', { closes, price, targets, horizon_hours: 24 });
    res.json({ pair, price, ...result.bayesian });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// GET /api/options/:pair — Option selling analysis
app.get('/api/options/:pair', async (req, res) => {
  try {
    const pair = req.params.pair.toUpperCase();
    const days = parseInt(req.query.days) || 7;
    const premiumPct = parseFloat(req.query.premium) || 2.0;
    const klines = await fetchKlines(pair, '1d', 200);
    const closes = klines.map(k => k.c);
    const price = closes[closes.length - 1];

    // Generate strikes from -15% to +15% in 1% steps
    const strikes = [];
    for (let i = -15; i <= 15; i++) {
      strikes.push(price * (1 + i / 100));
    }
    // Add round number strikes near price
    const step = price > 10000 ? 1000 : price > 1000 ? 100 : 10;
    const base = Math.round(price / step) * step;
    for (let i = -10; i <= 10; i++) {
      const s = base + i * step;
      if (!strikes.some(x => Math.abs(x - s) / s < 0.005)) strikes.push(s);
    }
    strikes.sort((a, b) => a - b);

    const result = await rustCall('options', {
      closes, price, strikes,
      days_to_expiry: days,
      premium_percent: premiumPct,
    });
    res.json({ pair, price, daysToExpiry: days, ...result.options });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ─── Python ML Sidecar ───────────────────────
async function runPythonML(symbol) {
  return new Promise((resolve) => {
    const py = spawn('python3', ['analysis/weekly_ml.py'], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let out = '', err = '';
    py.stdout.on('data', d => out += d.toString());
    py.stderr.on('data', d => err += d.toString());
    py.on('close', () => {
      try { resolve(JSON.parse(out)); }
      catch (_) { resolve({ error: err || 'ml parse error' }); }
    });
    py.on('error', e => resolve({ error: e.message }));
  });
}

// GET /api/ml/:symbol — ML weekly prediction
app.get('/api/ml/:symbol', async (req, res) => {
  try {
    const pair = req.params.symbol.toUpperCase();
    const result = await runPythonML(pair);
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// GET /api/signals — Multi-asset trading signals from trade engine
app.get('/api/signals', async (req, res) => {
  try {
    const assets = (req.query.assets || 'BTCUSDT,ETHUSDT,SOLUSDT,PAXGUSDT,XAUTUSDT').split(',');
    const results = {};
    for (const a of assets) {
      try {
        const r = await rustCall('orderflow', {});
        results[a] = { status: 'ok' };
      } catch (_) { results[a] = { status: 'error' }; }
    }
    res.json(results);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// Static dashboard
app.use(express.static(path.join(__dirname, 'public')));

app.listen(PORT, () => {
  console.log(`\n  Modern Market Analysis Server`);
  console.log(`  http://localhost:${PORT}`);
  console.log(`  http://localhost:${PORT}/api/analyze/BTCUSDT`);
  console.log(`  http://localhost:${PORT}/api/ml/BTCUSDT\n`);
});
