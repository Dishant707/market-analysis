// ──────────────────────────────────────────────────
//  LIVE TRADE MONITOR
//  Real-time WebSocket prices + periodic analysis +
//  alert system for signal changes and key events
// ──────────────────────────────────────────────────

import { WebSocket } from 'ws';
import { analyzeAsset } from './trade-engine.mjs';
import { spawn } from 'child_process';
import fs from 'fs';

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
  return { telegramToken: cfg.TELEGRAM_TOKEN || '', telegramChatId: cfg.TELEGRAM_CHAT_ID || '' };
}
const env = loadEnv();

// ─── Telegram Alert ───────────────────────────
async function telegramAlert(message, priority = 'normal') {
  if (!env.telegramToken || !env.telegramChatId) return;
  try {
    const prefix = priority === 'high' ? '🔴 ' : priority === 'medium' ? '🟡 ' : '';
    const text = encodeURIComponent(`${prefix}${message}`);
    await fetch(`https://api.telegram.org/bot${env.telegramToken}/sendMessage?chat_id=${env.telegramChatId}&text=${text}&parse_mode=HTML`);
  } catch (_) {}
}

// ─── Configuration ─────────────────────────────
const ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'PAXGUSDT', 'XAUTUSDT'];
const BINANCE_WS = 'wss://stream.binance.com:9443/ws';
const ANALYSIS_INTERVAL = 5 * 60 * 1000; // 5 minutes
const TICKER_INTERVAL = 15 * 60 * 1000;  // 15-min ticker refresh
const ALERT_LOG = './alerts.log';

// ─── State ─────────────────────────────────────
let latestPrices = {};
let latestTickers = {};
let lastSignals = {};
let alerts = [];

// ─── Display ───────────────────────────────────
const C = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
  white: '\x1b[37m',
};

function logAlert(msg, level = 'INFO') {
  const ts = new Date().toISOString();
  const colors = { INFO: C.cyan, WARN: C.yellow, CRITICAL: C.red };
  const entry = `[${ts}] [${level}] ${msg}`;
  alerts.push({ ts, level, msg });
  if (alerts.length > 100) alerts.shift();

  // Console output
  const c = colors[level] || '';
  console.log(`${c}${entry}${C.reset}`);

  // File log
  fs.appendFileSync(ALERT_LOG, entry + '\n');

  // Bell for critical
  if (level === 'CRITICAL') process.stdout.write('\x07');

  // Telegram notification for WARN/CRITICAL
  if (level === 'WARN' || level === 'CRITICAL') {
    telegramAlert(msg, level === 'CRITICAL' ? 'high' : 'medium');
  }
}

// ─── WebSocket Feed ────────────────────────────
function connectWebSocket() {
  const streams = ASSETS.flatMap(s => [
    `${s.toLowerCase()}@trade`,
    `${s.toLowerCase()}@ticker`,
    `${s.toLowerCase()}@kline_1h`,
  ]).join('/');

  const ws = new WebSocket(`${BINANCE_WS}/${streams}`);

  ws.on('open', () => {
    logAlert('WebSocket connected', 'INFO');
  });

  ws.on('message', raw => {
    try {
      const data = JSON.parse(raw.toString());
      handleStreamData(data);
    } catch (_) {}
  });

  ws.on('close', () => {
    logAlert('WebSocket disconnected — reconnecting in 5s', 'WARN');
    setTimeout(connectWebSocket, 5000);
  });

  ws.on('error', (e) => {
    logAlert(`WebSocket error: ${e.message}`, 'WARN');
  });

  return ws;
}

function handleStreamData(data) {
  if (data.e === 'trade') {
    latestPrices[data.s] = {
      price: parseFloat(data.p),
      qty: parseFloat(data.q),
      side: data.m ? 'SELL' : 'BUY',
      time: data.T,
    };
  } else if (data.e === '24hrTicker') {
    latestTickers[data.s] = {
      price: parseFloat(data.c),
      change: parseFloat(data.P),
      high: parseFloat(data.h),
      low: parseFloat(data.l),
      vol: parseFloat(data.q),
      time: data.E,
    };
  }
}

// ─── Analysis Runner ───────────────────────────
async function runAnalysis() {
  const start = Date.now();
  let changes = 0;

  for (const symbol of ASSETS) {
    try {
      const result = await analyzeAsset(symbol);
      if (result.error) continue;

      const prev = lastSignals[symbol];

      // Detect signal changes
      if (prev) {
        if (prev.signal !== result.signal) {
          changes++;
          const level = (result.signal.includes('STRONG') || prev.signal.includes('STRONG'))
            ? 'CRITICAL' : 'WARN';

          logAlert(
            `${result.label} SIGNAL FLIP: ${prev.signal} → ${result.signal} ` +
            `(Score: ${prev.score} → ${result.score}) ` +
            `Price: $${result.price}`,
            level
          );
        } else if (Math.abs(result.score - prev.score) > 15) {
          // Significant score change but same signal
          logAlert(
            `${result.label} SCORE SPIKE: ${prev.signal} ${prev.score > 0 ? '+' : ''}${prev.score} → ${result.score > 0 ? '+' : ''}${result.score}`,
            'WARN'
          );
        }
      }

      // Detect key levels
      if (result.structure) {
        const { nearestSupport, nearestResistance } = result.structure;
        const price = result.price;

        if (nearestSupport && price <= nearestSupport * 1.005 && price >= nearestSupport) {
          logAlert(`${result.label} TOUCHING SUPPORT $${nearestSupport} (dist: ${((price / nearestSupport - 1) * 100).toFixed(2)}%)`, 'WARN');
        }
        if (nearestResistance && price >= nearestResistance * 0.995 && price <= nearestResistance) {
          logAlert(`${result.label} TOUCHING RESISTANCE $${nearestResistance}`, 'WARN');
        }
      }

      // Volume surge
      if (result.volume && result.volume.volRatio > 2) {
        logAlert(`${result.label} VOLUME SURGE ${result.volume.volRatio}x avg`, 'WARN');
      }

      // Divergence detected
      for (const d of (result.divergences || [])) {
        logAlert(`${result.label} DIVERGENCE: ${d.type.toUpperCase()}`, 'WARN');
      }

      lastSignals[symbol] = {
        signal: result.signal,
        score: result.score,
        price: result.price,
        time: new Date().toISOString(),
      };
    } catch (e) {
      // Silent — will retry next cycle
    }
  }

  if (changes === 0) {
    logAlert(`Analysis cycle complete — no signal changes (${((Date.now() - start) / 1000).toFixed(1)}s)`, 'INFO');
  }
}

// ─── Dashboard Display ─────────────────────────
function renderDashboard() {
  console.clear();
  console.log(`${C.bold}${C.cyan}╔═══════════════════════════════════════════════════════════════╗${C.reset}`);
  console.log(`${C.bold}${C.cyan}║${C.reset}  ${C.bold}LIVE TRADE MONITOR${C.reset}   ${C.dim}${new Date().toLocaleString()}${C.reset}`.padEnd(64) + `${C.bold}${C.cyan}║${C.reset}`);
  console.log(`${C.bold}${C.cyan}╠═══════════════════════════════════════════════════════════════╣${C.reset}`);

  // Asset prices and signals
  for (const symbol of ASSETS) {
    const ticker = latestTickers[symbol];
    const signal = lastSignals[symbol];
    const label = symbol.replace('USDT', '/USDT');

    if (!ticker) {
      console.log(`${C.bold}${C.cyan}║${C.reset}  ${label.padEnd(8)} ${C.dim}waiting for data...${C.reset}`);
      continue;
    }

    const price = ticker.price;
    const chg = ticker.change || 0;
    const chgStr = chg >= 0 ? `${C.green}+${chg.toFixed(2)}%${C.reset}` : `${C.red}${chg.toFixed(2)}%${C.reset}`;
    const priceStr = `$${price >= 1000 ? price.toFixed(2) : price < 1 ? price.toFixed(4) : price.toFixed(2)}`;

    let sigStr = '';
    if (signal) {
      const sColor = signal.signal.includes('BUY') ? C.green :
                     signal.signal.includes('SELL') ? C.red : C.yellow;
      sigStr = `  ${sColor}${C.bold}[${signal.signal}]${C.reset} ${signal.score > 0 ? '+' : ''}${signal.score}%`;
    }

    const volStr = ticker.vol ? `  Vol: $${(ticker.vol / 1e6).toFixed(1)}M` : '';
    console.log(`${C.bold}${C.cyan}║${C.reset}  ${C.bold}${label.padStart(10)}${C.reset}  ${C.bold}${priceStr}${C.reset}  ${chgStr}${sigStr}${volStr}`);
  }

  console.log(`${C.bold}${C.cyan}╠═══════════════════════════════════════════════════════════════╣${C.reset}`);

  // Recent alerts (last 8)
  const recentAlerts = alerts.slice(-8);
  if (recentAlerts.length > 0) {
    for (const a of recentAlerts) {
      const aColor = a.level === 'CRITICAL' ? C.red : a.level === 'WARN' ? C.yellow : C.cyan;
      const shortTime = new Date(a.ts).toLocaleTimeString();
      console.log(`${C.bold}${C.cyan}║${C.reset}  ${C.dim}${shortTime}${C.reset} ${aColor}${a.msg.substring(0, 55)}${C.reset}`);
    }
  } else {
    console.log(`${C.bold}${C.cyan}║${C.reset}  ${C.dim}No alerts yet — building data...${C.reset}`);
  }

  console.log(`${C.bold}${C.cyan}╚═══════════════════════════════════════════════════════════════╝${C.reset}`);
  console.log(`${C.dim}  Analysis every ${ANALYSIS_INTERVAL / 60000}min | Ctrl+C to stop | Alerts → ${ALERT_LOG}${C.reset}\n`);
}

// ─── Python ML Sidecar ─────────────────────────
async function runPythonML() {
  return new Promise((resolve) => {
    const py = spawn('python3', ['analysis/weekly_ml.py'], {
      cwd: process.cwd(),
      timeout: 30000,
    });

    let out = '';
    py.stdout.on('data', d => out += d.toString());
    py.stderr.on('data', () => {});

    py.on('close', () => {
      try {
        const result = JSON.parse(out);
        resolve(result);
      } catch (_) {
        resolve(null);
      }
    });

    py.on('error', () => resolve(null));
  });
}

// ─── Main Loop ─────────────────────────────────
async function main() {
  console.clear();
  logAlert('Trade Monitor starting...', 'INFO');

  // Connect WebSocket
  const ws = connectWebSocket();

  // Initial analysis
  logAlert('Running initial analysis...', 'INFO');
  await runAnalysis();

  // Start periodic analysis
  setInterval(async () => {
    await runAnalysis();
  }, ANALYSIS_INTERVAL);

  // Start dashboard refresh
  setInterval(() => {
    renderDashboard();
  }, 5000);

  // Start ticker refresh (every 15 min via analysis includes it)
  // ML prediction once per hour
  setTimeout(async () => {
    logAlert('Running ML prediction...', 'INFO');
    const ml = await runPythonML();
    if (ml && ml.prediction) {
      logAlert(`ML WEEKLY BTC: ${ml.prediction.nextWeek} (${ml.prediction.upProb}% up, ${ml.prediction.downProb}% down) · ${ml.prediction.confidence} confidence`, 'INFO');
    }
    // Repeat every hour
    setInterval(async () => {
      logAlert('Running ML prediction...', 'INFO');
      const ml = await runPythonML();
      if (ml && ml.prediction) {
        logAlert(`ML WEEKLY BTC: ${ml.prediction.nextWeek} (${ml.prediction.upProb}% up) · ${ml.prediction.confidence}`, 'INFO');
      }
    }, 60 * 60 * 1000);
  }, 60 * 60 * 1000); // First run after 1 hour

  // Graceful shutdown
  process.on('SIGINT', () => {
    logAlert('Shutting down monitor...', 'INFO');
    ws.close();
    process.exit(0);
  });

  // Initial dashboard
  renderDashboard();
}

main();
