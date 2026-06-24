// ──────────────────────────────────────────────────
//  SIGNAL-MODEL.MJS — Unified Statistical Engine
//  Processes all data sources into one brief signal
//  Output: direction + confidence + action + rationale
// ──────────────────────────────────────────────────

import { computeEdgeLevels, isNearEdgeLevel } from './edge-levels.mjs';
import { fetchKlines, fetchTicker } from './twelvedata.mjs';
import { fetchFearGreed, getFgScore } from './sentiment.mjs';
import { fetchAllOnchain, formatOnchainAlert } from './coinglass.mjs';

function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

function calcEMA(data, period) {
  const r = []; const mult = 2 / (period + 1);
  let prev = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) { r.push(null); continue; }
    if (i === period - 1) { r.push(prev); continue; }
    prev = (data[i] - prev) * mult + prev; r.push(prev);
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

// ─── MAIN: Unified Signal Model ────────────────
export async function computeUnifiedSignal(symbol, externalData = {}) {
  try {
    const [k1h, ticker, edgeLevels] = await Promise.all([
      fetchKlines(symbol, '1h', 200),
      fetchTicker(symbol),
      externalData.edgeLevels || computeEdgeLevels(symbol),
    ]);

    if (!k1h.length || !ticker) return { error: 'No data' };

    const closes = k1h.map(k => k.c);
    const highs = k1h.map(k => k.h);
    const lows = k1h.map(k => k.l);
    const vols = k1h.map(k => k.v);
    const price = ticker.price || closes[closes.length - 1];

    // ─── 1. TECHNICAL SCORE (40%) ──────────────
    const ema20 = calcEMA(closes, 20);
    const ema50 = calcEMA(closes, 50);
    const rsi = calcRSI(closes, 14);

    const lastEMA20 = ema20[ema20.length - 1];
    const lastEMA50 = ema50[ema50.length - 1];
    const lastRSI = rsi[rsi.length - 1] || 50;

    let techScore = 0;

    // EMA alignment
    if (lastEMA20 && lastEMA50) {
      if (price > lastEMA20) techScore += 10;
      else techScore -= 10;
      if (price > lastEMA50) techScore += 10;
      else techScore -= 10;
      if (lastEMA20 > lastEMA50) techScore += 10;
      else techScore -= 10;
    }

    // RSI
    if (lastRSI < 30) techScore += 15;
    else if (lastRSI > 70) techScore -= 15;
    else if (lastRSI > 55) techScore += 5;
    else if (lastRSI < 45) techScore -= 5;

    // Recent momentum (last 6 candles)
    const recent6 = closes.slice(-6);
    const mom6 = (recent6[5] - recent6[0]) / recent6[0] * 100;
    if (mom6 > 1.5) techScore += 10;
    else if (mom6 < -1.5) techScore -= 10;

    // Normalize to ±100
    techScore = Math.max(-100, Math.min(100, techScore));

    // ─── 2. PROBABILISTIC SCORE (25%) ──────────
    const returns = [];
    for (let i = 1; i < closes.length; i++) returns.push((closes[i] - closes[i - 1]) / closes[i - 1] * 100);
    const meanR = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdR = Math.sqrt(returns.reduce((a, b) => a + (b - meanR) ** 2, 0) / returns.length);
    const dailyVol = stdR * Math.sqrt(24);
    const dailyDrift = meanR * 24;

    const probUp = stdR > 0 ? (1 / (1 + Math.exp(-2 * dailyDrift / dailyVol))) * 100 : 50;
    const probScore = (probUp - 50) * 2; // scale to ±100

    // ─── 3. VOLUME/FLOW SCORE (15%) ──────────
    const avgVol = vols.slice(-20).reduce((a, b) => a + b, 0) / 20;
    const lastVol = vols[vols.length - 1];
    const volRatio = avgVol > 0 ? lastVol / avgVol : 1;

    // Buy vs sell volume
    let buyVol = 0, sellVol = 0;
    for (let i = 0; i < k1h.length; i++) {
      if (closes[i] >= k1h[i].o) buyVol += vols[i];
      else sellVol += vols[i];
    }
    const totalVol = buyVol + sellVol;
    const flowImb = totalVol > 0 ? (buyVol - sellVol) / totalVol * 100 : 0;

    let flowScore = flowImb * 0.5;
    if (volRatio > 1.5 && techScore > 0) flowScore += 10; // volume confirming
    if (volRatio > 1.5 && techScore < 0) flowScore -= 10;
    flowScore = Math.max(-100, Math.min(100, flowScore));

    // ─── 4. STRUCTURAL SCORE (20%) ─────────────
    let structScore = 0;

    // Edge levels
    if (edgeLevels && !edgeLevels.error && edgeLevels.supports && edgeLevels.resistances) {
      const near = isNearEdgeLevel(edgeLevels, 3.0);
      const nearestSupport = near.filter(n => n.side === 'support')[0];
      const nearestResist = near.filter(n => n.side === 'resistance')[0];

      // Closer to strong support → bullish bias
      if (nearestSupport && Math.abs(nearestSupport.distPct) < 2) {
        structScore += nearestSupport.strength * 5;
      }
      // Closer to strong resistance → bearish bias
      if (nearestResist && nearestResist.distPct < 2) {
        structScore -= nearestResist.strength * 5;
      }
    }

    // Range/chop detection
    const recent20 = k1h.slice(-20);
    const rHigh = Math.max(...recent20.map(k => k.h));
    const rLow = Math.min(...recent20.map(k => k.l));
    const rangePct = (rHigh - rLow) / rLow * 100;
    const isChoppy = rangePct < 4 && rangePct > 0.3;
    const isTight = rangePct < 1.5;

    // Position in range
    const rangePos = (price - rLow) / (rHigh - rLow || 1) * 100;
    if (rangePos > 70) structScore -= 10; // near top of range
    if (rangePos < 30) structScore += 10; // near bottom of range

    structScore = Math.max(-100, Math.min(100, structScore));

    // ─── 5. SENTIMENT / ON-CHAIN (10%) ─────────
    let sentiScore = 0;
    try {
      const [fg, onchain] = await Promise.allSettled([
        fetchFearGreed(),
        fetchAllOnchain('BTCUSDT'),
      ]);
      if (fg.status === 'fulfilled' && fg.value) sentiScore += getFgScore(fg.value.value);
      if (onchain.status === 'fulfilled' && onchain.value) sentiScore += (onchain.value.compositeScore || 0);
    } catch (_) {}

    // ─── 6. WEIGHTED COMPOSITE ─────────────────
    const composite = round(
      techScore * 0.35 +
      probScore * 0.20 +
      flowScore * 0.15 +
      structScore * 0.20 +
      sentiScore * 0.10, 1
    );

    // Signal classification
    let signal, confidence;
    if (composite > 60) { signal = 'STRONG BUY'; confidence = round(composite / 100 * 100, 0); }
    else if (composite > 25) { signal = 'BUY'; confidence = round(composite / 100 * 100, 0); }
    else if (composite < -60) { signal = 'STRONG SELL'; confidence = round(Math.abs(composite) / 100 * 100, 0); }
    else if (composite < -25) { signal = 'SELL'; confidence = round(Math.abs(composite) / 100 * 100, 0); }
    else { signal = 'NEUTRAL'; confidence = 0; }

    // ─── 6. FORMAT CONSOLIDATED SUMMARY ────────
    const label = symbol.replace('USDT', '');
    const arrow = signal.includes('BUY') ? '🟢' : signal.includes('SELL') ? '🔴' : '⚪';
    const pctStr = composite > 0 ? `+${composite}` : composite;
    const oddsStr = `↑${round(probUp, 0)}%↓${round(100 - probUp, 0)}%`;

    let summary = `${arrow} ${label} $${round(price, 2)} | ${signal} ${confidence}% | ${pctStr}\n`;

    if (isChoppy) {
    summary += `[Range ${round(rangePct, 1)}%] `;
    if (isTight) summary += `BREAKOUT `;
    } else if (Math.abs(composite) > 25) {
      summary += `[Trending] `;
    }

    summary += `| Odds: ${oddsStr}\n`;

    // Edge levels summary
    if (edgeLevels && !edgeLevels.error) {
      const near2 = isNearEdgeLevel(edgeLevels, 2.5);
      if (near2.length > 0) {
        const nearest = near2[0];
        const edgeLabel = nearest.side === 'support' ? 'S' : 'R';
        summary += `${edgeLabel}:$${nearest.price}[${nearest.distPct}%]${nearest.stars} `;
      }
    }

    summary += `| RSI:${round(lastRSI, 1)} | Vol:${round(volRatio, 1)}x | Flow:${flowImb > 0 ? '+' : ''}${round(flowImb, 1)}%`;

    // Add sentiment/FG one-liner if available
    if (sentiScore > 10) summary += ` | FG:FEAR⬆`;
    else if (sentiScore > 5) summary += ` | FG:FEAR`;
    else if (sentiScore < -10) summary += ` | FG:GREED⬇`;
    else if (sentiScore < -5) summary += ` | FG:GREED`;

    // Action
    if (signal !== 'NEUTRAL') {
      summary += '\n';
      if (signal.includes('BUY')) {
        const sl = edgeLevels?.supports?.[0]?.price;
        const tp = edgeLevels?.resistances?.[0]?.price;
        summary += `SL:$${sl || 'TIGHT'} TP:$${tp || 'OPEN'} | Confirm vol>1.5x`;
      } else {
        const sl = edgeLevels?.resistances?.[0]?.price;
        const tp = edgeLevels?.supports?.[0]?.price;
        summary += `SL:$${sl || 'TIGHT'} TP:$${tp || 'OPEN'} | Confirm vol>1.5x`;
      }
    } else if (isTight) {
      summary += `\n⏳ Tight range — wait for breakout direction`;
    }

    return {
      symbol, label, price, signal, confidence, score: composite,
      summary,
      components: { techScore, probScore, flowScore, structScore, probUp, probDown: round(100 - probUp, 0) },
      structure: { isChoppy, isTight, rangePct: round(rangePct, 2), rangeHigh: round(rHigh, 2), rangeLow: round(rLow, 2) },
      indicators: { rsi: round(lastRSI, 1), volRatio: round(volRatio, 1), flowImb: round(flowImb, 1) },
      edgeLevels: edgeLevels && !edgeLevels.error ? {
        nearestS: edgeLevels.supports?.[0],
        nearestR: edgeLevels.resistances?.[0],
      } : null,
    };
  } catch (e) {
    return { error: e.message };
  }
}

// CLI test
async function main() {
  const r = await computeUnifiedSignal(process.argv[2] || 'BTCUSDT');
  console.log(r.summary);
  console.log('\nComponents:', r.components);
}

const isMain = process.argv[1]?.includes('signal-model.mjs');
if (isMain) main();
