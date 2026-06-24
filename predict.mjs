// ──────────────────────────────────────────────────
//  PREDICT.MJS — Probabilistic Direction + Magnitude
//  KDE + Monte Carlo + Regime Matching
//  Output: direction, expected move, confidence ranges
// ──────────────────────────────────────────────────

import { fetchKlines } from './twelvedata.mjs';

function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

// ─── 1. KDE: Real return distribution ──────────
function kdeEstimate(data, points, bandwidth = null) {
  const n = data.length;
  if (n < 10) return points.map(() => 0);

  // Silverman's rule for bandwidth
  const std = Math.sqrt(data.reduce((s, v) => s + (v - data.reduce((a, b) => a + b, 0) / n) ** 2, 0) / n);
  const h = bandwidth || (1.06 * std * Math.pow(n, -0.2));
  const actualH = Math.max(h, std * 0.1); // don't let bandwidth collapse

  return points.map(x => {
    let sum = 0;
    for (const d of data) {
      const z = (x - d) / actualH;
      sum += Math.exp(-0.5 * z * z) / (actualH * Math.sqrt(2 * Math.PI));
    }
    return sum / n;
  });
}

function kdePercentile(data, p) {
  const n = data.length;
  const h = 1.06 * Math.sqrt(data.reduce((s, v) => s + (v - data.reduce((a, b) => a + b, 0) / n) ** 2, 0) / n) * Math.pow(n, -0.2);
  // Simple: sort and index
  const sorted = [...data].sort((a, b) => a - b);
  return sorted[Math.floor(p / 100 * (n - 1))];
}

// ─── 2. Regime Matching ────────────────────────
function findSimilarRegimes(current, history, topN = 15) {
  const window = current.length;
  const matches = [];

  for (let i = 0; i < history.length - window - 48; i += 4) {
    const hist = history.slice(i, i + window);

    // Similarity based on: volatility, trend, volume profile
    const curVol = stdDev(current);
    const histVol = stdDev(hist);
    const volDiff = Math.abs(curVol - histVol) / Math.max(curVol, 0.001);

    const curTrend = linearTrend(current);
    const histTrend = linearTrend(hist);
    const trendDiff = Math.abs(curTrend - histTrend) / Math.max(Math.abs(curTrend), 0.001);

    const similarity = volDiff * 0.6 + trendDiff * 0.4;

    if (similarity < 1.5) {
      // What happened in the next 24 periods?
      const outcome = history.slice(i + window, i + window + 24);
      matches.push({
        idx: i,
        similarity: round(similarity, 2),
        outcome24: outcome.length > 0 ? outcome[outcome.length - 1] : history[i + window],
        outcomePct: outcome.length > 0 ? round((outcome[outcome.length - 1] - history[i + window - 1]) / history[i + window - 1] * 100, 2) : 0,
      });
    }
  }

  return matches.sort((a, b) => a.similarity - b.similarity).slice(0, topN);
}

function stdDev(arr) {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
}

function linearTrend(arr) {
  const n = arr.length;
  const xMean = (n - 1) / 2;
  const yMean = arr.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (arr[i] - yMean);
    den += (i - xMean) ** 2;
  }
  return den > 0 ? num / den : 0;
}

// ─── 3. Monte Carlo Simulation ─────────────────
function monteCarloSim(returns, price, numPaths = 10000, steps = 24) {
  const mu = returns.reduce((a, b) => a + b, 0) / returns.length;
  const sigma = stdDev(returns);

  const outcomes = [];
  const mins = [];
  const maxs = [];

  for (let p = 0; p < numPaths; p++) {
    let px = price;
    let minPx = px, maxPx = px;
    for (let s = 0; s < steps; s++) {
      // Sample from actual return distribution (not normal)
      const r = returns[Math.floor(Math.random() * returns.length)];
      // Cap each step to ±10% to prevent numerical explosion
      const cappedR = Math.max(-10, Math.min(10, r));
      px *= (1 + cappedR / 100);
      // Prevent numerical explosion
      if (px > price * 10 || px < price * 0.1) px = price;
      if (px < minPx) minPx = px;
      if (px > maxPx) maxPx = px;
    }
    outcomes.push(px);
    mins.push(minPx);
  }

  outcomes.sort((a, b) => a - b);

  const p05 = outcomes[Math.floor(numPaths * 0.05)];
  const p16 = outcomes[Math.floor(numPaths * 0.16)];
  const p50 = outcomes[Math.floor(numPaths * 0.50)];
  const p84 = outcomes[Math.floor(numPaths * 0.84)];
  const p95 = outcomes[Math.floor(numPaths * 0.95)];

  const upCount = outcomes.filter(p => p > price).length;
  const downCount = outcomes.filter(p => p < price).length;

  return {
    expected: round(p50, 2),
    expectedPct: round((p50 / price - 1) * 100, 2),
    probUp: round(upCount / numPaths * 100, 1),
    probDown: round(downCount / numPaths * 100, 1),
    ci68: { low: round(p16, 2), high: round(p84, 2) },
    ci95: { low: round(p05, 2), high: round(p95, 2) },
    worstCase: round(Math.min(...mins), 2),
    worstCasePct: round((Math.min(...mins) / price - 1) * 100, 1),
    bestCase: (() => {
      const v = maxs.filter(x => isFinite(x));
      return round(v.length > 0 ? v.reduce((a, b) => a > b ? a : b) : price, 2);
    })(),
    bestCasePct: (() => {
      const v = maxs.filter(x => isFinite(x));
      const bm = v.length > 0 ? v.reduce((a, b) => a > b ? a : b) : price;
      return round((bm / price - 1) * 100, 1);
    })(),
  };
}

// ─── 4. MAIN: Direction + Magnitude Prediction ──
export async function predictMove(symbol, horizonHours = 24) {
  try {
    const candles = await fetchKlines(symbol, '1h', 200);
    if (candles.length < 50) return { error: 'Insufficient data' };

    const closes = candles.map(k => k.c);
    const price = closes[closes.length - 1];
    const returns = [];
    for (let i = 1; i < closes.length; i++) {
      returns.push((closes[i] - closes[i - 1]) / closes[i - 1] * 100);
    }

    const steps = horizonHours; // 1h candles

    // 1. Monte Carlo using actual return distribution
    const mc = monteCarloSim(returns, price, 10000, steps);

    // 2. Regime matching
    const recent = closes.slice(-24); // last 24 hours
    const matches = findSimilarRegimes(recent, closes);

    let regimeUp = 0, regimeDown = 0, regimeSum = 0;
    for (const m of matches) {
      if (m.outcomePct > 0) regimeUp += m.outcomePct * (1 - m.similarity);
      else regimeDown += Math.abs(m.outcomePct) * (1 - m.similarity);
      regimeSum += (1 - m.similarity);
    }
    const regimeUpAvg = regimeSum > 0 ? round(regimeUp / regimeSum, 1) : 0;
    const regimeDownAvg = regimeSum > 0 ? round(regimeDown / regimeSum, 1) : 0;

    // 3. Direction probability (weighted: MC 60%, Regime 40%)
    const probUp = round(mc.probUp * 0.6 + (matches.filter(m => m.outcomePct > 0).length / Math.max(matches.length, 1)) * 100 * 0.4, 1);
    const probDown = round(100 - probUp, 1);

    // 4. Expected move (weighted average)
    const mcExpected = mc.expectedPct;
    const regimeExpected = regimeUp > 0 || regimeDown > 0
      ? round(((regimeUp - regimeDown) / regimeSum), 1)
      : 0;
    const expectedPct = round(mcExpected * 0.6 + regimeExpected * 0.4, 1);

    // 5. Confidence
    const regimeQuality = matches.length >= 10 ? 'HIGH' : matches.length >= 5 ? 'MEDIUM' : 'LOW';
    const volatility = stdDev(returns) * Math.sqrt(steps);
    const volConfidence = volatility < 5 ? 'HIGH' : volatility < 10 ? 'MEDIUM' : 'LOW';
    const confidence = regimeQuality === 'HIGH' ? 'HIGH' :
                       regimeQuality === 'MEDIUM' && volConfidence !== 'LOW' ? 'MEDIUM' : 'LOW';

    // 6. Format
    const label = symbol.replace('USDT', '');
    const direction = probUp >= 60 ? '🟢 LIKELY UP' :
                      probUp >= 50 ? '🟢 SLIGHTLY UP' :
                      probDown >= 60 ? '🔴 LIKELY DOWN' : '🔴 SLIGHTLY DOWN';

    const summary = `${label} $${round(price, 2)} → ${horizonHours}H\n` +
      `───────────────────────────\n` +
      `Direction: ↑${probUp}% ↓${probDown}% | ${direction}\n` +
      `Expected: ${expectedPct > 0 ? '+' : ''}${expectedPct}% ($${round(price * (1 + expectedPct / 100), 0)})\n` +
      `68% Range: $${round(mc.ci68.low, 0)} – $${round(mc.ci68.high, 0)}\n` +
      `95% Range: $${round(mc.ci95.low, 0)} – $${round(mc.ci95.high, 0)}\n` +
      `Worst: ${mc.worstCasePct}% ($${round(mc.worstCase, 0)}) | Best: +${mc.bestCasePct}% ($${round(mc.bestCase, 0)})\n` +
      `Confidence: ${confidence} | ${matches.length} similar regimes matched`;

    return {
      symbol, label, price,
      horizonHours,
      probUp, probDown,
      expectedPct,
      expectedPrice: round(price * (1 + expectedPct / 100), 2),
      direction,
      confidence,
      mc: {
        expectedPct: mc.expectedPct,
        probUp: mc.probUp,
        probDown: mc.probDown,
        ci68: mc.ci68,
        ci95: mc.ci95,
        worstCasePct: mc.worstCasePct,
        bestCasePct: mc.bestCasePct,
      },
      regime: {
        matches: matches.length,
        avgUp: regimeUpAvg,
        avgDown: regimeDownAvg,
        quality: regimeQuality,
      },
      summary,
    };
  } catch (e) {
    return { error: e.message };
  }
}

// CLI test
const isMain = process.argv[1]?.includes('predict.mjs');
if (isMain) {
  const sym = process.argv[2] || 'BTCUSDT';
  const hrs = parseInt(process.argv[3]) || 24;
  const result = await predictMove(sym, hrs);
  console.log(result.summary);
  console.log('\nMC:', result.mc);
  console.log('Regime:', result.regime);
}
