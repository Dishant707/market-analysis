// ──────────────────────────────────────────────────
//  EDGE-LEVELS.MJS — High-Impact Key Level Detection
//  Finds levels where major moves originate
//  Filters choppy zones, only alerts on "edge" levels
// ──────────────────────────────────────────────────

import { fetchKlines } from './twelvedata.mjs';

// ─── Find swing points with impact analysis ────
function findSwingPoints(klines, window = 5) {
  const swings = [];
  const len = klines.length;

  for (let i = window; i < len - window; i++) {
    // Check if swing high
    let isHigh = true;
    for (let j = i - window; j <= i + window; j++) {
      if (j === i) continue;
      if (klines[j].h >= klines[i].h) { isHigh = false; break; }
    }
    if (isHigh) {
      swings.push({
        type: 'high',
        price: klines[i].h,
        idx: i,
        time: klines[i].t,
        candlesAfter: len - i - 1,
      });
    }

    // Check if swing low
    let isLow = true;
    for (let j = i - window; j <= i + window; j++) {
      if (j === i) continue;
      if (klines[j].l <= klines[i].l) { isLow = false; break; }
    }
    if (isLow) {
      swings.push({
        type: 'low',
        price: klines[i].l,
        idx: i,
        time: klines[i].t,
        candlesAfter: len - i - 1,
      });
    }
  }
  return swings;
}

// ─── Calculate impact: what happened within N candles ───
function calculateImpact(klines, swingIdx, lookAhead = 20) {
  const startPrice = swingIdx < klines.length
    ? (klines[swingIdx].h + klines[swingIdx].l) / 2
    : 0;

  if (swingIdx >= klines.length - 3) {
    // Recent swing — check prior behavior instead
    return calculatePriorImpact(klines, swingIdx);
  }

  const end = Math.min(swingIdx + lookAhead, klines.length);
  let maxMove = 0;
  let direction = 'none';

  for (let i = swingIdx + 1; i < end; i++) {
    const move = ((klines[i].c - startPrice) / startPrice) * 100;
    if (Math.abs(move) > Math.abs(maxMove)) {
      maxMove = move;
      direction = move > 0 ? 'up' : 'down';
    }
  }

  return {
    movePct: round(maxMove, 1),
    direction,
    magnitude: round(Math.abs(maxMove), 1),
    isSignificant: Math.abs(maxMove) > 5,
    lookAhead,
  };
}

function calculatePriorImpact(klines, swingIdx) {
  // Look at what happened BEFORE this level formed
  const start = Math.max(0, swingIdx - 20);
  const end = swingIdx;
  const startPrice = klines[start].c;

  let maxMove = 0;
  let direction = 'none';

  for (let i = start + 1; i < end; i++) {
    const move = ((klines[i].c - startPrice) / startPrice) * 100;
    if (Math.abs(move) > Math.abs(maxMove)) {
      maxMove = move;
      direction = move > 0 ? 'up' : 'down';
    }
  }

  return {
    movePct: round(maxMove, 1),
    direction,
    magnitude: round(Math.abs(maxMove), 1),
    isSignificant: Math.abs(maxMove) > 5,
    lookAhead: 20,
  };
}

// ─── Round number levels ───────────────────────
function getRoundLevels(price) {
  const levels = [];
  const magnitude = Math.pow(10, Math.floor(Math.log10(price))); // 10000 for 60k range
  const step = magnitude / 2; // $5000 steps for BTC

  for (let i = -10; i <= 10; i++) {
    const level = Math.round(price / step) * step + i * step;
    if (level > 0 && Math.abs((level - price) / price) < 0.15) {
      // Only levels within 15%
      levels.push({
        price: level,
        type: 'round',
        touches: 1,
        strength: level % (step * 2) === 0 ? 3 : 1, // $60K, $70K stronger than $57.5K
      });
    }
  }
  return levels;
}

// ─── Previous timeframe high/low ───────────────
function getPrevPeriodLevels(klines, period = 24) {
  if (klines.length < period * 2) return { high: null, low: null };

  const prevPeriod = klines.slice(-period * 2, -period);
  const currentPeriod = klines.slice(-period);

  const prevHigh = Math.max(...prevPeriod.map(k => k.h));
  const prevLow = Math.min(...prevPeriod.map(k => k.l));
  const curClose = currentPeriod[currentPeriod.length - 1]?.c || prevHigh;

  return {
    high: {
      price: prevHigh,
      label: period >= 168 ? 'WEEKLY' : period >= 24 ? 'DAILY' : 'PERIOD',
      type: 'prev_high',
      touches: 3, // previous period high always significant
      broke: curClose > prevHigh,
    },
    low: {
      price: prevLow,
      label: period >= 168 ? 'WEEKLY' : period >= 24 ? 'DAILY' : 'PERIOD',
      type: 'prev_low',
      touches: 3,
      broke: curClose < prevLow,
    },
  };
}

// ─── Volume profile nodes (high volume = strong level) ──
function getVolumeNodes(klines, topN = 3) {
  const closes = klines.map(k => k.c);
  const vols = klines.map(k => k.v);
  const price = closes[closes.length - 1];

  if (klines.length < 20) return [];

  // Simple POC detection using volume-weighted bins
  const minP = Math.min(...closes.map(k => k));
  const maxP = Math.max(...closes.map(k => k));
  const bins = 30;
  const binW = (maxP - minP) / bins;
  const volBins = new Array(bins).fill(0);

  for (const k of klines) {
    const bin = Math.min(bins - 1, Math.max(0, Math.floor((k.c - minP) / binW)));
    volBins[bin] += k.v;
  }

  const nodes = [];
  for (let i = 0; i < bins; i++) {
    nodes.push({
      price: minP + (i + 0.5) * binW,
      volume: volBins[i],
      distPct: ((minP + (i + 0.5) * binW - price) / price) * 100,
    });
  }

  nodes.sort((a, b) => b.volume - a.volume);
  return nodes
    .filter(n => Math.abs(n.distPct) < 10) // within 10% of current price
    .slice(0, topN)
    .map(n => ({
      price: round(n.price, 2),
      type: 'volume_node',
      touches: Math.round(n.volume / Math.max(...nodes.map(x => x.volume)) * 5),
      label: 'VOL',
    }));
}

// ─── MAIN: Compute edge levels ─────────────────
export async function computeEdgeLevels(symbol) {
  try {
    // Fetch multi-timeframe data
    const [k1h, k4h, k1d] = await Promise.all([
      fetchKlines(symbol, '1h', 200),
      fetchKlines(symbol, '4h', 200),
      fetchKlines(symbol, '1d', 200),
    ]);

    const currentPrice = k1h[k1h.length - 1].c;
    if (!currentPrice) return { error: 'No data' };

    const allLevels = [];

    // 1. Swing points from 4h (medium-term)
    const swings4h = findSwingPoints(k4h, 4);
    for (const s of swings4h) {
      if (Math.abs((s.price - currentPrice) / currentPrice) > 0.12) continue; // within 12%
      const impact = calculateImpact(k4h, s.idx, 15);
      allLevels.push({
        price: s.price,
        type: s.type === 'high' ? 'resistance' : 'support',
        source: '4H',
        touches: 1,
        impact: impact.magnitude,
        impactDir: impact.direction,
        lastMove: `${impact.direction === 'up' ? '+' : ''}${impact.movePct}%`,
        isSignificant: impact.isSignificant,
      });
    }

    // 2. Swing points from 1d (long-term, higher weight)
    const swings1d = findSwingPoints(k1d, 3);
    for (const s of swings1d) {
      if (Math.abs((s.price - currentPrice) / currentPrice) > 0.15) continue;
      const impact = calculateImpact(k1d, s.idx, 10);
      allLevels.push({
        price: s.price,
        type: s.type === 'high' ? 'resistance' : 'support',
        source: '1D',
        touches: 2, // daily swings carry more weight
        impact: impact.magnitude,
        impactDir: impact.direction,
        lastMove: `${impact.direction === 'up' ? '+' : ''}${impact.movePct}%`,
        isSignificant: impact.isSignificant,
      });
    }

    // 3. Round number levels
    const rounds = getRoundLevels(currentPrice);
    for (const r of rounds) {
      allLevels.push({
        price: r.price,
        type: r.price > currentPrice ? 'resistance' : 'support',
        source: 'ROUND',
        touches: r.strength,
        impact: 3,
        impactDir: 'neutral',
        lastMove: 'N/A',
        isSignificant: r.strength >= 3,
      });
    }

    // 4. Volume nodes
    const volNodes = getVolumeNodes(k1h, 3);
    for (const v of volNodes) {
      allLevels.push({
        price: v.price,
        type: v.price > currentPrice ? 'resistance' : 'support',
        source: 'VOL',
        touches: v.touches,
        impact: 3,
        impactDir: 'neutral',
        lastMove: 'N/A',
        isSignificant: v.touches >= 3,
      });
    }

    // 5. Previous week high/low from 4h data
    const weekly = getPrevPeriodLevels(k4h, 42); // ~1 week of 4h candles
    if (weekly.high) {
      allLevels.push({
        price: weekly.high.price,
        type: 'resistance',
        source: 'W-PREV',
        touches: 4,
        impact: 5,
        impactDir: 'neutral',
        lastMove: weekly.high.broke ? 'BROKEN ↑' : 'RESPECTED',
        isSignificant: true,
      });
    }
    if (weekly.low) {
      allLevels.push({
        price: weekly.low.price,
        type: 'support',
        source: 'W-PREV',
        touches: 4,
        impact: 5,
        impactDir: 'neutral',
        lastMove: weekly.low.broke ? 'BROKEN ↓' : 'RESPECTED',
        isSignificant: true,
      });
    }

    // ─── Merge nearby levels and score them ────
    allLevels.sort((a, b) => a.price - b.price);

    function mergeLevels(levels) {
      const result = [];
      if (!levels.length) return result;

      let cur = [levels[0]];
      for (let i = 1; i < levels.length; i++) {
        if (Math.abs(levels[i].price - cur[0].price) / cur[0].price < 0.008) {
          cur.push(levels[i]);
        } else {
          const avgPrice = cur.reduce((s, l) => s + l.price, 0) / cur.length;
          const totalTouches = cur.reduce((s, l) => s + l.touches, 0);
          const maxImpact = Math.max(...cur.map(l => l.impact));
          const sources = [...new Set(cur.map(l => l.source))].join('+');
          const best = cur.sort((a, b) => b.touches - a.touches)[0];
          result.push({
            price: round(avgPrice, 2),
            type: best.type,
            source: sources,
            touches: totalTouches,
            impact: round(maxImpact, 1),
            impactDir: best.impactDir,
            lastMove: best.lastMove,
            strength: Math.min(5, Math.round((totalTouches + maxImpact / 5) / 2)),
          });
          cur = [levels[i]];
        }
      }
      // Last group
      if (cur.length) {
        const avgPrice = cur.reduce((s, l) => s + l.price, 0) / cur.length;
        const totalTouches = cur.reduce((s, l) => s + l.touches, 0);
        const maxImpact = Math.max(...cur.map(l => l.impact));
        const sources = [...new Set(cur.map(l => l.source))].join('+');
        const lastBest = cur.sort((a, b) => b.touches - a.touches)[0];
        result.push({
          price: round(avgPrice, 2),
          type: lastBest.type,
          source: sources,
          touches: totalTouches,
          impact: round(maxImpact, 1),
          impactDir: lastBest.impactDir,
          lastMove: lastBest.lastMove,
          strength: Math.min(5, Math.round((totalTouches + maxImpact / 5) / 2)),
        });
      }
      return result;
    }

    const merged = mergeLevels(allLevels);

    // Split into supports and resistances
    const supports = merged
      .filter(l => l.price < currentPrice)
      .sort((a, b) => b.price - a.price);
    const resistances = merged
      .filter(l => l.price > currentPrice)
      .sort((a, b) => a.price - b.price);

    // Only keep "edge" levels: strength >= 3 OR impact >= 5
    const edgeSupports = supports.filter(l => l.strength >= 3 || l.impact >= 5).slice(0, 4);
    const edgeResistances = resistances.filter(l => l.strength >= 3 || l.impact >= 5).slice(0, 4);

    const strengthStars = (s) => {
      if (s >= 5) return '★★★★★';
      if (s >= 4) return '★★★★☆';
      if (s >= 3) return '★★★☆☆';
      if (s >= 2) return '★★☆☆☆';
      return '★☆☆☆☆';
    };

    return {
      symbol,
      price: round(currentPrice, 2),
      supports: edgeSupports.map(s => ({
        ...s,
        distPct: round((s.price - currentPrice) / currentPrice * 100, 2),
        stars: strengthStars(s.strength),
      })),
      resistances: edgeResistances.map(r => ({
        ...r,
        distPct: round((r.price - currentPrice) / currentPrice * 100, 2),
        stars: strengthStars(r.strength),
      })),
      timestamp: new Date().toISOString(),
    };
  } catch (e) {
    return { error: e.message };
  }
}

// ─── Format edge level alert for Telegram ─────
export function formatEdgeAlert(levels, currentPrice) {
  if (levels.error) return `Edge: ${levels.error}`;
  if (!levels.supports.length && !levels.resistances.length) return null;

  let msg = `🎯 KEY LEVELS | ${levels.symbol.replace('USDT', '')} $${levels.price}\n`;
  msg += `───────────────────────────\n`;

  if (levels.supports.length) {
    msg += `SUPPORTS (major fall zones):\n`;
    for (const s of levels.supports.slice(0, 3)) {
      msg += `  $${s.price} [${s.distPct}%] ${s.stars} ${s.source}\n`;
      msg += `  Last: ${s.lastMove} | Impact: ${s.impact}%\n`;
    }
  }

  if (levels.resistances.length) {
    msg += `RESISTANCES (rally launch zones):\n`;
    for (const r of levels.resistances.slice(0, 3)) {
      msg += `  $${r.price} [+${r.distPct}%] ${r.stars} ${r.source}\n`;
      msg += `  Last: ${r.lastMove} | Impact: ${r.impact}%\n`;
    }
  }

  msg += `───────────────────────────`;
  return msg;
}

// ─── Check if price is near an edge level ─────
export function isNearEdgeLevel(levels, thresholdPct = 1.5) {
  const near = [];
  for (const s of levels.supports || []) {
    if (Math.abs(s.distPct) < thresholdPct) near.push({ ...s, side: 'support' });
  }
  for (const r of levels.resistances || []) {
    if (Math.abs(r.distPct) < thresholdPct) near.push({ ...r, side: 'resistance' });
  }
  return near.sort((a, b) => Math.abs(a.distPct) - Math.abs(b.distPct));
}

function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

// CLI test
async function main() {
  const symbol = process.argv[2] || 'BTCUSDT';
  const levels = await computeEdgeLevels(symbol);
  console.log(JSON.stringify(levels, null, 2));
  const alert = formatEdgeAlert(levels);
  if (alert) console.log('\n' + alert);
}

const isMain = process.argv[1]?.includes('edge-levels.mjs');
if (isMain) main();
