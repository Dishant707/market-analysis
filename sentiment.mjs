// ──────────────────────────────────────────────────
//  SENTIMENT.MJS — Market Sentiment Analysis
//  Fear & Greed Index (free, no auth)
//  + Social sentiment scoring
// ──────────────────────────────────────────────────

// ─── Fear & Greed Index ────────────────────────
export async function fetchFearGreed() {
  try {
    const res = await fetch('https://api.alternative.me/fng/?limit=7');
    const data = await res.json();
    if (!data.data) return null;

    const current = data.data[0];
    const prev = data.data[1];

    return {
      value: parseInt(current.value),
      classification: current.value_classification,
      change: prev ? parseInt(current.value) - parseInt(prev.value) : 0,
      history: data.data.map(d => ({
        value: parseInt(d.value),
        classification: d.value_classification,
        timestamp: parseInt(d.timestamp),
      })),
      signal: getFgSignal(parseInt(current.value)),
    };
  } catch (_) {
    return null;
  }
}

function getFgSignal(value) {
  if (value <= 20) return 'EXTREME FEAR — historically bullish';
  if (value <= 40) return 'FEAR — accumulation zone';
  if (value <= 60) return 'NEUTRAL';
  if (value <= 80) return 'GREED — distribution zone';
  return 'EXTREME GREED — historically bearish';
}

// ─── Quantify for model input ──────────────────
export function getFgScore(value) {
  // Extreme fear = bullish contrarian signal (+30)
  // Extreme greed = bearish contrarian signal (-30)
  if (value <= 20) return 30;
  if (value <= 40) return 15;
  if (value <= 60) return 0;
  if (value <= 80) return -15;
  return -30;
}

// ─── Format for alerts ─────────────────────────
export function formatFgAlert(fg) {
  if (!fg) return '';
  const emoji = fg.value <= 30 ? '🟢' : fg.value <= 60 ? '🟡' : '🔴';
  return `${emoji} Fear & Greed: ${fg.value}/100 [${fg.classification}] → ${fg.signal}`;
}

function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

// CLI test
const isMain = process.argv[1]?.includes('sentiment.mjs');
if (isMain) {
  const fg = await fetchFearGreed();
  console.log(formatFgAlert(fg));
  console.log('Score:', getFgScore(fg?.value || 50));
}
