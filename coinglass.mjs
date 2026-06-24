// ──────────────────────────────────────────────────
//  COINGLASS.MJS — Futures Market Data
//  Binance Futures API (free, no auth)
//  Open Interest + Funding Rate
// ──────────────────────────────────────────────────

const FAPI = 'https://fapi.binance.com/fapi/v1';

// ─── Open Interest ──────────────────────────────
export async function fetchOpenInterest(symbol = 'BTCUSDT') {
  try {
    const res = await fetch(`${FAPI}/openInterest?symbol=${symbol}`);
    const data = await res.json();
    if (!data.openInterest) return null;

    return {
      symbol,
      oi: parseFloat(data.openInterest),
      time: data.time,
      signal: 'OI fetched',
    };
  } catch (_) { return null; }
}

// ─── Funding Rate ───────────────────────────────
export async function fetchFundingRate(symbol = 'BTCUSDT') {
  try {
    const res = await fetch(`${FAPI}/fundingRate?symbol=${symbol}&limit=3`);
    const data = await res.json();
    if (!Array.isArray(data) || !data.length) return null;

    const latest = parseFloat(data[0].fundingRate);
    const prev = data.length > 1 ? parseFloat(data[1].fundingRate) : null;
    const ratePct = latest * 100; // as percentage per 8h

    let signal;
    if (ratePct > 0.05) signal = 'OVERHEATED — high funding';
    else if (ratePct > 0.01) signal = 'BULLISH — positive funding';
    else if (ratePct > -0.01) signal = 'NEUTRAL / NORMAL';
    else if (ratePct > -0.05) signal = 'BEARISH — negative funding';
    else signal = 'CAPITULATION — extreme negative funding';

    return {
      symbol,
      rate: round(ratePct, 4),
      rateRaw: latest,
      prevRate: prev ? round(prev * 100, 4) : null,
      signal,
      score: getFundingScore(ratePct),
    };
  } catch (_) { return null; }
}

function getFundingScore(ratePct) {
  if (ratePct > 0.05) return -20;  // overheated longs = bearish
  if (ratePct > 0.01) return -5;
  if (ratePct > -0.01) return 0;
  if (ratePct > -0.05) return 5;
  return 20;  // extreme negative = bullish
}

// ─── Fetch all on-chain data ────────────────────
export async function fetchAllOnchain(symbol = 'BTCUSDT') {
  const [oi, funding] = await Promise.allSettled([
    fetchOpenInterest(symbol),
    fetchFundingRate(symbol),
  ]);

  const oiData = oi.status === 'fulfilled' ? oi.value : null;
  const fundData = funding.status === 'fulfilled' ? funding.value : null;

  return {
    openInterest: oiData,
    fundingRate: fundData,
    compositeScore: fundData?.score || 0,
  };
}

// ─── Format for alerts ──────────────────────────
export function formatOnchainAlert(data) {
  if (!data) return '';
  let msg = '';
  if (data.openInterest?.oi) msg += `OI:${round(data.openInterest.oi / 1000, 1)}K `;
  if (data.fundingRate?.rate !== undefined) msg += `Fund:${data.fundingRate.rate}% `;
  msg += `| Score:${data.compositeScore > 0 ? '+' : ''}${data.compositeScore}`;
  return msg;
}

function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

export { getFundingScore };

// CLI test
const isMain = process.argv[1]?.includes('coinglass.mjs');
if (isMain) {
  const result = await fetchAllOnchain('BTCUSDT');
  console.log(JSON.stringify(result, null, 2));
  console.log(formatOnchainAlert(result));
}
