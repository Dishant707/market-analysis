// ──────────────────────────────────────────────────
//  DELTA.MJS — Delta Exchange Options Analysis
//  Fetches options chain, IV, OI, Greeks
//  Generates option selling signals → Telegram
// ──────────────────────────────────────────────────

const BASE = 'https://api.india.delta.exchange';
const ASSETS = { BTC: 'BTC', ETH: 'ETH', SOL: 'SOL' }; // underlying assets

// ─── Public API (no auth needed) ──────────────
const headers = { 'Accept': 'application/json', 'User-Agent': 'market-analysis' };

async function deltaFetch(path) {
  const res = await fetch(`${BASE}${path}`, { headers });
  if (!res.ok) throw new Error(`Delta ${res.status}`);
  const data = await res.json();
  if (!data.success) throw new Error(data.error?.code || 'delta api error');
  return data.result;
}

// ─── Get options chain ────────────────────────
export async function getOptionChain(asset = 'BTC') {
  // Get all live call/put options for the asset
  const products = await deltaFetch(
    `/v2/products?contract_types=call_options,put_options&states=live`
  );

  return products
    .filter(p => p.symbol.startsWith('C-') || p.symbol.startsWith('P-'))
    .filter(p => {
      // Match asset: C-BTC-xxxxx-xxxxxx or P-BTC-xxxxx-xxxxxx
      const parts = p.symbol.split('-');
      return parts.length >= 4 && parts[1] === asset;
    })
    .map(p => ({
      id: p.id,
      symbol: p.symbol,
      type: p.symbol.startsWith('C-') ? 'CALL' : 'PUT',
      strike: parseFloat(p.symbol.split('-')[2]),
      expiry: p.symbol.split('-').slice(3).join('-'),
      settlementTime: p.settlement_time,
      tickSize: p.tick_size,
      contractValue: p.contract_value,
      state: p.state,
    }));
}

// ─── Get option tickers (IV, OI, Greeks) ──────
export async function getOptionTickers(symbols) {
  if (!symbols.length) return [];
  // Max 10 symbols per request
  const batches = [];
  for (let i = 0; i < symbols.length; i += 10) {
    batches.push(symbols.slice(i, i + 10));
  }

  const results = [];
  for (const batch of batches) {
    try {
      const tickers = await deltaFetch(`/v2/tickers/${batch.join(',')}`);
      const tickerArr = Array.isArray(tickers) ? tickers : [tickers];
      results.push(...tickerArr.map(t => ({
        symbol: t.symbol,
        productId: t.product_id,
        spotPrice: parseFloat(t.spot_price),
        markPrice: parseFloat(t.mark_price),
        markVol: parseFloat(t.mark_vol || 0),
        oi: parseFloat(t.oi || 0),
        oiValueUsd: parseFloat(t.oi_value_usd || 0),
        change24h: parseFloat(t.ltp_change_24h || 0),
        volume: parseFloat(t.volume || 0),
        high: t.high, low: t.low,
        // Option-specific
        ivBid: t.quotes?.bid_iv ? parseFloat(t.quotes.bid_iv) : null,
        ivAsk: t.quotes?.ask_iv ? parseFloat(t.quotes.ask_iv) : null,
        bidSize: t.quotes?.bid_size ? parseFloat(t.quotes.bid_size) : 0,
        askSize: t.quotes?.ask_size ? parseFloat(t.quotes.ask_size) : 0,
        bestBid: t.quotes?.best_bid ? parseFloat(t.quotes.best_bid) : null,
        bestAsk: t.quotes?.best_ask ? parseFloat(t.quotes.best_ask) : null,
        // Greeks (delta/gamma/theta/vega/rho)
        delta: t.greeks?.delta ? parseFloat(t.greeks.delta) : null,
        gamma: t.greeks?.gamma ? parseFloat(t.greeks.gamma) : null,
        theta: t.greeks?.theta ? parseFloat(t.greeks.theta) : null,
        vega: t.greeks?.vega ? parseFloat(t.greeks.vega) : null,
        rho: t.greeks?.rho ? parseFloat(t.greeks.rho) : null,
        strikePrice: parseFloat(t.strike_price || 0),
        timestamp: t.timestamp,
      })));
    } catch (_) {}
  }
  return results;
}

// ─── Get spot ticker ──────────────────────────
export async function getSpotTicker(asset = 'BTC') {
  const ticker = await deltaFetch(`/v2/tickers/${asset}USD`);
  const t = Array.isArray(ticker) ? ticker[0] : ticker;
  return {
    symbol: t.symbol,
    price: parseFloat(t.close || t.mark_price || 0),
    markPrice: parseFloat(t.mark_price || 0),
    change24h: parseFloat(t.ltp_change_24h || 0),
    oi: parseFloat(t.oi || 0),
    oiValueUsd: parseFloat(t.oi_value_usd || 0),
    volume: parseFloat(t.volume || 0),
    high: t.high, low: t.low,
    fundingRate: t.funding_rate || null,
  };
}

// ─── Analyze Options for Selling ──────────────
export async function analyzeOptionSelling(asset = 'BTC', spotPrice = null) {
  // 1. Get spot price if not provided
  if (!spotPrice) {
    try {
      const spot = await getSpotTicker(asset);
      spotPrice = spot.price;
    } catch (_) {
      return { error: 'Cannot fetch spot price' };
    }
  }

  // 2. Get option chain
  const chain = await getOptionChain(asset);
  if (!chain.length) return { error: 'No options found' };

  // Sort by strike
  chain.sort((a, b) => a.strike - b.strike);

  // Get tickers (IV, OI, Greeks) for all strikes
  const symbols = chain.map(c => c.symbol);
  const tickers = await getOptionTickers(symbols);

  // Merge ticker data into chain
  const tickerMap = {};
  for (const t of tickers) tickerMap[t.symbol] = t;

  // 3. Build strike ladder
  const calls = [];
  const puts = [];

  for (const c of chain) {
    const t = tickerMap[c.symbol];
    if (!t) continue;

    const moneyness = ((t.strikePrice || c.strike) - spotPrice) / spotPrice * 100;
    const iv = t.ivAsk || t.ivBid || 0;
    const premium = t.bestAsk || t.markPrice || 0;
    const delta = Math.abs(t.delta || 0);
    const theta = t.theta || 0;
    const oi = t.oi || 0;

    const entry = {
      symbol: c.symbol,
      strike: t.strikePrice || c.strike,
      expiry: c.expiry,
      type: c.type,
      moneyness: round(moneyness, 1),
      iv: round(iv * 100, 1),       // as percentage
      premium: round(premium, 2),
      premiumPct: round(premium / (t.strikePrice || 1) * 100, 2),
      delta: round(delta, 3),
      theta: round(theta, 4),
      oi: round(oi, 0),
      oiValue: round(t.oiValueUsd, 0),
      bidSize: t.bidSize,
      askSize: t.askSize,
      volume: round(t.volume, 0),
      change24h: round(t.change24h, 1),
    };

    if (c.type === 'CALL') calls.push(entry);
    else puts.push(entry);
  }

  // 4. IV Analysis
  const allIVs = [...calls, ...puts].filter(x => x.iv > 0).map(x => x.iv);
  const avgIV = allIVs.length ? allIVs.reduce((a, b) => a + b, 0) / allIVs.length : 0;
  const maxIV = allIVs.length ? Math.max(...allIVs) : 0;
  const atmIV = [...calls, ...puts]
    .filter(x => Math.abs(x.moneyness) < 2)
    .reduce((s, x) => s + x.iv, 0) /
    [...calls, ...puts].filter(x => Math.abs(x.moneyness) < 2).length || avgIV;

  // IV skew: OTM puts IV vs OTM calls IV
  const otmPutIVs = puts.filter(p => p.moneyness < -5).map(p => p.iv);
  const otmCallIVs = calls.filter(c => c.moneyness > 5).map(c => c.iv);
  const avgPutIV = otmPutIVs.length ? otmPutIVs.reduce((a, b) => a + b, 0) / otmPutIVs.length : 0;
  const avgCallIV = otmCallIVs.length ? otmCallIVs.reduce((a, b) => a + b, 0) / otmCallIVs.length : 0;
  const ivSkew = round(avgPutIV - avgCallIV, 1);
  const skewDirection = ivSkew > 5 ? 'PUT SKEW (puts expensive — sell puts)' :
                        ivSkew < -5 ? 'CALL SKEW (calls expensive — sell calls)' : 'NEUTRAL';

  // 5. OI Analysis — Max Pain
  let maxPainStrike = spotPrice;
  let minPain = Infinity;
  const strikeRange = [...new Set([...calls, ...puts].map(x => x.strike))].sort((a, b) => a - b);

  for (const k of strikeRange) {
    let pain = 0;
    for (const c of calls) {
      if (k > c.strike) pain += Math.abs(k - c.strike) * (c.oi || 1);
    }
    for (const p of puts) {
      if (k < p.strike) pain += Math.abs(p.strike - k) * (p.oi || 1);
    }
    if (pain < minPain) {
      minPain = pain;
      maxPainStrike = k;
    }
  }

  // 6. Generate recommendations
  const recommendations = [];

  // Find best OTM PUTs to sell (15-25% OTM)
  const sellPuts = puts
    .filter(p => p.moneyness < -15 && p.moneyness > -30 && p.iv > 30)
    .sort((a, b) => b.iv - a.iv)
    .slice(0, 3);

  for (const p of sellPuts) {
    recommendations.push({
      type: 'SELL PUT',
      symbol: p.symbol,
      strike: p.strike,
      moneyness: p.moneyness,
      iv: p.iv,
      premium: p.premium,
      premiumPct: p.premiumPct,
      delta: p.delta,
      oi: p.oi,
      reason: `High IV (${p.iv}%), ${Math.abs(p.moneyness)}% OTM, Delta ${p.delta}`,
      risk: p.delta > 0.1 ? 'MEDIUM' : 'LOW',
    });
  }

  // Find best OTM CALLs to sell
  const sellCalls = calls
    .filter(c => c.moneyness > 15 && c.moneyness < 30 && c.iv > 30)
    .sort((a, b) => b.iv - a.iv)
    .slice(0, 3);

  for (const c of sellCalls) {
    recommendations.push({
      type: 'SELL CALL',
      symbol: c.symbol,
      strike: c.strike,
      moneyness: c.moneyness,
      iv: c.iv,
      premium: c.premium,
      premiumPct: c.premiumPct,
      delta: c.delta,
      oi: c.oi,
      reason: `High IV (${c.iv}%), ${c.moneyness}% OTM, Delta ${c.delta}`,
      risk: c.delta > 0.1 ? 'MEDIUM' : 'LOW',
    });
  }

  // Sort by best risk/reward (lowest delta, highest premium)
  recommendations.sort((a, b) => b.premiumPct / Math.max(b.delta, 0.01) - a.premiumPct / Math.max(a.delta, 0.01));

  return {
    asset,
    spotPrice: round(spotPrice, 2),
    timestamp: new Date().toISOString(),
    totalOptions: chain.length,
    iv: {
      atmIV: round(atmIV, 1),
      avgIV: round(avgIV, 1),
      maxIV: round(maxIV, 1),
      ivSkew,
      skewDirection,
    },
    oi: {
      totalOIValue: round([...calls, ...puts].reduce((s, x) => s + x.oiValue, 0), 0),
      maxPainStrike: round(maxPainStrike, 0),
      maxPainDist: round((maxPainStrike - spotPrice) / spotPrice * 100, 1),
      putCallOI: round(
        puts.reduce((s, p) => s + p.oiValue, 0) /
        Math.max(calls.reduce((s, c) => s + c.oiValue, 0), 1), 1
      ),
    },
    recommendations: recommendations.slice(0, 5),
    calls: calls.filter(c => Math.abs(c.moneyness) < 30),
    puts: puts.filter(p => Math.abs(p.moneyness) < 30),
  };
}

// ─── Format for Telegram ──────────────────────
export function formatDeltaAlert(analysis) {
  if (analysis.error) return `Delta: ${analysis.error}`;

  const { asset, spotPrice, iv, oi, recommendations } = analysis;

  let msg = `\n📐 DELTA OPTIONS | ${asset} $${spotPrice}\n`;
  msg += `───────────────────────────\n`;
  msg += `IV: ATM ${iv.atmIV}% | Avg ${iv.avgIV}% | ${iv.skewDirection}\n`;
  msg += `Max Pain: $${oi.maxPainStrike} [${oi.maxPainDist > 0 ? '+' : ''}${oi.maxPainDist}%]\n`;

  if (recommendations.length) {
    msg += `\nTOP SELLS:\n`;
    for (const r of recommendations.slice(0, 3)) {
      msg += `  ${r.type} $${r.strike} | IV:${r.iv}% | Prem:${r.premiumPct}% | D${r.delta} | ${r.risk}\n`;
      msg += `  Exp:${r.symbol.split('-').slice(3).join('-')} | OI:${r.oi}\n`;
    }
  }

  msg += `───────────────────────────`;
  return msg;
}

function round(n, d = 2) {
  return Math.round(n * 10 ** d) / 10 ** d;
}

// ─── CLI Mode ─────────────────────────────────
async function main() {
  const asset = process.argv[2] || 'BTC';
  const analysis = await analyzeOptionSelling(asset);
  console.log(JSON.stringify(analysis, null, 2));
  console.log('\n' + formatDeltaAlert(analysis));
}

const isMain = process.argv[1]?.includes('delta.mjs');
if (isMain) main();
