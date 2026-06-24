// ─── Correlation Strategy Backtester ───

import { pearson, correlationMatrix, detectRegime, findCorrelatedPairs } from '../core/correlation.mjs';
import { calcSpread, zscore, halfLife, engleGranger, pairsTradeBacktest } from '../core/cointegration.mjs';
import { computeMetrics } from './metrics.mjs';

/**
 * Full correlation strategy analysis for a set of assets
 */
export async function analyzeCorrelationStrategies(assets, {
  correlationMethod = 'pearson',
  correlationThreshold = 0.7,
  pairsEntryZ = 2.0,
  pairsExitZ = 0.5,
  pairsStopZ = 3.5,
  useKalman = false,
  minCointegratedPValue = 0.05,
} = {}) {
  const names = assets.map(a => a.name);
  const prices = assets.map(a => a.prices);
  const returns = prices.map(p => {
    const r = [];
    for (let i = 1; i < p.length; i++) r.push((p[i] - p[i - 1]) / p[i - 1] * 100);
    return r;
  });

  // 1. Correlation analysis on price returns
  const priceCorr = correlationMatrix(prices, correlationMethod);
  const returnCorr = correlationMatrix(returns, correlationMethod);
  const regime = detectRegime(returnCorr.matrix);
  const correlatedPairs = findCorrelatedPairs(names, returnCorr.matrix, correlationThreshold);

  // 2. Rolling correlation
  const rollingCorr = correlationMatrix(returns, correlationMethod, 30);

  // 3. For each highly correlated pair, test cointegration and run backtest
  const pairResults = [];
  for (const pair of correlatedPairs.slice(0, 10)) {
    const i1 = names.indexOf(pair.asset1);
    const i2 = names.indexOf(pair.asset2);
    const p1 = prices[i1];
    const p2 = prices[i2];

    // Cointegration test
    const coint = engleGranger(p1, p2);

    // Pairs trade backtest
    const bt = pairsTradeBacktest(p1, p2, {
      entryZ: pairsEntryZ,
      exitZ: pairsExitZ,
      stopZ: pairsStopZ,
      useKalman,
    });

    // Performance metrics
    const eq = buildEquityCurve(bt.trades, 10000);
    const metrics = computeMetrics(bt.trades, eq);

    pairResults.push({
      pair: `${pair.asset1}/${pair.asset2}`,
      correlation: pair.correlation,
      cointegration: coint,
      trades: bt.trades.slice(-20), // last 20 trades
      metrics,
      zscores: bt.zscores.slice(-50),
      spreads: bt.spreads.slice(-50),
      tradeCount: bt.trades.length,
    });
  }

  // 4. Best strategy selection
  const best = pairResults
    .filter(p => p.metrics && p.metrics.sharpeRatio > 0.5 && p.cointegration.isCointegrated)
    .sort((a, b) => b.metrics.sharpeRatio - a.metrics.sharpeRatio);

  return {
    assets: names,
    dataPoints: prices[0]?.length || 0,
    correlation: {
      price: priceCorr,
      returns: returnCorr,
      rolling: rollingCorr.rolling?.slice(-10), // last 10 windows
      regime,
      correlatedPairs: correlatedPairs.slice(0, 10),
    },
    pairs: pairResults,
    bestStrategy: best[0] || null,
    summary: best.length > 0
      ? `Best pair: ${best[0].pair} (Sharpe ${best[0].metrics.sharpeRatio}, ${best[0].metrics.winRate}% win rate, ${best[0].tradeCount} trades)`
      : 'No profitable cointegrated pairs found',
  };
}

function buildEquityCurve(trades, initialCapital) {
  const eq = [initialCapital];
  let capital = initialCapital;
  for (const t of trades) {
    capital *= (1 + t.pnlPct / 100);
    eq.push(capital);
  }
  return eq;
}
