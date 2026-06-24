// ─── Cointegration Module ───
// Engle-Granger test, spread calculation, half-life mean reversion, Kalman filter

/**
 * Linear regression: returns { slope, intercept }
 */
function ols(y, x) {
  const n = Math.min(y.length, x.length);
  if (n < 3) return { slope: 0, intercept: 0 };
  const mx = x.reduce((s, v) => s + v, 0) / n;
  const my = y.reduce((s, v) => s + v, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my);
    den += (x[i] - mx) ** 2;
  }
  const slope = den > 0 ? num / den : 0;
  return { slope, intercept: my - slope * mx };
}

/**
 * Compute the spread between two assets: y - slope * x
 * Positive spread = y is expensive relative to x, short y / long x
 */
export function calcSpread(prices1, prices2) {
  const n = Math.min(prices1.length, prices2.length);
  const { slope } = ols(prices1, prices2);
  const spread = [];
  for (let i = 0; i < n; i++) {
    spread.push(prices1[i] - slope * prices2[i]);
  }
  return { spread, hedgeRatio: slope };
}

/**
 * Z-score normalize a series
 */
export function zscore(series) {
  const n = series.length;
  if (n < 3) return series.map(() => 0);
  const mean = series.reduce((s, v) => s + v, 0) / n;
  const std = Math.sqrt(series.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  return std > 0 ? series.map(v => (v - mean) / std) : series.map(() => 0);
}

/**
 * Half-life of mean reversion using OLS on spread(t) - spread(t-1) vs spread(t-1)
 * Lower half-life = faster mean reversion
 */
export function halfLife(spread) {
  const n = spread.length;
  if (n < 30) return Infinity;
  const y = []; // spread(t) - spread(t-1)
  const x = []; // spread(t-1)
  for (let i = 1; i < n; i++) {
    y.push(spread[i] - spread[i - 1]);
    x.push(spread[i - 1]);
  }
  const { slope } = ols(y, x);
  if (slope >= 0) return Infinity; // not mean-reverting
  return Math.log(2) / -slope;
}

/**
 * Engle-Granger cointegration test (simplified ADF on spread)
 * Returns z-score of the ADF test statistic — more negative = more cointegrated
 */
export function engleGranger(prices1, prices2) {
  const n = Math.min(prices1.length, prices2.length);
  if (n < 30) return { isCointegrated: false, adfStat: 0, pValue: 1, halfLifeDays: Infinity };

  const { spread } = calcSpread(prices1, prices2);
  const hl = halfLife(spread);

  // ADF test on spread (simplified Dickey-Fuller regression)
  const dy = [];
  const ly = [];
  for (let i = 1; i < spread.length; i++) {
    dy.push(spread[i] - spread[i - 1]);
    ly.push(spread[i - 1]);
  }
  const m = dy.length;
  if (m < 10) return { isCointegrated: false, adfStat: 0, pValue: 1, halfLifeDays: Infinity };

  // Include lagged differences for serial correlation
  const X = [];
  const Y = [];
  for (let i = 2; i < m; i++) {
    Y.push(dy[i]);
    X.push([ly[i], dy[i - 1]]);
  }
  const yLen = Y.length;
  if (yLen < 5) return { isCointegrated: false, adfStat: 0, pValue: 1, halfLifeDays: Infinity };

  // OLS: Y = beta0 * ly + beta1 * dy_lag1
  const lyReg = X.map(x => x[0]);
  const { slope } = ols(Y, lyReg);

  // ADF statistic = slope / std_error (simplified)
  const predicted = lyReg.map(v => v * slope);
  const residuals = Y.map((v, i) => v - predicted[i]);
  const residualStd = Math.sqrt(residuals.reduce((s, v) => s + v * v, 0) / (yLen - 1));
  const se = residualStd / Math.sqrt(lyReg.reduce((s, v) => s + v * v, 0));
  const adfStat = se > 0 ? slope / se : 0;

  // Critical values: -3.43 (1%), -2.86 (5%), -2.57 (10%)
  const pValue = adfStat < -3.43 ? 0.01 : adfStat < -2.86 ? 0.05 : adfStat < -2.57 ? 0.10 : 0.50;
  const isCointegrated = adfStat < -2.86;

  return {
    isCointegrated,
    adfStat: round(adfStat, 4),
    pValue,
    halfLifeDays: round(hl, 1),
    interpretation: isCointegrated
      ? `Cointegrated (ADF ${adfStat.toFixed(2)}). Half-life ${hl.toFixed(0)} periods. Mean-reverting spread.`
      : `Not cointegrated (ADF ${adfStat.toFixed(2)}). No stationary spread.`,
  };
}

/**
 * Kalman filter for dynamic hedge ratio
 * Tracks time-varying beta between two assets
 */
export function kalmanHedgeRatio(y, x) {
  const n = Math.min(y.length, x.length);
  if (n < 10) return { ratios: [], predictions: [] };

  // State: [beta, intercept] — we track beta only for simplicity
  let beta = 1.0;
  let P = 100; // error covariance
  const R = 0.01; // measurement noise
  const Q = 0.001; // process noise

  const ratios = [];
  const predictions = [];

  for (let i = 0; i < n; i++) {
    // Predict
    const betaPred = beta;
    const PPred = P + Q;

    // Update
    const F = x[i]; // measurement = x value
    const K = PPred * F / (F * PPred * F + R); // Kalman gain
    const innovation = y[i] - F * betaPred;
    beta = betaPred + K * innovation;
    P = (1 - K * F) * PPred;

    ratios.push(beta);
    predictions.push(F * beta);
  }

  return { ratios, latestBeta: round(ratios[ratios.length - 1], 4) };
}

/**
 * Backtest a pairs trading strategy on the spread
 */
export function pairsTradeBacktest(prices1, prices2, {
  entryZ = 2.0,
  exitZ = 0.5,
  stopZ = 3.5,
  useKalman = false,
} = {}) {
  const n = Math.min(prices1.length, prices2.length);
  if (n < 50) return { trades: [], metrics: null };

  const hedgeRatios = [];
  const spreads = [];
  const zscores = [];

  // Use either OLS (fixed) or Kalman (dynamic) hedge ratio
  if (useKalman) {
    const { ratios } = kalmanHedgeRatio(prices1, prices2);
    for (let i = 0; i < n; i++) {
      const hr = ratios[i] || 1;
      hedgeRatios.push(hr);
      spreads.push(prices1[i] - hr * prices2[i]);
    }
  } else {
    const { spread, hedgeRatio } = calcSpread(prices1, prices2);
    for (let i = 0; i < n; i++) {
      hedgeRatios.push(hedgeRatio);
      spreads.push(spread[i]);
    }
  }

  // Z-score with expanding window
  for (let i = 0; i < n; i++) {
    const window = spreads.slice(Math.max(0, i - 60), i + 1);
    const z = zscore(window);
    zscores.push(i < 30 ? 0 : z[z.length - 1]);
  }

  // Simulate trades
  const trades = [];
  let position = 0; // 1 = long spread (long P1, short P2), -1 = short spread
  let entryIdx = 0;
  let entrySpread = 0;

  for (let i = 30; i < n; i++) {
    const z = zscores[i];
    const p1 = prices1[i];
    const p2 = prices2[i];
    const hr = hedgeRatios[i];

    // Entry signals
    if (position === 0) {
      if (z > entryZ) {
        // Spread too high — short spread (short P1, long P2)
        position = -1;
        entryIdx = i;
        entrySpread = spreads[i];
      } else if (z < -entryZ) {
        // Spread too low — long spread (long P1, short P2)
        position = 1;
        entryIdx = i;
        entrySpread = spreads[i];
      }
    }

    // Exit signals
    if (position !== 0) {
      const exit = (
        (position === 1 && z >= -exitZ) || // long exited when z rises back
        (position === -1 && z <= exitZ) ||  // short exited when z falls back
        Math.abs(z) > stopZ                 // stop loss
      );

      if (exit || i === n - 1) {
        const pnlPct = (position === 1)
          ? (spreads[i] - entrySpread) / entrySpread * 100
          : (entrySpread - spreads[i]) / entrySpread * 100;

        trades.push({
          entryIdx,
          exitIdx: i,
          direction: position === 1 ? 'long' : 'short',
          entrySpread: round(entrySpread, 4),
          exitSpread: round(spreads[i], 4),
          pnlPct: round(pnlPct, 2),
          entryZ: round(zscores[entryIdx], 2),
          exitZ: round(z, 2),
          duration: i - entryIdx,
          exitReason: Math.abs(z) > stopZ ? 'stop' : 'signal',
        });
        position = 0;
      }
    }
  }

  return { trades, zscores, spreads: spreads.map(v => round(v, 4)) };
}

function round(n, d = 4) {
  return Math.round(n * 10 ** d) / 10 ** d;
}
