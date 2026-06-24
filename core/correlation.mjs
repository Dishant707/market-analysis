// ─── Correlation Module ───
// Pearson, Spearman, rolling window, threshold matrix, regime detection

/**
 * Pearson correlation coefficient between two arrays
 */
export function pearson(a, b) {
  const n = a.length;
  if (n < 3) return 0;
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, da2 = 0, db2 = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    num += da * db;
    da2 += da * da;
    db2 += db * db;
  }
  const den = Math.sqrt(da2 * db2);
  return den > 0 ? num / den : 0;
}

/**
 * Spearman rank correlation
 */
export function spearman(a, b) {
  const n = a.length;
  if (n < 3) return 0;
  const rank = arr => {
    const sorted = arr.map((v, i) => ({ v, i })).sort((x, y) => x.v - y.v);
    const ranks = new Array(n);
    sorted.forEach((x, i) => { ranks[x.i] = i + 1; });
    return ranks;
  };
  return pearson(rank(a), rank(b));
}

/**
 * Returns correlation matrix for N assets
 * @param {number[][]} prices - array of price arrays [asset1[], asset2[], ...]
 * @param {string} method - 'pearson' | 'spearman'
 * @param {number} [rollingWindow] - if set, compute rolling correlations
 */
export function correlationMatrix(prices, method = 'pearson', rollingWindow) {
  const n = prices.length;
  const names = prices.map((_, i) => String(i));
  const matrix = Array.from({ length: n }, () => Array(n).fill(0));

  if (rollingWindow) {
    // Return rolling correlation over time
    const len = prices[0].length;
    const result = [];
    for (let t = rollingWindow; t < len; t++) {
      const slice = prices.map(p => p.slice(t - rollingWindow, t));
      const mat = Array.from({ length: n }, () => Array(n).fill(0));
      for (let i = 0; i < n; i++) {
        mat[i][i] = 1;
        for (let j = i + 1; j < n; j++) {
          const corr = method === 'spearman' ? spearman(slice[i], slice[j]) : pearson(slice[i], slice[j]);
          mat[i][j] = mat[j][i] = round(corr, 4);
        }
      }
      result.push({ index: t, matrix: mat });
    }
    return { method, rolling: result, window: rollingWindow };
  }

  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const corr = method === 'spearman' ? spearman(prices[i], prices[j]) : pearson(prices[i], prices[j]);
      matrix[i][j] = matrix[j][i] = round(corr, 4);
    }
  }
  return { method, matrix };
}

/**
 * Threshold matrix: only show correlations above |threshold|
 */
export function thresholdMatrix(matrix, threshold = 0.5) {
  const n = matrix.length;
  const result = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      result[i][j] = Math.abs(matrix[i][j]) >= threshold ? matrix[i][j] : 0;
    }
  }
  return result;
}

/**
 * Detect correlation regime based on average pairwise correlation
 * Regimes: 'high', 'moderate', 'low', 'negative'
 */
export function detectRegime(matrix) {
  const n = matrix.length;
  let sum = 0, count = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      sum += matrix[i][j];
      count++;
    }
  }
  const avg = count > 0 ? sum / count : 0;
  if (avg > 0.6) return { regime: 'HIGH', avgCorr: round(avg, 3), interpretation: 'Assets move together — crisis/risk-on mode' };
  if (avg > 0.3) return { regime: 'MODERATE', avgCorr: round(avg, 3), interpretation: 'Normal correlation regime' };
  if (avg > 0) return { regime: 'LOW', avgCorr: round(avg, 3), interpretation: 'Assets mostly independent — good for diversification' };
  return { regime: 'NEGATIVE', avgCorr: round(avg, 3), interpretation: 'Assets move opposite — strong hedge environment' };
}

/**
 * Find highly correlated pairs above threshold
 */
export function findCorrelatedPairs(names, matrix, threshold = 0.7) {
  const pairs = [];
  const n = names.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(matrix[i][j]) >= threshold) {
        pairs.push({
          asset1: names[i], asset2: names[j],
          correlation: matrix[i][j],
          strength: Math.abs(matrix[i][j]) >= 0.85 ? 'very strong' : 'strong',
        });
      }
    }
  }
  return pairs.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
}

function round(n, d = 4) {
  return Math.round(n * 10 ** d) / 10 ** d;
}
