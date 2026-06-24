// ─── Performance Metrics ───

export function computeMetrics(trades, equityCurve) {
  const n = trades.length;
  if (n === 0) return null;

  const wins = trades.filter(t => t.pnlPct > 0);
  const losses = trades.filter(t => t.pnlPct <= 0);
  const winRate = round((wins.length / n) * 100, 1);

  const grossProfit = wins.reduce((s, t) => s + t.pnlPct, 0);
  const grossLoss = Math.abs(losses.reduce((s, t) => s + t.pnlPct, 0));
  const netProfit = grossProfit - grossLoss;
  const profitFactor = grossLoss > 0 ? round(grossProfit / grossLoss, 2) : grossProfit > 0 ? 999 : 0;

  const avgWin = wins.length > 0 ? round(grossProfit / wins.length, 2) : 0;
  const avgLoss = losses.length > 0 ? round(grossLoss / losses.length, 2) : 0;
  const expectancy = round(netProfit / n, 2);

  // Consecutive wins/losses
  let curW = 0, curL = 0, maxW = 0, maxL = 0;
  for (const t of trades) {
    if (t.pnlPct > 0) { curW++; curL = 0; if (curW > maxW) maxW = curW; }
    else { curL++; curW = 0; if (curL > maxL) maxL = curL; }
  }

  // Sharpe & Sortino from equity curve
  let sharpe = 0, sortino = 0, maxDD = 0, maxDDpct = 0;
  if (equityCurve && equityCurve.length > 1) {
    const returns = [];
    for (let i = 1; i < equityCurve.length; i++) {
      returns.push((equityCurve[i] - equityCurve[i - 1]) / equityCurve[i - 1]);
    }
    const meanRet = returns.reduce((s, v) => s + v, 0) / returns.length;
    const std = Math.sqrt(returns.reduce((s, v) => s + (v - meanRet) ** 2, 0) / returns.length);
    sharpe = std > 0 ? round((meanRet / std) * Math.sqrt(365), 2) : 0;

    const downside = returns.filter(r => r < 0);
    const ddStd = downside.length > 0
      ? Math.sqrt(downside.reduce((s, v) => s + v * v, 0) / downside.length)
      : 0.0001;
    sortino = ddStd > 0 ? round((meanRet / ddStd) * Math.sqrt(365), 2) : 0;

    // Max drawdown
    let peak = equityCurve[0];
    for (const v of equityCurve) {
      if (v > peak) peak = v;
      const dd = (peak - v) / peak * 100;
      if (dd > maxDDpct) { maxDDpct = dd; maxDD = peak - v; }
    }
  }

  return {
    totalTrades: n,
    winningTrades: wins.length,
    losingTrades: losses.length,
    winRate,
    profitFactor,
    netProfit: round(netProfit, 2),
    avgWin,
    avgLoss,
    expectancy,
    sharpeRatio: sharpe,
    sortinoRatio: sortino,
    maxDrawdownPct: round(maxDDpct, 2),
    maxDrawdown: round(maxDD, 2),
    consecutiveWins: maxW,
    consecutiveLosses: maxL,
  };
}

function round(n, d = 2) {
  return Math.round(n * 10 ** d) / 10 ** d;
}
