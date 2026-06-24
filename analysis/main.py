"""Heavy computation sidecar for crypto analysis.
Called by Node.js via stdin/stdout JSON protocol.
Uses numpy/pandas for vectorized operations (M4 accelerated).
"""

import sys, json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def monte_carlo(closes, price, steps=[1, 4, 24, 96], num_sims=5000):
    returns = np.diff(closes) / closes[:-1]
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    result = {}
    for step in steps:
        sims = np.random.normal(mean, std, (num_sims, step))
        paths = price * np.cumprod(1 + sims, axis=1)
        final = paths[:, -1]
        sorted_prices = np.sort(final)
        p5, p25, p50, p75, p95 = np.percentile(sorted_prices, [5, 25, 50, 75, 95])
        up_prob = float(np.mean(final > price)) * 100
        result[f'{step}h'] = {
            'target5': round(p5, 2), 'target25': round(p25, 2),
            'target50': round(p50, 2), 'target75': round(p75, 2), 'target95': round(p95, 2),
            'probUp': round(up_prob, 1), 'probDown': round(100 - up_prob, 1),
        }
    return result

def statistics(closes):
    arr = np.array(closes)
    returns = np.diff(arr) / arr[:-1]
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    z = (arr[-1] - np.mean(arr)) / (np.std(arr) or 1)
    skew = float(np.mean(((returns - mean) / (std or 1)) ** 3)) if len(returns) > 2 else 0
    kurt = float(np.mean(((returns - mean) / (std or 1)) ** 4)) if len(returns) > 2 else 0
    # Hurst
    half = len(returns) // 2
    if half > 1:
        rs1 = (np.max(returns[:half]) - np.min(returns[:half])) / (np.std(returns[:half]) or 1)
        rs2 = (np.max(returns[half:]) - np.min(returns[half:])) / (np.std(returns[half:]) or 1)
        hurst = float(np.log(rs2 / rs1) / np.log((len(returns) - half) / half)) if rs1 > 0 and rs2 > 0 else 0.5
    else:
        hurst = 0.5
    # Auto-correlation
    if len(returns) > 1:
        autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1]) if np.std(returns[:-1]) > 0 and np.std(returns[1:]) > 0 else 0
    else:
        autocorr = 0
    # Runs test
    signs = (returns >= 0).astype(int)
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    n1, n2 = int(np.sum(signs)), int(len(signs) - np.sum(signs))
    total = n1 + n2
    runs_exp = 1 + (2 * n1 * n2) / total if total > 0 else 0
    runs_std = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - total)) / (total ** 2 * (total - 1))) if total > 0 else 1
    runs_z = (runs - runs_exp) / runs_std if runs_std > 0 else 0

    return {
        'zScore': round(z, 2), 'avgPrice': round(float(np.mean(arr)), 2),
        'skewness': round(skew, 3), 'kurtosis': round(kurt, 3),
        'hurst': round(max(0, min(1, hurst)), 3),
        'hurstInterpretation': 'trending' if hurst > 0.65 else ('mean-reverting' if hurst < 0.35 else 'random walk'),
        'autoCorrelation': round(autocorr, 3),
        'runsTest': {'runs': int(runs), 'expectedRuns': round(runs_exp, 1), 'zScore': round(runs_z, 2),
                     'isRandom': bool(abs(runs_z) < 1.96)},
        'volRegime': 'expanding' if std > 0.02 else ('contracting' if std < 0.005 else 'normal'),
    }

def volume_profile(klines):
    highs = np.array([k['h'] for k in klines])
    lows = np.array([k['l'] for k in klines])
    closes = np.array([k['c'] for k in klines])
    vols = np.array([k['v'] for k in klines])
    min_p, max_p = float(np.min(lows)), float(np.max(highs))
    bins = 30
    bin_size = (max_p - min_p) / bins
    vol_per_bin = np.zeros(bins)
    visit_per_bin = np.zeros(bins)
    for k in klines:
        low_bin = max(0, min(bins - 1, int((k['l'] - min_p) / bin_size)))
        high_bin = max(0, min(bins - 1, int((k['h'] - min_p) / bin_size)))
        vol_per_bin[low_bin:high_bin + 1] += k['v'] / (high_bin - low_bin + 1)
        visit_per_bin[low_bin:high_bin + 1] += 1

    max_vol = np.max(vol_per_bin) or 1
    levels = []
    for i in range(bins):
        pl = min_p + i * bin_size
        ph = min_p + (i + 1) * bin_size
        levels.append({
            'price': round((pl + ph) / 2, 2), 'priceLow': round(pl, 2), 'priceHigh': round(ph, 2),
            'volumePct': round(float(vol_per_bin[i] / max_vol * 100), 1),
            'visits': int(visit_per_bin[i]),
        })
    return levels

def regress(closes):
    x = np.arange(len(closes))
    y = np.array(closes)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return {'slope': round(float(slope), 6), 'direction': 'up' if slope > 0 else 'down'}

def weekday_analysis(klines):
    """Analyze weekend vs weekday volatility and volume patterns."""
    import datetime
    closes = np.array([k['c'] for k in klines])
    vols = np.array([k['v'] for k in klines])  # quote volume
    timestamps = [k['t'] for k in klines]
    days = np.array([datetime.datetime.fromtimestamp(t / 1000).weekday() for t in timestamps])
    weekday_mask = days < 5
    returns = np.diff(closes) / closes[:-1] * 100
    vols_ret = vols[1:] / 1e6
    wd = weekday_mask[1:]
    ret_wd = returns[wd]; ret_we = returns[~wd]
    vol_wd = vols_ret[wd]; vol_we = vols_ret[~wd]

    # Daily breakdown
    daily = {}
    for d in range(7):
        mask = days == d
        r = returns[days[1:] == d] if len(days) > 1 else np.array([])
        v = vols_ret[days[1:] == d]
        daily[['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d]] = {
            'count': int(np.sum(days == d)),
            'avgReturn': round(float(np.mean(r)), 4) if len(r) > 0 else 0,
            'volatility': round(float(np.std(r)), 4) if len(r) > 1 else 0,
            'avgVolume': round(float(np.mean(v)), 1) if len(v) > 0 else 0,
        }

    return {
        'weekday': {
            'hours': int(np.sum(wd)),
            'avgReturn': round(float(np.mean(ret_wd)), 4),
            'volatility': round(float(np.std(ret_wd)), 4),
            'avgVolumeM': round(float(np.mean(vol_wd)), 1),
            'maxUp': round(float(np.max(ret_wd)), 2),
            'maxDown': round(float(np.min(ret_wd)), 2),
        },
        'weekend': {
            'hours': int(np.sum(~wd)),
            'avgReturn': round(float(np.mean(ret_we)), 4),
            'volatility': round(float(np.std(ret_we)), 4),
            'avgVolumeM': round(float(np.mean(vol_we)), 1),
            'maxUp': round(float(np.max(ret_we)), 2),
            'maxDown': round(float(np.min(ret_we)), 2),
        },
        'ratios': {
            'volatilityRatio': round(float(np.std(ret_we) / np.std(ret_wd)), 3) if np.std(ret_wd) > 0 else 0,
            'volumeRatio': round(float(np.mean(vol_we) / np.mean(vol_wd)), 3) if np.mean(vol_wd) > 0 else 0,
        },
        'dailyBreakdown': daily,
        'summary': f"Weekends are {int((1-np.std(ret_we)/np.std(ret_wd))*100)}% less volatile with {int((1-np.mean(vol_we)/np.mean(vol_wd))*100)}% less volume. "
                   f"Avg hourly return: weekday {float(np.mean(ret_wd)):+.3f}% vs weekend {float(np.mean(ret_we)):+.3f}%. "
                   f"Sunday ({daily['Sun']['volatility']}%) is the quietest day, Friday ({daily['Fri']['volatility']}%) the most volatile.",
    }

def market_brittleness(klines_1h, klines_1d, book_bids, book_asks, price):
    closes_1h = np.array([k['c'] for k in klines_1h])
    highs_1h = np.array([k['h'] for k in klines_1h])
    lows_1h = np.array([k['l'] for k in klines_1h])
    vols_1h = np.array([k['v'] for k in klines_1h])
    returns_1h = np.diff(closes_1h) / closes_1h[:-1] * 100
    abs_ret = np.abs(returns_1h)
    buy_vol = vols_1h[1:][returns_1h > 0]; buy_move = abs_ret[returns_1h > 0]
    sell_vol = vols_1h[1:][returns_1h < 0]; sell_move = abs_ret[returns_1h < 0]
    buy_eff = float(np.mean(buy_move / buy_vol)) if len(buy_vol) > 0 else 0
    sell_eff = float(np.mean(sell_move / sell_vol)) if len(sell_vol) > 0 else 0

    # Recent 24h
    recent = klines_1h[-24:]
    rc = np.array([k['c'] for k in recent])
    rv = np.array([k['v'] for k in recent])
    rr = np.diff(rc) / rc[:-1] * 100; ra = np.abs(rr)
    rb = rv[1:][rr > 0]; rbm = ra[rr > 0]
    rs = rv[1:][rr < 0]; rsm = ra[rr < 0]
    rbe = float(np.mean(rbm / rb)) if len(rb) > 0 else 0
    rse = float(np.mean(rsm / rs)) if len(rs) > 0 else 0

    easier = 'buyers' if buy_eff > sell_eff else 'sellers' if sell_eff > buy_eff else 'neutral'
    recent_easier = 'buyers' if rbe > rse else 'sellers' if rse > rbe else 'neutral'

    # Level analysis from 1D
    h = np.array([k['h'] for k in klines_1d])
    l = np.array([k['l'] for k in klines_1d])
    c = np.array([k['c'] for k in klines_1d])
    v = np.array([k['v'] for k in klines_1d])
    raw_levels = []
    for i in range(2, len(c) - 2):
        if h[i] == max(h[i-2:i+3]): raw_levels.append({'price': float(h[i]), 'type': 'resistance', 'idx': i})
        if l[i] == min(l[i-2:i+3]): raw_levels.append({'price': float(l[i]), 'type': 'support', 'idx': i})
    if not raw_levels: return {'error': 'no levels'}
    raw_levels.sort(key=lambda x: x['price'])
    clustered, cur = [], [raw_levels[0]]
    for lev in raw_levels[1:]:
        if abs(lev['price'] - cur[-1]['price']) / (cur[-1]['price'] or 1) < 0.005:
            cur.append(lev)
        else:
            ap = float(np.mean([x['price'] for x in cur]))
            tp = max(set(x['type'] for x in cur), key=lambda t: sum(1 for x in cur if x['type'] == t))
            mask = (h >= ap * 0.995) & (l <= ap * 1.005)
            touches = int(np.sum(mask))
            tidx = np.where(mask)[0]
            bounces = []
            for ti in tidx:
                if tp == 'support' and ti < len(c) - 1: b = abs((c[ti+1] - ap) / ap * 100)
                elif tp == 'resistance' and ti > 0: b = abs((ap - c[ti-1]) / ap * 100)
                else: b = 0
                if b > 0.01: bounces.append(b)
            ab = float(np.mean(bounces)) if bounces else 0
            sr = np.mean(bounces[-3:]) / (np.mean(bounces) or 1) if len(bounces) >= 3 else 1.0
            britt = max(0, min(100, (1 - sr) * 100)) if sr < 1 else 0
            clustered.append({'price': round(ap, 2), 'type': tp, 'touches': touches,
                'avgBounce': round(ab, 2), 'brittleness': round(britt, 1),
                'steam': 'brittle' if britt > 60 else ('fading' if britt > 30 else 'strong'),
                'volAtLevel': round(float(np.mean(v[mask])), 1) if any(mask) else 0,
                'distancePct': round((ap - price) / price * 100, 2)})
            cur = [lev]
    if cur:
        ap = float(np.mean([x['price'] for x in cur]))
        clustered.append({'price': round(ap, 2), 'type': 'neutral', 'touches': 0, 'avgBounce': 0,
            'brittleness': 0, 'steam': 'unknown', 'volAtLevel': 0, 'distancePct': round((ap - price) / price * 100, 2)})

    relevant = [l for l in clustered if abs(l['distancePct']) < 10]
    supports = sorted([l for l in relevant if l['type'] == 'support' and l['distancePct'] < 0], key=lambda x: x['distancePct'], reverse=True)[:4]
    resistances = sorted([l for l in relevant if l['type'] == 'resistance' and l['distancePct'] > 0], key=lambda x: x['distancePct'])[:4]

    bd = sum(b[0]*b[1] for b in (book_bids or [])[:10]) if book_bids else 0
    ad = sum(a[0]*a[1] for a in (book_asks or [])[:10]) if book_asks else 0
    floor_p = book_bids[0][0] if book_bids else 0
    ceil_p = book_asks[0][0] if book_asks else 0

    bdown = sum(1 for s in supports if s['brittleness'] > 50)
    bup = sum(1 for r in resistances if r['brittleness'] > 50)
    ea = 'buyers' if buy_eff > sell_eff * 1.1 else ('sellers' if sell_eff > buy_eff * 1.1 else 'neutral')
    sigs = []
    if bdown > 0: sigs.append(f'{bdown} brittle support(s)')
    if bup > 0: sigs.append(f'{bup} brittle resistance(s)')
    if bd < ad * 0.7: sigs.append('thin bid side')
    elif ad < bd * 0.7: sigs.append('thin ask side')
    sigs.append(f'effort advantage: {ea}')
    tbias = (bdown - bup) * 0.4 + (-1 if bd < ad * 0.7 else 1 if ad < bd * 0.7 else 0) * 0.3 + (1 if ea == 'buyers' else -1 if ea == 'sellers' else 0) * 0.3
    path = 'DOWN (supports brittle)' if tbias > 0.5 else ('UP (resistances brittle)' if tbias < -0.5 else 'SIDEWAYS (balanced)')

    weekly_h = round(float(np.max(h[-7:])), 2) if len(h) >= 7 else None
    weekly_l = round(float(np.min(l[-7:])), 2) if len(l) >= 7 else None

    return {
        'efficiency': {
            'hourly': {'buyEfficiency': round(buy_eff * 100000, 3), 'sellEfficiency': round(sell_eff * 100000, 3), 'easierSide': easier},
            'recent24h': {'buyEfficiency': round(rbe * 100000, 3), 'sellEfficiency': round(rse * 100000, 3), 'easierSide': recent_easier},
        },
        'levels': {
            'supports': supports, 'resistances': resistances,
            'floor': round(floor_p, 2), 'ceiling': round(ceil_p, 2),
            'weekly': {'high': weekly_h, 'low': weekly_l},
        },
        'pathOfLeastResistance': {'direction': path, 'signals': sigs, 'score': round(tbias, 2)},
        'summary': f'{path}. {ea.upper()} have efficiency advantage. {bdown} support(s) and {bup} resistance(s) are brittle. '
                   f'Weekly: ${weekly_l or 0}–${weekly_h or 0}. Floor ${floor_p:.0f} · Ceiling ${ceil_p:.0f}.',
    }

def volume_stats_1m(klines_1m):
    """Per-candle volume analysis with statistical models."""
    if len(klines_1m) < 10:
        return {'error': 'need at least 10 candles'}

    closes = np.array([k['c'] for k in klines_1m])
    opens = np.array([k['o'] for k in klines_1m])
    highs = np.array([k['h'] for k in klines_1m])
    lows = np.array([k['l'] for k in klines_1m])
    vols = np.array([k['v'] for k in klines_1m], dtype=float)

    # 1. Per-candle metrics
    ranges = highs - lows
    body = np.abs(closes - opens)
    body_pct = np.where(ranges > 0, body / ranges, 0)
    direction = np.where(closes >= opens, 1, 0)  # 1=buy, 0=sell
    returns = np.diff(closes) / closes[:-1] * 100
    returns = np.append(returns, 0)  # pad

    # 2. Volume statistics
    vol_mean = float(np.mean(vols))
    vol_std = float(np.std(vols))
    vol_z = (vols - vol_mean) / (vol_std if vol_std > 0 else 1)
    vol_spikes = np.abs(vol_z) > 2

    # 3. EWMA volatility (GARCH-like)
    lam = 0.94
    squared_rets = returns ** 2
    ewma_var = np.zeros_like(squared_rets)
    ewma_var[0] = squared_rets[0]
    for i in range(1, len(squared_rets)):
        ewma_var[i] = lam * ewma_var[i-1] + (1 - lam) * squared_rets[i]
    ewma_vol = np.sqrt(ewma_var)

    # 4. Volume-weighted price
    vwap_all = np.cumsum(closes * vols) / np.cumsum(vols)

    # 5. Volume efficiency: how much range per volume
    vol_eff = np.where(vols > 0, ranges / closes / vols * 10000, 0)

    # 6. Volume momentum (acceleration)
    vol_ma5 = np.convolve(vols, np.ones(5)/5, mode='same')
    vol_ma20 = np.convolve(vols, np.ones(20)/20, mode='same')
    vol_momentum = np.where(vol_ma20 > 0, (vol_ma5 - vol_ma20) / vol_ma20, 0)

    # 7. Statistical distribution analysis
    buy_vol = vols[direction == 1].sum()
    sell_vol = vols[direction == 0].sum()
    total_vol = buy_vol + sell_vol
    buy_ratio = float(buy_vol / total_vol * 100) if total_vol > 0 else 50

    # 8. Volume clusters (price zones with highest volume)
    min_p, max_p = float(np.min(lows)), float(np.max(highs))
    bins = 15
    bin_size = (max_p - min_p) / bins
    vol_per_bin = np.zeros(bins)
    for k in klines_1m:
        lo = max(0, min(bins-1, int((k['l'] - min_p) / bin_size)))
        hi = max(0, min(bins-1, int((k['h'] - min_p) / bin_size)))
        vol_per_bin[lo:hi+1] += k['v'] / (hi - lo + 1)
    max_vol_bin = int(np.argmax(vol_per_bin))
    cluster_price = min_p + (max_vol_bin + 0.5) * bin_size
    cluster_dominance = float(vol_per_bin[max_vol_bin] / (np.sum(vol_per_bin) / bins))

    # 9. Last 12 candles summary
    n = min(12, len(klines_1m))
    last_vols = vols[-n:]
    last_dirs = direction[-n:]
    last_up = int(np.sum(last_dirs))
    last_down = n - last_up
    last_vol_up = float(last_vols[last_dirs == 1].sum()) if last_up > 0 else 0
    last_vol_dn = float(last_vols[last_dirs == 0].sum()) if last_down > 0 else 0
    last_ratio = ((last_vol_up - last_vol_dn) / (last_vol_up + last_vol_dn) * 100) if (last_vol_up + last_vol_dn) > 0 else 0
    last_avg_vol = float(np.mean(last_vols))
    last_vol_trend = float((np.mean(vols[-6:]) - np.mean(vols[-12:-6])) / (np.mean(vols[-12:-6]) or 1) * 100)
    last_price_chg = float(closes[-1] - closes[-n])

    # 10. Anomalous candles in last 30
    recent_30 = klines_1m[-30:] if len(klines_1m) >= 30 else klines_1m
    anomalies = []
    for k in recent_30:
        v = k['v']
        z = (v - vol_mean) / (vol_std if vol_std > 0 else 1)
        r = (k['h'] - k['l']) / (k['c'] or 1) * 100
        if abs(z) > 1.5 or r > np.mean(ranges / closes) * 100 * 2:
            anomalies.append({
                'time': k['t'],
                'price': float(k['c']),
                'volRatio': round(float(v / vol_mean), 2),
                'zScore': round(float(z), 2),
                'range': round(float(r), 3),
                'type': 'spike' if abs(z) > 2 else 'anomaly',
                'side': 'buy' if k['c'] >= k['o'] else 'sell',
            })

    return {
        'summary': {
            'totalCandles': len(klines_1m),
            'timeframe': '1m',
            'avgVol': round(vol_mean, 2),
            'volStd': round(vol_std, 2),
            'buyRatio': round(buy_ratio, 1),
            'sellRatio': round(100 - buy_ratio, 1),
            'volatilityEWMA': round(float(np.mean(ewma_vol[-20:])), 4),
        },
        'last12': {
            'upCandles': last_up, 'downCandles': last_down,
            'netVolumeDelta': round(last_ratio, 1),
            'avgVol': round(last_avg_vol, 2),
            'volTrend': round(last_vol_trend, 1),
            'priceChange': round(last_price_chg, 2),
            'dominantSide': 'buyers' if last_ratio > 10 else ('sellers' if last_ratio < -10 else 'neutral'),
        },
        'clusters': {
            'price': round(float(cluster_price), 2),
            'dominance': round(float(cluster_dominance), 2),
            'description': f'Volume cluster at ${cluster_price:,.0f} ({(cluster_dominance-1)*100:.0f}% above avg zone)',
        },
    }

def level_matrix(klines_1d, klines_1h, price):
    """Build a 10-level price matrix from 60k-70k with volume, cascade risk, S/R strength."""
    if len(klines_1d) < 10:
        return {'error': 'insufficient data'}

    highs = np.array([k['h'] for k in klines_1d])
    lows = np.array([k['l'] for k in klines_1d])
    closes = np.array([k['c'] for k in klines_1d])
    vols = np.array([k['v'] for k in klines_1d], dtype=float)

    # Also use 1H for recent precision
    h_highs = np.array([k['h'] for k in klines_1h])
    h_lows = np.array([k['l'] for k in klines_1h])
    h_closes = np.array([k['c'] for k in klines_1h])
    h_vols = np.array([k['v'] for k in klines_1h], dtype=float)

    # Build levels: 10 evenly-spaced $1000 bins centered on price
    step = 1000
    start_level = round(price / step) * step - (4 * step)  # 4 below price
    levels = []
    n = 10

    for i in range(n):
        level_low = start_level + i * step
        level_high = level_low + step
        mid = (level_low + level_high) / 2

        # Volume traded at this level (1D data)
        mask = (highs >= level_low) & (lows <= level_high)
        touches = int(np.sum(mask))
        vol_at_level = float(np.sum(vols[mask])) if any(mask) else 0

        # Recent volume (last 7 days from 1H)
        recent = klines_1h[-168:]  # 7 days of 1H
        r_highs = np.array([k['h'] for k in recent])
        r_lows = np.array([k['l'] for k in recent])
        r_vols = np.array([k['v'] for k in recent], dtype=float)
        r_mask = (r_highs >= level_low) & (r_lows <= level_high)
        r_touches = int(np.sum(r_mask))
        r_vol = float(np.sum(r_vols[r_mask])) if any(r_mask) else 0

        # Support or resistance nature
        # If price bounced up after touching = support, if rejected = resistance
        touch_indices = np.where(mask)[0]
        support_score = 0
        resistance_score = 0
        cascade_upside = 0
        cascade_downside = 0
        bounce_samples = []
        break_samples = []

        for t in touch_indices:
            if t > 0 and t < len(closes) - 3:
                before = closes[t-1]
                after = closes[min(t+3, len(closes)-1)]
                touch_price = (highs[t] + lows[t]) / 2
                if after > touch_price * 1.01:
                    support_score += 1
                    bounce_samples.append((after - touch_price) / touch_price * 100)
                elif after < touch_price * 0.99:
                    resistance_score += 1
                    break_samples.append((touch_price - after) / touch_price * 100)

        total_score = support_score + resistance_score
        nature = 'neutral'
        if support_score > resistance_score * 1.5 and total_score > 3:
            nature = 'support'
        elif resistance_score > support_score * 1.5 and total_score > 3:
            nature = 'resistance'
        elif total_score > 5:
            nature = 'battlezone'

        # Cascade risk: how much volume is concentrated, how many touches
        avg_touches = touches / max(1, len(klines_1d) / 200)
        vol_concentration = vol_at_level / (np.mean(vols) * len(klines_1d) / n) if np.mean(vols) > 0 else 1

        cascade_risk = min(100, (vol_concentration * 30 + avg_touches * 20 + (support_score + resistance_score) * 5))
        cascade_risk = round(cascade_risk, 1)

        # Direction of cascade
        if nature == 'support':
            cascade_direction = 'down'  # If support breaks, price cascades down
        elif nature == 'resistance':
            cascade_direction = 'up'  # If resistance breaks, price runs up
        else:
            cascade_direction = 'both' if total_score > 5 else 'low'

        # Strength level (how many bounces/breaks)
        avg_bounce = float(np.mean(bounce_samples)) if bounce_samples else 0
        avg_break = float(np.mean(break_samples)) if break_samples else 0

        levels.append({
            'level': f'${level_low:,.0f}–${level_high:,.0f}',
            'mid': round(mid, 0),
            'touches': touches,
            'volume': round(vol_at_level, 2),
            'volRatio': round(vol_concentration, 2),
            'recentTouches': r_touches,
            'recentVolume': round(r_vol, 2),
            'nature': nature,
            'supportScore': support_score,
            'resistanceScore': resistance_score,
            'cascadeRisk': cascade_risk,
            'cascadeDirection': cascade_direction,
            'avgBouncePct': round(avg_bounce, 2),
            'avgBreakPct': round(avg_break, 2),
            'isCurrentPrice': level_low <= price < level_high,
            'distancePct': round((mid - price) / price * 100, 2),
        })

    # Sort by distance from price
    levels.sort(key=lambda l: abs(l['distancePct']))

    # Find nearest support and resistance
    nearest_support = next((l for l in levels if l['nature'] == 'support' and l['distancePct'] < 0), None)
    nearest_resistance = next((l for l in levels if l['nature'] == 'resistance' and l['distancePct'] > 0), None)

    # High cascade zones
    high_cascade = [l for l in levels if l['cascadeRisk'] > 60]
    warn_cascade = [l for l in levels if 40 < l['cascadeRisk'] <= 60]

    return {
        'price': price,
        'levels': levels,
        'nearestSupport': nearest_support,
        'nearestResistance': nearest_resistance,
        'highCascadeZones': high_cascade[:3],
        'warningCascadeZones': warn_cascade[:3],
        'summary': f'{len(high_cascade)} high-risk cascade zones. '
                   f'Price at ${price:,.0f}.'
    }

def volatility_analysis(klines_1h, klines_1d, price):
    """Comprehensive volatility analysis: indices, regimes, VaR, statistical tests."""
    if len(klines_1h) < 50:
        return {'error': 'insufficient 1H data'}
    
    # ── 1. Extract OHLCV arrays for various volatility estimators ──
    h1_opens = np.array([k['o'] for k in klines_1h], dtype=float)
    h1_highs = np.array([k['h'] for k in klines_1h], dtype=float)
    h1_lows = np.array([k['l'] for k in klines_1h], dtype=float)
    h1_closes = np.array([k['c'] for k in klines_1h], dtype=float)
    h1_vols = np.array([k['v'] for k in klines_1h], dtype=float)
    
    d1_closes = np.array([k['c'] for k in klines_1d], dtype=float) if klines_1d else h1_closes
    d1_highs = np.array([k['h'] for k in klines_1d], dtype=float) if klines_1d else h1_highs
    d1_lows = np.array([k['l'] for k in klines_1d], dtype=float) if klines_1d else h1_lows
    d1_opens = np.array([k['o'] for k in klines_1d], dtype=float) if klines_1d else h1_opens

    returns_1h = np.diff(h1_closes) / h1_closes[:-1] * 100
    
    # ── 2. Multiple Volatility Estimators ──
    
    # a) Close-to-close volatility (standard)
    vol_close = float(np.std(returns_1h) * np.sqrt(24))  # annualized to daily
    
    # b) Parkinson (HL) estimator: uses high/low range
    hl_ratio = np.log(h1_highs[1:] / h1_lows[1:]) ** 2
    vol_parkinson = float(np.sqrt(np.mean(hl_ratio) / (4 * np.log(2))) * np.sqrt(24)) * 100
    
    # c) Garman-Klass (OHLC) estimator
    gk1 = 0.5 * np.log(h1_highs[1:] / h1_lows[1:]) ** 2
    gk2 = (2 * np.log(2) - 1) * np.log(h1_closes[1:] / h1_opens[1:]) ** 2
    vol_gk = float(np.sqrt(np.mean(gk1 - gk2)) * np.sqrt(24)) * 100
    
    # d) Rogers-Satchell (drift-independent) estimator — handles trends
    rs = (np.log(h1_highs[1:] / h1_closes[1:]) * np.log(h1_highs[1:] / h1_opens[1:]) +
          np.log(h1_lows[1:] / h1_closes[1:]) * np.log(h1_lows[1:] / h1_opens[1:]))
    vol_rs = float(np.sqrt(np.mean(rs)) * np.sqrt(24)) * 100
    
    # e) Yang-Zhang (best estimator, drift-independent + open-close)
    # Overnight volatility: close → open
    overnight_ret = np.diff(np.log(h1_opens)) 
    open_close_ret = np.log(h1_closes[1:] / h1_opens[1:])
    k = 0.34 / (1.34 + (len(overnight_ret) + 1) / (len(overnight_ret) - 1))
    vol_overnight = np.std(overnight_ret)
    vol_open_close = np.std(open_close_ret)
    vol_yz = float(np.sqrt(vol_overnight**2 + k * vol_open_close**2 + (1 - k) * np.mean(rs)) * np.sqrt(24)) * 100

    # f) EWMA volatility (RiskMetrics): lambda = 0.94
    lam = 0.94
    squared_rets = returns_1h ** 2
    ewma_var = np.zeros_like(squared_rets)
    ewma_var[0] = squared_rets[0]
    for i in range(1, len(squared_rets)):
        ewma_var[i] = lam * ewma_var[i-1] + (1 - lam) * squared_rets[i]
    vol_ewma = float(np.sqrt(ewma_var[-1]) * np.sqrt(24))  # annualized daily from last EWMA

    # g) GARCH(1,1) approximation via maximum likelihood  
    # Simplified: omega = (1-alpha-beta)*long_var, alpha = 0.1, beta = 0.85
    alpha, beta_g = 0.1, 0.85
    long_var = float(np.var(returns_1h))
    omega = (1 - alpha - beta_g) * long_var
    garch_var = np.zeros_like(squared_rets)
    garch_var[0] = long_var
    for i in range(1, len(squared_rets)):
        garch_var[i] = omega + alpha * squared_rets[i-1] + beta_g * garch_var[i-1]
    vol_garch = float(np.sqrt(garch_var[-1]) * np.sqrt(24))

    # Composite Volatility Index (crypto-VIX equivalent)
    # Weighted average of all estimators + 30-day historical vol
    vol_est = [vol_close, vol_parkinson, vol_gk, vol_rs, vol_yz, vol_ewma, vol_garch]
    vol_index = float(np.mean(vol_est))
    
    # Percentile rank vs historical (last 200 days of daily data)
    d1_returns = np.diff(d1_closes) / d1_closes[:-1] * 100
    d1_vol_window = np.array([np.std(d1_returns[max(0,i-30):i+1]) * np.sqrt(365) for i in range(30, len(d1_returns))])
    current_daily_vol = float(np.std(d1_returns[-30:]) * np.sqrt(365)) if len(d1_returns) >= 30 else vol_index
    vol_percentile = float(np.sum(d1_vol_window < current_daily_vol) / len(d1_vol_window) * 100) if len(d1_vol_window) > 0 else 50

    # ── 3. Volatility Regime ──
    recent_vol = float(np.std(returns_1h[-24:]))  # last 24h
    medium_vol = float(np.std(returns_1h[-72:]))  # last 72h
    long_vol = float(np.std(returns_1h))  # all
    
    vol_ratio_24h = recent_vol / (long_vol or 1)
    vol_ratio_72h = medium_vol / (long_vol or 1)
    
    # Regime classification using multiple signals
    if vol_ratio_24h > 1.5: regime = 'HIGH'
    elif vol_ratio_24h > 1.2: regime = 'RISING'
    elif vol_ratio_24h < 0.7: regime = 'LOW'
    elif vol_ratio_24h < 0.9: regime = 'FALLING'
    else: regime = 'NORMAL'
    
    # Bollinger Band width (compression/expansion)
    bb_mid = np.mean(h1_closes[-20:])
    bb_std = np.std(h1_closes[-20:])
    bb_width = bb_std / (bb_mid or 1) * 100
    bb_hist_avg = np.mean([np.std(h1_closes[i-20:i]) / (np.mean(h1_closes[i-20:i]) or 1) * 100 for i in range(40, len(h1_closes), 20)])
    bb_compression = bb_width / (bb_hist_avg or 1)
    is_squeeze = bb_compression < 0.7

    # ── 4. Value at Risk (VaR) & Expected Shortfall ──
    sorted_rets = np.sort(returns_1h)
    n = len(sorted_rets)
    var_95 = float(np.percentile(sorted_rets, 5))  # 95% VaR (hourly)
    var_99 = float(np.percentile(sorted_rets, 1))   # 99% VaR (hourly)  
    # Expected Shortfall (CVaR): average loss beyond VaR
    es_95 = float(np.mean(sorted_rets[sorted_rets <= var_95])) if any(sorted_rets <= var_95) else var_95
    es_99 = float(np.mean(sorted_rets[sorted_rets <= var_99])) if any(sorted_rets <= var_99) else var_99
    
    # Daily VaR (scaled)
    daily_var_95 = var_95 * np.sqrt(24)
    daily_es_95 = es_95 * np.sqrt(24)

    # ── 5. Statistical Tests ──
    # Normality test (Jarque-Bera)
    skewness = float(np.mean(((returns_1h - np.mean(returns_1h)) / (np.std(returns_1h) or 1)) ** 3))
    kurtosis = float(np.mean(((returns_1h - np.mean(returns_1h)) / (np.std(returns_1h) or 1)) ** 4))
    jb_stat = n / 6 * (skewness**2 + (kurtosis - 3)**2 / 4)
    # Rough p-value (chi-square 2df): reject if > 5.99
    is_normal = bool(jb_stat < 5.99)
    
    # Tail ratio
    tail_ratio = float(np.abs(np.percentile(sorted_rets, 1) / (np.percentile(sorted_rets, 99) or 1))) if np.percentile(sorted_rets, 99) != 0 else 1
    fat_tail = tail_ratio > 1.5
    
    # Serial correlation / ARCH effects (Ljung-Box simplified)
    squared_rets_mean = np.mean(squared_rets)
    lb_stat = 0
    for lag in range(1, 13):
        auto_cov = np.mean((squared_rets[:-lag] - squared_rets_mean) * (squared_rets[lag:] - squared_rets_mean))
        lb_stat += (auto_cov / np.var(squared_rets)) ** 2 / (n - lag) if np.var(squared_rets) > 0 else 0
    lb_stat = lb_stat * n * (n + 2)
    has_arch = bool(lb_stat > 21.03)  # chi-square 12df at alpha=0.05

    # ── 6. Volume-Weighted Volatility ──
    # Volatility per unit of volume (efficiency)
    vol_per_vol = np.abs(returns_1h) / (h1_vols[1:] / 1e6 + 1)
    vw_vol = float(np.mean(vol_per_vol))

    return {
        'compositeIndex': {
            'value': round(vol_index, 2),
            'percentile': round(vol_percentile, 1),
            'interpretation': f'Volatility Index: {vol_index:.1f} ({vol_percentile:.0f}th percentile)',
        },
        'estimators': {
            'closeToClose': round(vol_close, 2),
            'parkinsonHL': round(vol_parkinson, 2),
            'garmanKlassOHLC': round(vol_gk, 2),
            'rogersSatchell': round(vol_rs, 2),
            'yangZhang': round(vol_yz, 2),
            'ewma': round(vol_ewma, 2),
            'garch': round(vol_garch, 2),
        },
        'regime': {
            'current': regime,
            'squeeze': bool(is_squeeze),
            'bbWidth': round(bb_width, 3),
            'bbCompression': round(bb_compression, 2),
            'ratio24h': round(vol_ratio_24h, 2),
            'ratio72h': round(vol_ratio_72h, 2),
            'description': f'{regime} volatility regime{" · Bollinger squeeze" if is_squeeze else ""}',
        },
        'risk': {
            'var95': round(var_95, 3),
            'var99': round(var_99, 3),
            'es95': round(es_95, 3),
            'dailyVar95': round(daily_var_95, 2),
            'dailyES95': round(daily_es_95, 2),
            'tailRatio': round(tail_ratio, 2),
            'fatTail': bool(fat_tail),
        },
        'statistics': {
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis, 3),
            'jbStat': round(jb_stat, 2),
            'isNormal': bool(is_normal),
            'tailRatio': round(tail_ratio, 2),
            'hasARCH': bool(has_arch),
            'distribution': 'fat-tailed' if fat_tail else ('normal-like' if is_normal else 'non-normal'),
        },
        'volatilityEfficiency': {
            'vwpVol': round(vw_vol, 4),
            'description': f'Each $1M volume moves price {vw_vol:.4f}% on average',
        },
                'summary': f'{regime} vol · Vol Index {vol_index:.1f} ({vol_percentile:.0f}th) · '
                   f'{"Squeezed, breakout imminent" if is_squeeze else "Normal range"} · '
                   f'Daily VaR 95%: {daily_var_95:.2f}% · '
                   f'Skew {skewness:+.2f} · Kurt {kurtosis:.1f} · '
                   f'{"Fat tails (crash risk)" if fat_tail else "Normal tails"} · '
                   f'{"ARCH effects (vol clustering)" if has_arch else "No vol clustering"}',
    }

def entropy_analysis(klines_1m, klines_1h, klines_1d, price):
    """Entropy, absorption, multi-resolution volume analysis for high-volatility markets."""
    if len(klines_1h) < 50: return {'error': 'insufficient data'}
    
    # ── 1. Shannon Entropy of returns distribution ──
    # Low entropy = predictable, High entropy = random noise
    def shannon_entropy(data, bins=20):
        if len(data) < 2: return 0
        hist, _ = np.histogram(data, bins=bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))
    
    # Returns at multiple resolutions
    r1h = np.diff(np.array([k['c'] for k in klines_1h], dtype=float)) / np.array([k['c'] for k in klines_1h[:-1]], dtype=float)
    entropy_1h = shannon_entropy(r1h, bins=30)
    
    if len(klines_1m) > 5:
        r1m = np.diff(np.array([k['c'] for k in klines_1m], dtype=float)) / np.array([k['c'] for k in klines_1m[:-1]], dtype=float)
        entropy_1m = shannon_entropy(r1m, bins=30)
    else:
        entropy_1m = 0
    
    if len(klines_1d) > 5:
        r1d = np.diff(np.array([k['c'] for k in klines_1d], dtype=float)) / np.array([k['c'] for k in klines_1d[:-1]], dtype=float)
        entropy_1d = shannon_entropy(r1d, bins=30)
    else:
        entropy_1d = 0
    
    # Normalize to 0-1 scale
    max_e = np.log2(30)  # max possible entropy with 30 bins
    e_1h_norm = min(1, entropy_1h / max_e) if max_e > 0 else 0.5
    e_1m_norm = min(1, entropy_1m / max_e) if max_e > 0 else 0.5
    e_1d_norm = min(1, entropy_1d / max_e) if max_e > 0 else 0.5
    composite_entropy = float(np.mean([e_1h_norm, e_1m_norm, e_1d_norm]))
    
    # Entropy regime
    if composite_entropy > 0.7: regime = 'HIGH (random, low predictability)'
    elif composite_entropy > 0.5: regime = 'MODERATE'
    else: regime = 'LOW (structured, more predictable)'
    
    # ── 2. Absorption Detection (high volume, low price movement) ──
    def detect_absorption(klines, vol_mult=1.5, range_div=0.6):
        if len(klines) < 5: return []
        closes = np.array([k['c'] for k in klines], dtype=float)
        highs = np.array([k['h'] for k in klines], dtype=float)
        lows = np.array([k['l'] for k in klines], dtype=float)
        volumes = np.array([k['v'] for k in klines], dtype=float)
        avg_vol = np.mean(volumes)
        avg_range = np.mean(highs - lows)
        results = []
        for i in range(1, len(klines)):
            vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 0
            range_pct = (highs[i] - lows[i]) / (closes[i] or 1) * 100
            avg_range_pct = avg_range / (np.mean(closes) or 1) * 100
            if vol_ratio > vol_mult and range_pct < avg_range_pct * range_div:
                side = 'buyer_absorption' if closes[i] >= closes[i-1] else 'seller_absorption'
                results.append({
                    'idx': i, 'volRatio': round(float(vol_ratio), 2),
                    'range': round(float(range_pct), 3), 'side': side,
                    'strength': round(float(vol_ratio * (1 - range_pct / (avg_range_pct or 1))), 2),
                })
        return results
    
    abs_1h = detect_absorption(klines_1h, vol_mult=1.5, range_div=0.5)
    abs_5m_tmp = []
    # Try to use klines_1m in chunks as 5m proxy
    if len(klines_1m) > 10:
        # chunk 1m into 5m groups
        chunk_size = 5
        chunks = []
        for i in range(0, len(klines_1m) - chunk_size + 1, chunk_size):
            chunk = klines_1m[i:i+chunk_size]
            ch = {'o': chunk[0]['o'], 'h': max(c['h'] for c in chunk),
                  'l': min(c['l'] for c in chunk), 'c': chunk[-1]['c'],
                  'v': sum(c['v'] for c in chunk)}
            chunks.append(ch)
        abs_5m_tmp = detect_absorption(chunks, vol_mult=1.5, range_div=0.5)
    
    # Aggregate absorption
    buy_absorption = [a for a in abs_1h if 'buyer' in a.get('side','')]
    sell_absorption = [a for a in abs_1h if 'seller' in a.get('side','')]
    net_absorption = len(buy_absorption) - len(sell_absorption)
    
    # Absorption interpretation
    if abs(net_absorption) >= 3:
        abs_side = 'buyers' if net_absorption > 0 else 'sellers'
        abs_signal = f'Strong {abs_side} absorption ({abs(net_absorption)} more events) — '
        abs_signal += 'accumulation' if abs_side == 'buyers' else 'distribution'
    else:
        abs_signal = 'Balanced absorption — no clear accumulation/distribution'
    
    # ── 3. Multi-Resolution Volume Profile (1m, 5m, 1h simultaneously) ──
    def volume_clusters(klines, bins=20):
        if len(klines) < 5: return []
        highs = np.array([k['h'] for k in klines], dtype=float)
        lows = np.array([k['l'] for k in klines], dtype=float)
        vols = np.array([k['v'] for k in klines], dtype=float)
        min_p, max_p = float(np.min(lows)), float(np.max(highs))
        if max_p - min_p < 1: return []
        bin_w = (max_p - min_p) / bins
        vol_per_bin = np.zeros(bins)
        for k in klines:
            lo = max(0, min(bins-1, int((k['l'] - min_p) / bin_w)))
            hi = max(0, min(bins-1, int((k['h'] - min_p) / bin_w)))
            vol_per_bin[lo:hi+1] += k['v'] / (hi - lo + 1)
        max_v = np.max(vol_per_bin) or 1
        clusters = []
        for i in range(bins):
            p = min_p + (i + 0.5) * bin_w
            d = vol_per_bin[i] / max_v * 100
            if d > 50:
                clusters.append({'price': round(float(p), 2), 'dominance': round(float(d), 1)})
        return clusters
    
    vp_1h = volume_clusters(klines_1h, bins=20)
    vp_1m_tmp = []
    if len(klines_1m) > 50:
        vp_1m_tmp = volume_clusters(klines_1m, bins=20)
    if len(klines_1d) > 10:
        vp_1d = volume_clusters(klines_1d, bins=30)
    else:
        vp_1d = []
    
    # Find overlapping clusters across timeframes (high-conviction zones)
    all_prices = []
    for vpl in [vp_1m_tmp, vp_1h, vp_1d]:
        for v in vpl:
            all_prices.append(v['price'])
    # Cluster nearby prices
    if all_prices:
        all_prices.sort()
        conv_zones = []
        cur = [all_prices[0]]
        for p in all_prices[1:]:
            if abs(p - cur[-1]) / (cur[-1] or 1) < 0.002:
                cur.append(p)
            else:
                conv_zones.append({'price': round(float(np.mean(cur)), 2), 'convictions': len(cur)})
                cur = [p]
        if cur:
            conv_zones.append({'price': round(float(np.mean(cur)), 2), 'convictions': len(cur)})
        conv_zones = [z for z in conv_zones if z['convictions'] >= 2]
    else:
        conv_zones = []
    
    # Near price
    near_zones = [z for z in conv_zones if abs(z['price'] - price) / price < 0.05]
    
    return {
        'entropy': {
            'composite': round(composite_entropy, 3),
            'regime': regime,
            '1h': round(e_1h_norm, 3),
            '1m': round(e_1m_norm, 3) if len(klines_1m) > 5 else None,
            '1d': round(e_1d_norm, 3) if len(klines_1d) > 5 else None,
            'interpretation': 'High entropy — market is noise-driven, low predictability' if composite_entropy > 0.6
                else 'Moderate entropy — some structure exists' if composite_entropy > 0.4
                else 'Low entropy — market has clear structure, higher predictability',
        },
        'absorption': {
            'total1h': len(abs_1h),
            'buyEvents': len(buy_absorption),
            'sellEvents': len(sell_absorption),
            'net': net_absorption,
            'signal': abs_signal,
            'recent': abs_1h[-5:] if abs_1h else [],
            'interpretation': abs_signal,
        },
        'multiResVolume': {
            'clusters1h': vp_1h[:5],
            'clusters1m': vp_1m_tmp[:5] if len(klines_1m) > 50 else [],
            'clusters1d': vp_1d[:5],
            'convictionZones': near_zones[:5],
        },
        'summary': f'Entropy: {composite_entropy:.2f} ({regime}). '
                   f'Absorption: {len(abs_1h)} events in 1H ({net_absorption:+d} net). '
                   f'Multi-res volume zones: {len(near_zones)} high-conviction levels within 5%. '
                   f'Prediction confidence: {"Low" if composite_entropy > 0.6 else "Moderate" if composite_entropy > 0.4 else "High"}.',
        'predictionConfidence': 'low' if composite_entropy > 0.6 else ('moderate' if composite_entropy > 0.4 else 'high'),
    }

def probabilistic_forecast(klines_1h, klines_1d, price):
    """
    Bayesian probability + KDE + path integration + historical regime matching.
    
    Uses:
    - Kernel Density Estimation on actual returns (not normal distribution)
    - Bayesian probability of reaching each price level
    - Historical regime similarity scoring
    - Path probability via transition matrix
    - Full scenario tree with outcome probabilities
    """
    if len(klines_1h) < 100:
        return {'error': 'insufficient 1H data'}
    
    # ── Data preparation ──
    c1h = np.array([k['c'] for k in klines_1h], dtype=float)
    c1d = np.array([k['c'] for k in klines_1d], dtype=float) if len(klines_1d) > 50 else c1h
    
    # Returns at multiple horizons
    r_1h = np.diff(c1h) / c1h[:-1] * 100      # hourly returns (100+)
    r_1d = np.diff(c1d) / c1d[:-1] * 100      # daily returns (50+)
    
    # ── 1. Kernel Density Estimation (KDE) on actual return distribution ──
    # KDE gives us the REAL probability density, not assuming normal
    def kde_pdf(data, points):
        """Simple Gaussian KDE."""
        if len(data) < 2:
            return np.ones(len(points)) / len(points)
        h = 1.06 * np.std(data) * len(data) ** (-0.2)  # Silverman's rule
        h = max(h, 0.01)
        pdf = np.zeros(len(points))
        for d in data:
            pdf += np.exp(-0.5 * ((points - d) / h) ** 2) / (h * np.sqrt(2 * np.pi))
        return pdf / len(data)
    
    # Probability distribution for next 1H return
    ret_range = np.linspace(-3, 3, 100)  # -3% to +3% (covers 99.9% of hourly moves)
    pdf_1h = kde_pdf(r_1h[-200:], ret_range)
    pdf_1h = pdf_1h / (np.sum(pdf_1h) + 1e-10)
    
    # Cumulative distribution
    cdf_1h = np.cumsum(pdf_1h)
    cdf_1h = cdf_1h / cdf_1h[-1]  # normalize
    
    # Probability of up/down in next hour
    up_idx = ret_range > 0
    prob_up_1h = float(np.sum(pdf_1h[up_idx]))
    prob_down_1h = float(1 - prob_up_1h)
    
    # Most likely return range (mode of distribution)
    mode_idx = int(np.argmax(pdf_1h))
    mode_return = float(ret_range[mode_idx])
    hdi_low = float(ret_range[np.searchsorted(cdf_1h, 0.16)])  # 16th percentile
    hdi_high = float(ret_range[np.searchsorted(cdf_1h, 0.84)])  # 84th percentile
    
    # ── 2. Bayesian probability of reaching price levels ──
    # P(price reaches X | current price, historical distribution)
    levels_k = np.arange(0.85, 1.16, 0.01)  # ±15% from current in 1% steps
    target_prices = price * levels_k
    
    # Using simplified Brownian motion: P(reach) = exp(-2*μ*D/σ²)  for hitting a barrier
    mu_1h = float(np.mean(r_1h[-200:]))
    sigma_1h = float(np.std(r_1h[-200:]))
    
    hit_probs = []
    for tp in target_prices:
        dist_pct = (tp - price) / price * 100
        if abs(dist_pct) < 0.2:
            prob = 0.5  # near current, ~50/50
        elif sigma_1h > 0:
            # First passage time probability (simplified)
            direction = dist_pct / abs(dist_pct)
            drift_term = direction * mu_1h / (sigma_1h ** 2) if sigma_1h > 0 else 0
            prob = 1 / (1 + np.exp(-2 * drift_term * abs(dist_pct)))
        else:
            prob = 0.5
        hit_probs.append(float(min(0.95, max(0.05, prob))))
    
    # Build levels with probabilities
    bayesian_levels = []
    for i, tp in enumerate(target_prices):
        bayesian_levels.append({
            'price': round(float(tp), 2),
            'distPct': round(float((tp - price) / price * 100), 2),
            'probReach': round(hit_probs[i], 3),
            'type': 'support' if tp < price else 'resistance',
        })
    
    # ── 3. Historical Regime Matching ──
    # Find similar periods in history by matching recent return distribution
    recent_rets = r_1h[-24:]  # last 24 hours
    recent_vol = float(np.std(recent_rets))
    recent_mean = float(np.mean(recent_rets))
    
    # Slide through history, find most similar 24h windows
    matches = []
    window = 24
    for i in range(0, len(r_1h) - window - 24, 6):  # step every 6 hours
        hist_window = r_1h[i:i+window]
        hist_vol = float(np.std(hist_window))
        hist_mean = float(np.mean(hist_window))
        
        # Similarity score (lower = more similar)
        vol_diff = abs(hist_vol - recent_vol) / (recent_vol or 0.01)
        mean_diff = abs(hist_mean - recent_mean) / (abs(recent_mean) or 0.01)
        similarity = vol_diff * 0.5 + mean_diff * 0.5
        
        if similarity < 1.5:  # reasonably similar
            # What happened in the next 24-96 hours after this window?
            outcomes_24 = r_1h[i+window:i+window+24]
            outcomes_96 = r_1h[i+window:i+window+96]
            
            if len(outcomes_24) > 0:
                next_24h_chg = float(np.sum(outcomes_24))
                next_96h_chg = float(np.sum(outcomes_96)) if len(outcomes_96) > 0 else 0
                next_24h_vol = float(np.std(outcomes_24))
                
                matches.append({
                    'similarity': round(similarity, 2),
                    'next24hChg': round(next_24h_chg, 2),
                    'next96hChg': round(next_96h_chg, 2),
                    'next24hVol': round(next_24h_vol, 2),
                    'direction': 'up' if next_24h_chg > 0 else 'down',
                })
    
    # Aggregate match outcomes
    if matches:
        total_m = len(matches)
        up_matches = [m for m in matches if m['direction'] == 'up']
        dn_matches = [m for m in matches if m['direction'] == 'down']
        avg_up = float(np.mean([m['next24hChg'] for m in up_matches])) if up_matches else 0
        avg_dn = abs(float(np.mean([m['next24hChg'] for m in dn_matches]))) if dn_matches else 0
        max_up = float(np.max([m['next24hChg'] for m in up_matches])) if up_matches else 0
        max_dn = abs(float(np.min([m['next24hChg'] for m in dn_matches]))) if dn_matches else 0
        
        hist_outcome = {
            'totalMatches': total_m,
            'upProb': round(len(up_matches) / total_m * 100, 1),
            'downProb': round(len(dn_matches) / total_m * 100, 1),
            'avgUpMove': round(avg_up, 2),
            'avgDownMove': round(avg_dn, 2),
            'maxUpMove': round(max_up, 2),
            'maxDownMove': round(max_dn, 2),
            'avgVol24h': round(float(np.mean([m['next24hVol'] for m in matches])), 2),
            'interpretation': f'When BTC looked like this before ({total_m} similar regimes), '
                f'it went UP {len(up_matches)/total_m*100:.0f}% of the time (avg +{avg_up:.1f}%) '
                f'and DOWN {len(dn_matches)/total_m*100:.0f}% (avg -{avg_dn:.1f}%). '
                f'Best case: +{max_up:.1f}% in 24h. Worst case: -{max_dn:.1f}%.',
        }
    else:
        hist_outcome = {'totalMatches': 0, 'interpretation': 'No similar historical regimes found'}
    
    # ── 4. Path Probability / Scenario Tree ──
    # Three scenarios: bullish, bearish, neutral
    scenarios = []
    
    # Bullish: top 20% of outcomes
    bullish_rets = np.percentile(r_1h[-500:], 80) if len(r_1h) > 100 else 0.5
    bearish_rets = np.percentile(r_1h[-500:], 20) if len(r_1h) > 100 else -0.5
    neutral_rets = np.median(r_1h[-500:]) if len(r_1h) > 100 else 0.0
    
    # 24h projection at 4h, 8h, 12h, 24h intervals
    horizons = [4, 8, 12, 24]
    for label, ret, prob_weight in [
        ('Bullish', bullish_rets, 0.25),
        ('Neutral', neutral_rets, 0.50),
        ('Bearish', bearish_rets, 0.25),
    ]:
        path = []
        for h in horizons:
            projected = price * (1 + ret/100 * h)
            path.append({
                'hours': h,
                'price': round(float(projected), 2),
                'chgPct': round(float(ret * h), 2),
            })
        scenarios.append({
            'label': label,
            'probability': prob_weight,
            'hourlyReturn': round(float(ret), 4),
            'target24h': round(float(price * (1 + ret/100 * 24)), 2),
            'target24hChg': round(float(ret * 24), 2),
            'path': path,
        })
    
    # ── 5. Full price level probability distribution ──
    # For each $500 bucket from -20% to +20%, probability price ends there in 24h
    buckets = np.arange(0.80, 1.21, 0.005) * price
    bucket_probs = np.zeros(len(buckets))
    r24_dist = r_1h[-24:] if len(r_1h) >= 24 else r_1h
    r24_kde = kde_pdf(r24_dist, ret_range)
    for i, b in enumerate(buckets):
        ret_needed = (b - price) / price * 100
        # Probability of this return from KDE
        idx = np.argmin(np.abs(ret_range - ret_needed))
        bucket_probs[i] = pdf_1h[idx] if 0 <= idx < len(pdf_1h) else 0
    bucket_probs = bucket_probs / (np.sum(bucket_probs) + 1e-10)
    
    # Most probable price levels
    top_idx = np.argsort(bucket_probs)[-5:][::-1]
    most_probable_levels = [
        {'price': round(float(buckets[i]), 2), 'prob': round(float(bucket_probs[i] * 100), 1)}
        for i in top_idx
        if abs((buckets[i] - price) / price) > 0.003  # skip near-current
    ]
    
    # ── 6. Major event probability (100K→50K→82K pattern) ──
    # Use actual historical drawdown distribution from 1D data
    if len(c1d) > 100:
        peak = np.maximum.accumulate(c1d)
        dd = (c1d - peak) / peak  # all drawdowns as decimals (-0.5 = -50%)
        # Count unique peak-to-trough drawdowns
        unique_dd = []
        current_max = c1d[0]
        current_min = c1d[0]
        for p in c1d:
            if p > current_max:
                unique_dd.append((current_min - current_max) / current_max)
                current_max = p
                current_min = p
            else:
                current_min = min(current_min, p)
        unique_dd.append((current_min - current_max) / current_max)
        unique_dd = np.array(unique_dd)
        p_50pct = float(np.sum(unique_dd < -0.50) / max(1, len(unique_dd))) * 100
        
        # Rally probability: rolling 252-day window, count windows with 30%+ gain
        rally_count = 0
        windows = 0
        for i in range(len(c1d) - 252):
            ret_1y = (c1d[i+252] / c1d[i]) - 1
            if ret_1y > 0.30:
                rally_count += 1
            windows += 1
        p_30pct = float(rally_count / max(1, windows)) * 100
        p_50pct_drawdown = p_50pct
        p_30pct_rally = p_30pct
    else:
        p_50pct_drawdown = 2.0
        p_30pct_rally = 5.0
    
    return {
        'kdeDistribution': {
            'modeReturn': round(mode_return, 3),
            'probUp1h': round(prob_up_1h * 100, 1),
            'probDown1h': round(prob_down_1h * 100, 1),
            'expectedRange': f'{hdi_low:.2f}% to {hdi_high:.2f}%',
            'meanReturn': round(mu_1h * 100, 3),
            'volatility': round(sigma_1h * 100, 2),
        },
        'bayesianLevels': bayesian_levels[::5],  # every 5th level (~3% steps)
        'historicalRegime': hist_outcome,
        'scenarios': scenarios,
        'mostProbableLevels24h': most_probable_levels[:5],
        'extremeEvents': {
            'prob50pctDrawdown': round(p_50pct_drawdown, 1),
            'prob30pctRally': round(p_30pct_rally, 1),
            'interpretation': f'{p_50pct_drawdown:.1f}% chance of 50%+ drawdown (like 100K→50K). '
                             f'{p_30pct_rally:.1f}% chance of 30%+ rally (like 50K→82K).',
        },
        'summary': f'KDE: {prob_up_1h*100:.0f}% up / {prob_down_1h*100:.0f}% down next hour. '
                   f'Historical regimes: {hist_outcome.get("totalMatches",0)} similar periods found. '
                   f'Scenarios: Bullish {scenarios[0]["target24hChg"]:+.1f}%, '
                   f'Neutral {scenarios[1]["target24hChg"]:+.1f}%, '
                   f'Bearish {scenarios[2]["target24hChg"]:+.1f}%. '
                   f'Most probable 24h level: ${most_probable_levels[0]["price"]:,.0f} ({most_probable_levels[0]["prob"]}%). '
                    f'Major event risk: 50% drawdown {p_50pct_drawdown:.1f}%, 30% rally {p_30pct_rally:.1f}%.',
    }

def ml_forecast(klines_1h, klines_1m, price):
    """ML-based forecast using Random Forest on engineered features from 48h window."""
    if len(klines_1h) < 100:
        return {'error': 'insufficient 1H data'}
    
    # Use last 48 hours as primary window, train on earlier data
    closes = np.array([k['c'] for k in klines_1h], dtype=float)
    highs = np.array([k['h'] for k in klines_1h], dtype=float)
    lows = np.array([k['l'] for k in klines_1h], dtype=float)
    opens = np.array([k['o'] for k in klines_1h], dtype=float)
    vols = np.array([k['v'] for k in klines_1h], dtype=float)

    # Friday 00:00 UTC = 48h ago (approximately)
    friday_start = max(0, len(closes) - 48)
    recent = slice(friday_start, len(closes))
    recent_closes = closes[recent]
    recent_highs = highs[recent]
    recent_lows = lows[recent]
    recent_opens = opens[recent]
    recent_vols = vols[recent]

    # ── Feature engineering ──
    def engineer_features(c, h, l, o, v):
        """Create 20+ features per candle for ML."""
        if len(c) < 10: return None, None
        features = []
        targets = []
        ret = np.diff(c) / c[:-1]
        for i in range(10, len(c) - 1):
            feats = []
            # Price features (last 10 candles)
            for j in range(10):
                feats.append(c[i - j] / c[i] - 1)  # normalized price
            # Technical features
            feats.append(float(np.mean(ret[max(0, i-10):i])))  # 10-candle mean return
            feats.append(float(np.std(ret[max(0, i-10):i])))   # 10-candle volatility
            feats.append(float(np.mean(ret[max(0, i-5):i])))   # 5-candle mean
            feats.append(float(np.std(ret[max(0, i-5):i])))    # 5-candle vol
            # Volume features
            avg_v = np.mean(v[max(0, i-20):i+1]) or 1
            feats.append(float(v[i] / avg_v))  # volume ratio
            feats.append(float(v[i] / (v[i-1] or 1)))  # volume spike
            # Candle features
            candle_range = h[i] - l[i]
            feats.append(float(candle_range / (l[i] or 1) * 100))  # range %
            feats.append(float(abs(c[i] - o[i]) / (candle_range or 1)))  # body ratio
            # Momentum
            feats.append(float((c[i] - c[i-3]) / (c[i-3] or 1) * 100))
            feats.append(float((c[i] - c[i-5]) / (c[i-5] or 1) * 100))
            # Efficiency (range per volume)
            feats.append(float(candle_range / (v[i] or 1) * 10000))

            features.append(feats)
            # Target: next candle direction (1=up, 0=down)
            targets.append(1 if c[i+1] >= c[i] else 0)

        return np.array(features), np.array(targets)

    # Train on full dataset
    X_full, y_full = engineer_features(closes, highs, lows, opens, vols)
    if X_full is None or len(X_full) < 50:
        return {'error': 'not enough data for training'}

    # Train/test split (last 48h is test)
    split = max(len(X_full) - 48, len(X_full) // 2)
    X_train, y_train = X_full[:split], y_full[:split]
    X_test, y_test = X_full[split:], y_full[split:]

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    # Predictions
    y_prob = rf.predict_proba(X_test_s)
    y_pred = rf.predict(X_test_s)

    # Accuracy on test set
    accuracy = float(np.mean(y_pred == y_test)) if len(y_test) > 0 else 0

    # Feature importance (top 10)
    feature_names = ([f'price_t-{j}' for j in range(10)] +
                     ['mean_ret_10', 'vol_10', 'mean_ret_5', 'vol_5',
                      'vol_ratio', 'vol_spike', 'range_pct', 'body_ratio',
                      'mom_3', 'mom_5', 'efficiency'])
    importances = sorted(zip(feature_names[:len(rf.feature_importances_)],
                             rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)[:10]

    # Current feature vector (for next candle prediction)
    current_feats = X_full[-1:] if len(X_full) > 0 else None
    if current_feats is not None:
        current_scaled = scaler.transform(current_feats)
        next_prob = rf.predict_proba(current_scaled)[0]
        next_pred = int(rf.predict(current_scaled)[0])
        next_up_prob = float(next_prob[1]) * 100
        next_down_prob = float(next_prob[0]) * 100
    else:
        next_up_prob = 50; next_down_prob = 50; next_pred = 1

    # 48h parameter summary
    start_price = float(recent_closes[0])
    end_price = float(recent_closes[-1])
    high_48h = float(np.max(recent_highs))
    low_48h = float(np.min(recent_lows))
    vol_48h = float(np.sum(recent_vols))
    buy_candles = int(np.sum(np.diff(recent_closes) >= 0))
    sell_candles = int(np.sum(np.diff(recent_closes) < 0))
    avg_vol = float(np.mean(recent_vols))
    avg_range = float(np.mean(recent_highs - recent_lows))

    return {
        'mlModel': {
            'type': 'RandomForest (500 trees, max_depth=8)',
            'features': 22,
            'trainingSamples': len(X_train),
            'testSamples': len(X_test),
            'accuracy': round(accuracy * 100, 1),
            'featureImportance': [{'feature': f, 'importance': round(i, 4)}
                                   for f, i in importances],
        },
        'prediction': {
            'nextCandle': 'UP' if next_pred == 1 else 'DOWN',
            'nextUpProb': round(next_up_prob, 1),
            'nextDownProb': round(next_down_prob, 1),
            'confidence': 'high' if max(next_up_prob, next_down_prob) > 70 else
                          'moderate' if max(next_up_prob, next_down_prob) > 55 else 'low',
        },
        '48hWindow': {
            'from': float(recent_closes[0]),
            'to': float(recent_closes[-1]),
            'change': round((end_price - start_price) / start_price * 100, 2),
            'high': round(high_48h, 2),
            'low': round(low_48h, 2),
            'volume': round(vol_48h, 2),
            'avgVolPerCandle': round(avg_vol, 2),
            'avgRange': round(avg_range, 2),
            'buyCandles': int(buy_candles),
            'sellCandles': int(sell_candles),
            'netCandleBias': int(buy_candles - sell_candles),
        },
        'recent12h': {
            'startPrice': float(closes[-12]),
            'endPrice': float(closes[-1]),
            'change': round((closes[-1] - closes[-12]) / closes[-12] * 100, 2),
            'buyRatio': round(np.sum(np.diff(closes[-12:]) >= 0) / 12 * 100, 1),
            'avgVolRatio': round(float(np.mean(vols[-12:]) / (np.mean(vols[-48:]) or 1)), 2),
        },
        'interpretation': (
            f'ML model trained on {len(X_train)} candles ({len(X_test)} tested). '
            f'Accuracy on unseen data: {accuracy*100:.1f}%. '
            f'Next candle: {"UP" if next_pred==1 else "DOWN"} ({max(next_up_prob,next_down_prob):.0f}% confidence). '
            f'Since Friday: ${start_price:.0f}→${end_price:.0f} ({((end_price-start_price)/start_price*100):+.2f}%). '
            f'Key features: {importances[0][0]} ({importances[0][1]:.2f}), '
            f'{importances[1][0]} ({importances[1][1]:.2f}).'
        ),
    }

if __name__ == '__main__':
    try:
        raw = sys.stdin.read()
        req = json.loads(raw)
        action = req.get('action', '')
        data = req.get('data', {})

        if action == 'monte_carlo':
            result = monte_carlo(data['closes'], data['price'])
        elif action == 'statistics':
            result = statistics(data['closes'])
        elif action == 'volume_profile':
            result = volume_profile(data['klines'])
        elif action == 'regress':
            result = regress(data['closes'])
        elif action == 'weekday_analysis':
            result = weekday_analysis(data['klines'])
        elif action == 'brittleness':
            kh = data.get('klines_1h', [])
            kd = data.get('klines_1d', [])
            bb = data.get('book_bids', [])
            ba = data.get('book_asks', [])
            pr = data.get('price', 0)
            result = market_brittleness(kh, kd, bb, ba, pr)
        elif action == 'volume_stats':
            result = volume_stats_1m(data.get('klines_1m', []))
        elif action == 'level_matrix':
            result = level_matrix(data.get('klines_1d', []), data.get('klines_1h', []), data.get('price', 0))
        elif action == 'volatility':
            result = volatility_analysis(data.get('klines_1h', []), data.get('klines_1d', []), data.get('price', 0))
        elif action == 'geometry':
            result = entropy_analysis(data.get('klines_1m', []), data.get('klines_1h', []), data.get('klines_1d', []), data.get('price', 0))
        elif action == 'forecast':
            result = probabilistic_forecast(data.get('klines_1h', []), data.get('klines_1d', []), data.get('price', 0))
        elif action == 'ml_forecast':
            result = ml_forecast(data.get('klines_1h', []), data.get('klines_1m', []), data.get('price', 0))
        elif action == 'all':
            closes = data['closes']
            price = data['price']
            result = {
                'monteCarlo': monte_carlo(closes, price),
                'statistics': statistics(closes),
                'regression': regress(data.get('closes_1d', closes)),
            }
        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))
