"""Full market overview: indices, metals, crypto, forex — stats, probability, geometry."""
import json, urllib.request, sys, numpy as np, datetime

# ── Yahoo Finance helper ──
def yahoo_chart(symbol, interval='1d', range_='6mo'):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        r = d['chart']['result'][0]
        t = r['timestamp']
        q = r['indicators']['quote'][0]
        c = [x for x in q['close'] if x is not None]
        h = [x for x in q.get('high', [None]*len(q['close'])) if x is not None]
        l = [x for x in q.get('low', [None]*len(q['close'])) if x is not None]
        return {'closes': c, 'highs': h, 'lows': l, 'timestamps': t[:len(c)]}
    except Exception as e:
        return {'error': str(e)}

# ── Statistical calculator ──
def stats(arr):
    a = np.array(arr, dtype=float)
    r = np.diff(a) / a[:-1]
    mu = float(np.mean(r)) * 100
    sigma = float(np.std(r)) * 100
    # Skew, kurtosis
    s = float(np.mean(((r - np.mean(r)) / (np.std(r) or 1)) ** 3)) if len(r) > 2 else 0
    k = float(np.mean(((r - np.mean(r)) / (np.std(r) or 1)) ** 4)) if len(r) > 2 else 0
    # VaR
    sorted_r = np.sort(r)
    var95 = float(np.percentile(sorted_r, 5)) * 100
    var99 = float(np.percentile(sorted_r, 1)) * 100
    # Hurst (simplified)
    half = len(r) // 2
    hurst = 0.5
    if half > 5:
        rs1 = (np.max(r[:half]) - np.min(r[:half])) / (np.std(r[:half]) or 1)
        rs2 = (np.max(r[half:]) - np.min(r[half:])) / (np.std(r[half:]) or 1)
        hurst = float(np.log(rs2 / rs1) / np.log((len(r) - half) / half)) if rs1 > 0 and rs2 > 0 else 0.5
        hurst = max(0, min(1, hurst))
    return {
        'price': round(float(a[-1]), 2),
        'chg1d': round(float((a[-1] / a[-2] - 1) * 100), 2) if len(a) > 1 else 0,
        'chg1w': round(float((a[-1] / a[-7] - 1) * 100), 2) if len(a) > 7 else 0,
        'chg1m': round(float((a[-1] / a[-30] - 1) * 100), 2) if len(a) > 30 else 0,
        'meanRet': round(mu, 3),
        'volatility': round(sigma, 3),
        'skew': round(s, 3),
        'kurtosis': round(k, 3),
        'var95': round(var95, 3),
        'var99': round(var99, 3),
        'hurst': round(hurst, 3),
        'trend': 'up' if mu > 0 else 'down',
        'regime': 'trending' if hurst > 0.65 else ('mean-reverting' if hurst < 0.35 else 'random'),
        'fatTail': abs(k) > 3,
    }

# ── Correlation matrix ──
def correlation_matrix(price_dict):
    names = list(price_dict.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == j:
                mat[i][j] = 1.0
            else:
                a1 = np.array(price_dict[n1], dtype=float)
                a2 = np.array(price_dict[n2], dtype=float)
                min_len = min(len(a1), len(a2))
                r1 = np.diff(a1[-min_len:]) / a1[-min_len:-1]
                r2 = np.diff(a2[-min_len:]) / a2[-min_len:-1]
                if len(r1) > 2 and np.std(r1) > 0 and np.std(r2) > 0:
                    mat[i][j] = round(float(np.corrcoef(r1, r2)[0,1]), 3)
                else:
                    mat[i][j] = 0
    return {'names': names, 'matrix': mat.tolist()}

# ── Market Regime Detection ──
def market_regime(assets_stats):
    """Detect overall market regime from all assets."""
    avg_vol = np.mean([s['volatility'] for s in assets_stats.values() if s])
    avg_hurst = np.mean([s['hurst'] for s in assets_stats.values() if s])
    avg_skew = np.mean([s['skew'] for s in assets_stats.values() if s])
    up_count = sum(1 for s in assets_stats.values() if s and s['trend'] == 'up')
    dn_count = sum(1 for s in assets_stats.values() if s and s['trend'] == 'down')
    total = up_count + dn_count
    
    if avg_vol > 2:
        vol_regime = "HIGH VOLATILITY"
    elif avg_vol > 1:
        vol_regime = "ELEVATED VOLATILITY"
    else:
        vol_regime = "NORMAL VOLATILITY"
    
    if avg_hurst > 0.6:
        trend_regime = "TRENDING"
    elif avg_hurst < 0.4:
        trend_regime = "MEAN-REVERTING"
    else:
        trend_regime = "RANDOM WALK"
    
    bias = "BULLISH" if up_count > dn_count * 1.5 else ("BEARISH" if dn_count > up_count * 1.5 else "MIXED")
    
    return f"{vol_regime} · {trend_regime} · {bias} ({up_count}U/{dn_count}D)"

# ── Geometry / Levels ──
def key_levels(arr):
    a = np.array(arr, dtype=float)
    price = a[-1]
    # Support/Resistance from recent swings
    hi = float(np.max(a[-20:]))
    lo = float(np.min(a[-20:]))
    mid = (hi + lo) / 2
    # Fib levels
    rng = hi - lo
    fibs = {}
    for ratio, label in [(0.236, '23.6'), (0.382, '38.2'), (0.5, '50'), (0.618, '61.8')]:
        price_level = hi - rng * ratio
        fibs[label] = round(price_level, 2)
    
    return {
        'price': round(price, 2),
        'high20': round(hi, 2),
        'low20': round(lo, 2),
        'range': round(rng, 2),
        'rangePct': round(rng / lo * 100, 2),
        'mid': round(mid, 2),
        'fibs': fibs,
        'position': 'oversold' if price < lo + rng * 0.236 else ('overbought' if price > hi - rng * 0.236 else 'neutral'),
    }

# ── Main ──
def full_market_overview():
    assets = {
        'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'DOW': '^DJI',
        'GOLD': 'GC=F', 'SILVER': 'SI=F',
        'DXY': 'DX-Y.NYB',
        'BTC': None, 'ETH': None,
    }
    
    result = {}
    prices = {}
    failures = []
    
    for name, symbol in assets.items():
        if name in ('BTC', 'ETH'):
            # Use Binance for crypto
            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={name}USDT&interval=1d&limit=200"
                with urllib.request.urlopen(url, timeout=10) as r:
                    data = json.loads(r.read())
                closes = [float(k[4]) for k in data]
                high = [float(k[2]) for k in data]
                low = [float(k[3]) for k in data]
                s = stats(closes)
                s['name'] = name
                s['source'] = 'Binance'
                result[name] = s
                prices[name] = closes
            except Exception as e:
                failures.append(f"{name}: {e}")
        else:
            y = yahoo_chart(symbol)
            if 'error' not in y and len(y['closes']) > 20:
                s = stats(y['closes'])
                s['name'] = name
                s['source'] = 'Yahoo'
                result[name] = s
                prices[name] = y['closes']
            else:
                failures.append(f"{name}: {y.get('error','no data')}")
    
    # Geometry for each
    geo = {}
    for name in result:
        if name in prices and len(prices[name]) > 20:
            geo[name] = key_levels(prices[name])
    
    # Correlation matrix
    corr = correlation_matrix(prices) if len(prices) >= 2 else {'error': 'insufficient data'}
    
    # Market regime
    regime = market_regime(result)
    
    # Interpretation
    leaders = sorted(result.keys(), key=lambda n: abs(result[n]['volatility']), reverse=True)
    safest = sorted(result.keys(), key=lambda n: abs(result[n]['volatility']))[:3]
    
    return {
        'assets': {k: v for k, v in result.items()},
        'geometry': geo,
        'correlation': corr,
        'marketRegime': regime,
        'volatilityLeaders': [(n, result[n]['volatility']) for n in leaders[:3]],
        'safestAssets': [(n, result[n]['volatility']) for n in safest],
        'failures': failures,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

if __name__ == '__main__':
    result = full_market_overview()
    print(json.dumps(result, default=str))
