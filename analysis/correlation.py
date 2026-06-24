"""Cross-asset correlation: currencies, commodities, crypto, indices."""
import sys, json, urllib.request, numpy as np, datetime, time

# ── Asset Universe ──
ASSETS = {
    # Indices
    'SP500': {'url': 'yahoo', 'symbol': '^GSPC', 'type': 'index'},
    'DOW':   {'url': 'yahoo', 'symbol': '^DJI', 'type': 'index'},
    'NASDAQ':{'url': 'yahoo', 'symbol': '^IXIC', 'type': 'index'},
    # Currencies
    'DXY':   {'url': 'yahoo', 'symbol': 'DX-Y.NYB', 'type': 'currency'},
    'EURUSD':{'url': 'yahoo', 'symbol': 'EURUSD=X', 'type': 'currency'},
    'GBPUSD':{'url': 'yahoo', 'symbol': 'GBPUSD=X', 'type': 'currency'},
    'USDJPY':{'url': 'yahoo', 'symbol': 'USDJPY=X', 'type': 'currency'},
    # Commodities
    'GOLD':  {'url': 'yahoo', 'symbol': 'GC=F', 'type': 'commodity'},
    'SILVER':{'url': 'yahoo', 'symbol': 'SI=F', 'type': 'commodity'},
    'CRUDE': {'url': 'yahoo', 'symbol': 'CL=F', 'type': 'commodity'},
    'NGAS':  {'url': 'yahoo', 'symbol': 'NG=F', 'type': 'commodity'},
    # Crypto
    'BTC':   {'url': 'binance', 'symbol': 'BTCUSDT', 'type': 'crypto'},
    'ETH':   {'url': 'binance', 'symbol': 'ETHUSDT', 'type': 'crypto'},
    'SOL':   {'url': 'binance', 'symbol': 'SOLUSDT', 'type': 'crypto'},
    'XAUT':  {'url': 'binance', 'symbol': 'XAUTUSDT', 'type': 'crypto'},
}

def fetch_yahoo(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=6mo"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read())
        c = d['chart']['result'][0]['indicators']['quote'][0]['close']
        return np.array([float(x) for x in c if x is not None])
    except: return np.array([])

def fetch_binance(symbol):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=200"
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.loads(r.read())
        return np.array([float(k[4]) for k in d])
    except: return np.array([])

def compute_stats(prices):
    if len(prices) < 10:
        return None
    r = np.diff(prices) / prices[:-1]
    mu = float(np.mean(r)) * 100
    sigma = float(np.std(r)) * 100
    var95 = float(np.percentile(r, 5)) * 100
    half = len(r) // 2
    hurst = 0.5
    if half > 5:
        rs1 = (np.max(r[:half]) - np.min(r[:half])) / (np.std(r[:half]) or 1)
        rs2 = (np.max(r[half:]) - np.min(r[half:])) / (np.std(r[half:]) or 1)
        if rs1 > 0 and rs2 > 0:
            hurst = float(np.log(rs2/rs1) / np.log((len(r)-half)/half))
            hurst = max(0, min(1, hurst))
    return {
        'price': round(float(prices[-1]), 2),
        'chg1d': round((prices[-1]/prices[-2]-1)*100, 2) if len(prices) > 1 else 0,
        'chg1w': round((prices[-1]/prices[-7]-1)*100, 2) if len(prices) > 7 else 0,
        'volatility': round(sigma, 2),
        'var95': round(var95, 2),
        'hurst': round(hurst, 2),
        'regime': 'trending' if hurst > 0.65 else ('mean-reverting' if hurst < 0.35 else 'random'),
    }

def correlation_matrix(all_prices):
    names = sorted(all_prices.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 1.0
            else:
                a1 = all_prices[names[i]]
                a2 = all_prices[names[j]]
                ml = min(len(a1), len(a2))
                r1 = np.diff(a1[-ml:]) / a1[-ml:-1]
                r2 = np.diff(a2[-ml:]) / a2[-ml:-1]
                if len(r1) > 2 and np.std(r1) > 0 and np.std(r2) > 0:
                    mat[i][j] = round(float(np.corrcoef(r1, r2)[0,1]), 3)
                else:
                    mat[i][j] = 0
    return {'names': names, 'matrix': mat.tolist()}

def cluster_assets(corr):
    """Group assets by correlation similarity using simple threshold clustering."""
    names = corr['names']
    mat = corr['matrix']
    n = len(names)
    assigned = [False] * n
    clusters = []
    
    for i in range(n):
        if assigned[i]: continue
        cluster = [i]
        assigned[i] = True
        for j in range(i+1, n):
            if not assigned[j] and abs(mat[i][j]) > 0.5:
                cluster.append(j)
                assigned[j] = True
        cluster_types = [ASSETS[names[c]]['type'] for c in cluster if names[c] in ASSETS]
        types_str = '/'.join(sorted(set(cluster_types)))
        clusters.append({
            'name': '+'.join(names[c] for c in cluster),
            'size': len(cluster),
            'members': [{'name': names[c], 'type': ASSETS.get(names[c], {}).get('type', '?')} for c in cluster],
            'types': types_str,
        })
    
    return clusters

def main():
    all_prices = {}
    stats = {}
    failures = []
    
    for name, cfg in ASSETS.items():
        try:
            if cfg['url'] == 'yahoo':
                prices = fetch_yahoo(cfg['symbol'])
            else:
                prices = fetch_binance(cfg['symbol'])
            
            if len(prices) > 10:
                all_prices[name] = prices
                s = compute_stats(prices)
                if s:
                    s['name'] = name
                    s['type'] = cfg['type']
                    stats[name] = s
            else:
                failures.append(f"{name}: insufficient data ({len(prices)} candles)")
            time.sleep(0.3)  # Rate limit
        except Exception as e:
            failures.append(f"{name}: {e}")
    
    # Correlation
    corr = correlation_matrix(all_prices) if len(all_prices) >= 2 else {'names':[],'matrix':[]}
    
    # Clusters
    clusters = cluster_assets(corr) if corr['names'] else []
    
    return {
        'assets': stats,
        'correlation': corr,
        'clusters': clusters,
        'failures': failures,
        'total': len(stats),
        'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

if __name__ == '__main__':
    print(json.dumps(main(), default=str))
