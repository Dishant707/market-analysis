"""Multi-asset crypto quant analysis: BTC, ETH, SOL, XAUT."""
import sys, json, urllib.request, numpy as np, datetime

# ── Data Fetching ──
def binance_klines(symbol, interval='1d', limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url) as r:
        d = json.loads(r.read())
    return [{'t':k[0],'o':float(k[1]),'h':float(k[2]),'l':float(k[3]),'c':float(k[4]),'v':float(k[5])} for k in d]

def binance_book(symbol):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
    with urllib.request.urlopen(url) as r:
        d = json.loads(r.read())
    return {'bids':[[float(b[0]),float(b[1])] for b in d['bids']], 'asks':[[float(a[0]),float(a[1])] for a in d['asks']]}

def binance_ticker(symbol):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    with urllib.request.urlopen(url) as r:
        d = json.loads(r.read())
    return {'price':float(d['lastPrice']),'chg24h':float(d['priceChangePercent']),'high':float(d['highPrice']),'low':float(d['lowPrice']),'vol':float(d['quoteVolume'])}

# ── Statistical Analysis ──
def stats(klines):
    c = np.array([k['c'] for k in klines])
    h = np.array([k['h'] for k in klines])
    l = np.array([k['l'] for k in klines])
    r = np.diff(c) / c[:-1] * 100
    price = float(c[-1])
    
    mu = float(np.mean(r))
    sigma = float(np.std(r))
    skew = float(np.mean(((r-mu)/(sigma or 1))**3)) if len(r)>2 else 0
    kurt = float(np.mean(((r-mu)/(sigma or 1))**4)) if len(r)>2 else 0
    var95 = float(np.percentile(r, 5))
    
    # Hurst
    half = len(r)//2
    hurst = 0.5
    if half > 5:
        rs1 = (np.max(r[:half])-np.min(r[:half]))/(np.std(r[:half]) or 1)
        rs2 = (np.max(r[half:])-np.min(r[half:]))/(np.std(r[half:]) or 1)
        hurst = float(np.log(rs2/rs1)/np.log((len(r)-half)/half)) if rs1>0 and rs2>0 else 0.5
        hurst = max(0, min(1, hurst))
    
    # CVD
    cvd_vals = []
    cum = 0
    for k in klines[1:]:
        dir_ = 1 if k['c'] >= k['o'] else -1
        cum += dir_ * k['v']
        cvd_vals.append(cum)
    cvd_trend = cvd_vals[-1] - (cvd_vals[-10] if len(cvd_vals) > 10 else cvd_vals[0]) if cvd_vals else 0
    
    # Volume profile - find POC and clusters
    bins = 20
    min_p = float(np.min(l))
    max_p = float(np.max(h))
    bin_w = (max_p - min_p) / bins
    vol_per_bin = np.zeros(bins)
    for k in klines:
        lo = max(0, min(bins-1, int((k['l']-min_p)/bin_w)))
        hi = max(0, min(bins-1, int((k['h']-min_p)/bin_w)))
        vol_per_bin[lo:hi+1] += k['v'] / (hi-lo+1)
    poc_bin = int(np.argmax(vol_per_bin))
    poc_price = min_p + (poc_bin+0.5)*bin_w
    
    # Weekly structure
    wk_high = float(np.max(h[-7:]))
    wk_low = float(np.min(l[-7:]))
    wk_open = float(klines[-8]['c']) if len(klines) > 8 else float(klines[0]['c'])
    wk_close = price
    wk_range = (wk_high - wk_low) / wk_low * 100
    
    # Key levels from recent swings
    swings = []
    for i in range(2, len(h)-2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swings.append({'price':float(h[i]),'type':'resistance','idx':i})
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swings.append({'price':float(l[i]),'type':'support','idx':i})
    
    supports = sorted([s for s in swings if s['type']=='support' and s['price']<price], key=lambda x: x['price'], reverse=True)[:3]
    resistances = sorted([s for s in swings if s['type']=='resistance' and s['price']>price], key=lambda x: x['price'])[:3]
    
    # Weekly levels
    wk_supports = [s for s in supports if s['price'] < wk_low * 1.01][:2]
    wk_resistances = [r for r in resistances if r['price'] > wk_high * 0.99][:2]
    
    return {
        'price': round(price, 2),
        'chg24h': round((c[-1]/c[-2]-1)*100, 2) if len(c)>1 else 0,
        'volatility': round(sigma, 3),
        'var95': round(var95, 3),
        'skew': round(skew, 3),
        'kurtosis': round(kurt, 3),
        'hurst': round(hurst, 3),
        'cvdTrend': round(cvd_trend, 2),
        'cvdDirection': 'up' if cvd_trend > 0 else 'down',
        'poc': round(poc_price, 2),
        'weekly': {
            'open': round(wk_open, 2),
            'high': round(wk_high, 2),
            'low': round(wk_low, 2),
            'close': round(wk_close, 2),
            'range': round(wk_range, 2),
            'direction': 'down' if wk_close < wk_open else 'up'
        },
        'levels': {
            'supports': [{'price':round(s['price'],2),'dist':round((s['price']/price-1)*100,2)} for s in supports],
            'resistances': [{'price':round(r['price'],2),'dist':round((r['price']/price-1)*100,2)} for r in resistances],
            'weekly_support': [{'price':round(s['price'],2),'dist':round((s['price']/price-1)*100,2)} for s in wk_supports],
            'weekly_resistance': [{'price':round(r['price'],2),'dist':round((r['price']/price-1)*100,2)} for r in wk_resistances],
        },
        'regime': 'trending' if hurst > 0.65 else ('mean-reverting' if hurst < 0.35 else 'random'),
    }

def control_analysis(klines, book):
    c = np.array([k['c'] for k in klines])
    price = float(c[-1])
    # Price action control
    up = sum(1 for i in range(1, len(c)) if c[i] >= c[i-1])
    dn = sum(1 for i in range(1, len(c)) if c[i] < c[i-1])
    pa_control = 'buyers' if up > dn else 'sellers'
    
    # Volume control (last 24 candles)
    last = klines[-24:] if len(klines) >= 24 else klines
    buy_vol = sum(k['v'] for k in last if k['c'] >= k['o'])
    sell_vol = sum(k['v'] for k in last if k['c'] < k['o'])
    vol_control = 'sellers' if sell_vol > buy_vol * 1.1 else ('buyers' if buy_vol > sell_vol * 1.1 else 'neutral')
    
    # Order book control
    bid_depth = sum(b[0]*b[1] for b in book['bids'][:10])
    ask_depth = sum(a[0]*a[1] for a in book['asks'][:10])
    ob_control = 'buyers' if bid_depth > ask_depth else 'sellers'
    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) * 100 if (bid_depth + ask_depth) > 0 else 0
    
    # Reversal level
    best_bid = book['bids'][0][0] if book['bids'] else price
    best_ask = book['asks'][0][0] if book['asks'] else price
    reversal = (best_bid + best_ask) / 2
    rev_dist = (reversal / price - 1) * 100
    
    # Consensus
    controls = [ob_control, pa_control, vol_control]
    buyer_count = sum(1 for c in controls if c == 'buyers')
    seller_count = sum(1 for c in controls if c == 'sellers')
    controller = 'buyers' if buyer_count > seller_count else ('sellers' if seller_count > buyer_count else 'neutral')
    
    # Top walls
    max_bid = max(book['bids'], key=lambda x: x[1]) if book['bids'] else None
    max_ask = max(book['asks'], key=lambda x: x[1]) if book['asks'] else None
    total_bid_qty = sum(b[1] for b in book['bids'])
    total_ask_qty = sum(a[1] for a in book['asks'])
    
    return {
        'controller': controller,
        'evidences': {'orderBook': ob_control, 'priceAction': pa_control, 'volumeDelta': vol_control},
        'reversalLevel': round(reversal, 2),
        'reversalDist': round(rev_dist, 2),
        'imbalance': round(imbalance, 1),
        'topBid': {'price':round(max_bid[0],2),'qty':max_bid[1],'pct':round(max_bid[1]/total_bid_qty*100,1)} if max_bid else None,
        'topAsk': {'price':round(max_ask[0],2),'qty':max_ask[1],'pct':round(max_ask[1]/total_ask_qty*100,1)} if max_ask else None,
    }

def volume_analysis(klines):
    """Volume profile, buy/sell ratio, POC, volume clusters."""
    c = np.array([k['c'] for k in klines])
    o = np.array([k['o'] for k in klines])
    v = np.array([k['v'] for k in klines])
    h = np.array([k['h'] for k in klines])
    l = np.array([k['l'] for k in klines])
    
    buy_vol = sum(v[i] for i in range(len(c)) if c[i] >= o[i])
    sell_vol = sum(v[i] for i in range(len(c)) if c[i] < o[i])
    total_vol = buy_vol + sell_vol
    
    # Efficiency: how much range per unit volume
    eff = (h-l) / (v+1) * 10000
    avg_eff = float(np.mean(eff))
    
    return {
        'buyRatio': round(buy_vol/total_vol*100, 1) if total_vol > 0 else 0,
        'sellRatio': round(sell_vol/total_vol*100, 1) if total_vol > 0 else 0,
        'delta': round(buy_vol - sell_vol, 2),
        'avgEfficiency': round(avg_eff, 3),
        'totalVol': round(total_vol, 2),
    }

def correlation(assets_data):
    """Compute correlation matrix from daily returns."""
    names = list(assets_data.keys())
    prices = {}
    for n in names:
        prices[n] = np.array([k['c'] for k in assets_data[n]])
    
    n = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 1.0
            else:
                min_l = min(len(prices[names[i]]), len(prices[names[j]]))
                r1 = np.diff(prices[names[i]][-min_l:]) / prices[names[i]][-min_l:-1]
                r2 = np.diff(prices[names[j]][-min_l:]) / prices[names[j]][-min_l:-1]
                if len(r1) > 2 and np.std(r1) > 0 and np.std(r2) > 0:
                    mat[i][j] = round(float(np.corrcoef(r1, r2)[0,1]), 3)
    
    return {'names': names, 'matrix': mat.tolist()}

# ── Main Entry ──
def multissset_analysis():
    pairs = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'XAUT': 'XAUTUSDT'}
    result = {}
    raw_klines = {}
    
    for name, pair in pairs.items():
        try:
            d1 = binance_klines(pair, '1d', 200)
            h1 = binance_klines(pair, '1h', 168)
            book = binance_book(pair)
            ticker = binance_ticker(pair)
            price = ticker['price']
            
            s = stats(d1)
            ctrl = control_analysis(h1, book)
            vol = volume_analysis(h1)
            
            result[name] = {
                'price': price,
                'chg24h': ticker['chg24h'],
                'high24h': ticker['high'],
                'low24h': ticker['low'],
                'vol24h': ticker['vol'],
                'stats': s,
                'control': ctrl,
                'volume': vol,
                'weekly': s['weekly'],
                'levels': s['levels'],
                'orderBook': {
                    'bestBid': book['bids'][0][0] if book['bids'] else 0,
                    'bestAsk': book['asks'][0][0] if book['asks'] else 0,
                    'spread': round(book['asks'][0][0] - book['bids'][0][0], 2) if book['bids'] and book['asks'] else 0,
                    'bidWalls': [(b[0], b[1]) for b in book['bids'][:5]],
                    'askWalls': [(a[0], a[1]) for a in book['asks'][:5]],
                },
                'volatility': s['volatility'],
                'hurst': s['hurst'],
                'var95': s['var95'],
            }
            raw_klines[name] = d1
        except Exception as e:
            result[name] = {'error': str(e)}
    
    # Correlation matrix
    corr_matrix = correlation(raw_klines)
    
    return {
        'assets': result,
        'correlation': corr_matrix,
        'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

if __name__ == '__main__':
    print(json.dumps(multissset_analysis(), default=str))
