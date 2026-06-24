"""Level strength analysis: which supports/resistances have highest probability of holding."""
import sys, json, urllib.request, numpy as np

def fetch_klines(symbol, interval='1h', limit=200):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    with urllib.request.urlopen(url) as r:
        d = json.loads(r.read())
    return [{'o':float(k[1]),'h':float(k[2]),'l':float(k[3]),'c':float(k[4]),'v':float(k[5]),'t':k[0]} for k in d]

def fetch_book(symbol):
    url = f'https://api.binance.com/api/v3/depth?symbol={symbol}&limit=50'
    with urllib.request.urlopen(url) as r:
        d = json.loads(r.read())
    return {'bids':[[float(b[0]),float(b[1])] for b in d['bids']],'asks':[[float(a[0]),float(a[1])] for a in d['asks']]}

def analyze_level_strength(klines_d1, klines_h1, book, price, name):
    highs = np.array([k['h'] for k in klines_d1])
    lows = np.array([k['l'] for k in klines_d1])
    closes = np.array([k['c'] for k in klines_d1])
    vols = np.array([k['v'] for k in klines_d1], dtype=float)
    h_highs = np.array([k['h'] for k in klines_h1])
    h_lows = np.array([k['l'] for k in klines_h1])

    swings = []
    for i in range(2, len(highs)-2):
        if highs[i] == max(highs[i-2:i+3]):
            swings.append({'price': float(highs[i]), 'type': 'resistance', 'idx': i})
        if lows[i] == min(lows[i-2:i+3]):
            swings.append({'price': float(lows[i]), 'type': 'support', 'idx': i})

    hh = float(np.max(highs[-60:]))
    ll = float(np.min(lows[-60:]))
    rng = hh - ll
    fib_levels = []
    for ratio, label in [(0.236,'23.6'),(0.382,'38.2'),(0.5,'50'),(0.618,'61.8'),(0.786,'78.6')]:
        pl = hh - rng * ratio
        fib_levels.append({'price': pl, 'type': 'support' if pl < price else 'resistance', 'fib': label})

    w_low, w_high = float(np.min(lows[-7:])), float(np.max(highs[-7:]))
    ema20, ema50 = float(np.mean(closes[-20:])), float(np.mean(closes[-50:]))
    vwap = float(np.sum(closes * vols) / np.sum(vols))

    bids = sorted(book['bids'], key=lambda x: x[1], reverse=True)[:3]
    asks = sorted(book['asks'], key=lambda x: x[1], reverse=True)[:3]

    def assess_level(lvl_price, lvl_type, lvl_source):
        dist = abs(lvl_price - price) / price * 100
        if lvl_type == 'support':
            mask = (lows <= lvl_price * 1.005) & (highs >= lvl_price * 0.995)
        else:
            mask = (highs >= lvl_price * 0.995) & (lows <= lvl_price * 1.005)
        touches = int(np.sum(mask))
        recent_mask = (h_lows <= lvl_price * 1.005) & (h_highs >= lvl_price * 0.995)
        recent_touches = int(np.sum(recent_mask))

        touch_indices = np.where(mask)[0]
        bounces = 0; breaks = 0
        for t in touch_indices:
            if t > 0 and t < len(closes) - 3:
                before = closes[t-1]
                after = closes[min(t+3, len(closes)-1)]
                if lvl_type == 'support':
                    if after > before * 1.005: bounces += 1
                    elif after < before * 0.995: breaks += 1
                else:
                    if after < before * 0.995: bounces += 1
                    elif after > before * 1.005: breaks += 1
        total_tests = bounces + breaks
        bounce_rate = bounces / max(1, total_tests) * 100

        score = 50
        score += min(20, touches * 3)
        score += min(15, bounces * 5)
        score -= breaks * 8
        if lvl_source in ('fib_61.8', 'fib_78.6') and lvl_price > price * 0.85 and lvl_price < price * 1.15:
            score += 10
        if lvl_source == 'weekly': score += 5
        score += recent_touches * 2
        score -= min(20, dist * 2) if dist < 1 else 0

        vol_at_level = float(np.sum(vols[mask])) if any(mask) else 0
        avg_vol = float(np.mean(vols))
        vol_ratio = vol_at_level / max(1, avg_vol * len(klines_d1) / 20)
        if vol_ratio > 1.5: score += 10
        elif vol_ratio > 1: score += 5

        if lvl_type == 'support':
            if any(abs(b[0] - lvl_price) / max(lvl_price,1) < 0.005 for b in bids): score += 15
        else:
            if any(abs(a[0] - lvl_price) / max(lvl_price,1) < 0.005 for a in asks): score += 15

        score = max(0, min(100, score))

        cat = 'HIGH IMPACT' if score >= 75 else 'STRONG' if score >= 60 else 'MODERATE' if score >= 40 else 'WEAK'
        return {
            'price': round(lvl_price, 2), 'type': lvl_type, 'source': lvl_source,
            'strength': score, 'touches': touches, 'recentTouches': recent_touches,
            'bounces': bounces, 'breaks': breaks, 'bounceRate': round(bounce_rate, 1),
            'distPct': round((lvl_price/price - 1) * 100, 2),
            'volRatio': round(vol_ratio, 2), 'category': cat,
        }

    candidates = []
    for s in swings:
        if s['type'] == 'support' and s['price'] < price:
            candidates.append(assess_level(s['price'], 'support', 'swing'))
    for s in swings:
        if s['type'] == 'resistance' and s['price'] > price:
            candidates.append(assess_level(s['price'], 'resistance', 'swing'))
    for f in fib_levels:
        candidates.append(assess_level(f['price'], f['type'], f'fib_{f["fib"]}'))
    candidates.append(assess_level(w_low, 'support', 'weekly'))
    candidates.append(assess_level(w_high, 'resistance', 'weekly'))
    candidates.append(assess_level(ema20, 'support' if ema20 < price else 'resistance', 'ema20'))
    candidates.append(assess_level(ema50, 'support' if ema50 < price else 'resistance', 'ema50'))
    candidates.append(assess_level(vwap, 'support' if vwap < price else 'resistance', 'vwap'))

    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: x['strength'], reverse=True):
        key = round(c['price'] / 100) * 100
        if key not in seen:
            seen.add(key)
            unique.append(c)
    for c in unique:
        c['distPct'] = round((c['price']/price - 1)*100, 2)

    supports = sorted([c for c in unique if c['type']=='support' and c['price']<price], key=lambda x: -x['price'])
    resistances = sorted([c for c in unique if c['type']=='resistance' and c['price']>price], key=lambda x: x['price'])

    def best(cat):
        matches = [c for c in unique if c['type']==cat and ((cat=='support' and c['price']<price) or (cat=='resistance' and c['price']>price))]
        return max(matches, key=lambda x: x['strength']) if matches else None

    return {
        'name': name, 'price': round(price, 2),
        'supports': supports[:10], 'resistances': resistances[:10],
        'topSupport': best('support'), 'topResistance': best('resistance'),
        'bidWall': bids[0] if bids else None, 'askWall': asks[0] if asks else None,
    }

def print_results(data):
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'  {data["name"]}  @  ${data["price"]:,.2f}')
    print(f'{sep}')
    print(f'\n  SUPPORTS  (sorted by strength ↓)')
    print(f'  {"-"*60}')
    hdr = f'  {"Level":<14} {"Source":<14} {"Dist":<7} {"Score":<7} {"Touches":<9} {"Bounce":<8} {"Break":<7}'
    print(hdr)
    print(f'  {"-"*60}')
    for s in data['supports']:
        print(f'  ${s["price"]:<8,.2f}  {s["source"]:<14} {s["distPct"]:>+5.2f}%  {s["strength"]:>3}/100 {s["category"]:<10} {s["touches"]}t/{s["bounces"]}b  {s["breaks"]}x')

    print(f'\n  RESISTANCES  (sorted by strength ↓)')
    print(f'  {"-"*60}')
    print(hdr)
    print(f'  {"-"*60}')
    for r in data['resistances']:
        print(f'  ${r["price"]:<8,.2f}  {r["source"]:<14} {r["distPct"]:>+5.2f}%  {r["strength"]:>3}/100 {r["category"]:<10} {r["touches"]}t/{r["bounces"]}b  {r["breaks"]}x')

    ts = data['topSupport']
    tr = data['topResistance']
    print(f'\n  >>> BEST SUPPORT: ${ts["price"]:,.2f} ({ts["source"]}, {ts["strength"]}/100, {ts["touches"]} touches, {ts["bounceRate"]}% bounce)')
    print(f'  >>> BEST RESISTANCE: ${tr["price"]:,.2f} ({tr["source"]}, {tr["strength"]}/100, {tr["touches"]} touches, {tr["bounceRate"]}% bounce)')
    if data['bidWall']:
        print(f'  >>> ORDER BOOK BID WALL: ${data["bidWall"][0]:,.2f} | {data["bidWall"][1]:.4f} units')
    if data['askWall']:
        print(f'  >>> ORDER BOOK ASK WALL: ${data["askWall"][0]:,.2f} | {data["askWall"][1]:.4f} units')

if __name__ == '__main__':
    btc_d1 = fetch_klines('BTCUSDT', '1d', 200)
    btc_h1 = fetch_klines('BTCUSDT', '1h', 168)
    btc_book = fetch_book('BTCUSDT')
    btc_price = btc_d1[-1]['c']

    xaut_d1 = fetch_klines('XAUTUSDT', '1d', 200)
    xaut_h1 = fetch_klines('XAUTUSDT', '1h', 168)
    xaut_book = fetch_book('XAUTUSDT')
    xaut_price = xaut_d1[-1]['c']

    paxg_d1 = fetch_klines('PAXGUSDT', '1d', 200)
    paxg_h1 = fetch_klines('PAXGUSDT', '1h', 168)
    paxg_book = fetch_book('PAXGUSDT')
    paxg_price = paxg_d1[-1]['c']

    btc = analyze_level_strength(btc_d1, btc_h1, btc_book, btc_price, 'BTC')
    xaut = analyze_level_strength(xaut_d1, xaut_h1, xaut_book, xaut_price, 'XAUT (Gold)')
    paxg = analyze_level_strength(paxg_d1, paxg_h1, paxg_book, paxg_price, 'PAXG (Gold)')

    print_results(btc)
    print_results(xaut)
    print_results(paxg)
