"""Advanced crypto quant: Markov chain, thrust, whales, simulations."""
import sys, json, urllib.request, numpy as np, datetime
from collections import Counter

# ── Market Regimes ──
def detect_regimes(closes):
    """Classify each candle as BULLISH, BEARISH, or RANGING."""
    r = np.diff(closes) / closes[:-1] * 100
    vol = np.std(r)
    up_thresh = vol * 0.5
    dn_thresh = -vol * 0.5
    regimes = []
    for i in range(len(r)):
        if r[i] > up_thresh: regimes.append('BULL')
        elif r[i] < dn_thresh: regimes.append('BEAR')
        else: regimes.append('RANGE')
    return regimes

# ── Markov Chain ──
def markov_chain(closes):
    """Build 3-state Markov chain and predict next regime."""
    if len(closes) < 10: return {'error': 'insufficient data'}
    regimes = detect_regimes(closes)
    states = ['BULL', 'BEAR', 'RANGE']
    n = len(states)
    trans = {s: {t: 0 for t in states} for s in states}
    counts = {s: 0 for s in states}
    
    for i in range(len(regimes) - 1):
        curr = regimes[i]
        nxt = regimes[i + 1]
        trans[curr][nxt] += 1
        counts[curr] += 1
    
    # Normalize to probabilities
    probs = {}
    for s in states:
        total = sum(trans[s].values())
        probs[s] = {t: round(trans[s][t] / total, 3) if total > 0 else 0 for t in states}
    
    # Stationary distribution (eigenvector)
    # Simplified: iterate transition until convergence
    vec = np.ones(n) / n
    P = np.array([[probs[s][t] for t in states] for s in states])
    for _ in range(100):
        vec_new = vec @ P
        if np.allclose(vec, vec_new): break
        vec = vec_new
    stationary = {states[i]: round(float(vec[i]) * 100, 1) for i in range(n)}
    
    # Next regime prediction from last state
    last = regimes[-1]
    next_probs = probs[last]
    next_regime = max(next_probs, key=next_probs.get)
    next_confidence = next_probs[next_regime]
    
    # Regime durations
    durations = {s: [] for s in states}
    curr = regimes[0]
    length = 1
    for r in regimes[1:]:
        if r == curr: length += 1
        else: durations[curr].append(length); curr = r; length = 1
    durations[curr].append(length)
    avg_duration = {s: round(np.mean(v), 1) for s, v in durations.items() if v}
    
    return {
        'transitionMatrix': probs,
        'stationaryDistribution': stationary,
        'currentRegime': last,
        'nextRegime': {'predicted': next_regime, 'probability': round(next_confidence * 100, 1)},
        'avgDuration': avg_duration,
        'regimeCounts': {s: counts[s] for s in states},
        'interpretation': (
            f"Currently {last}. Next most likely: {next_regime} ({next_confidence*100:.0f}% prob). "
            f"Long-term: {max(stationary, key=stationary.get)} dominates ({stationary[max(stationary, key=stationary.get)]:.0f}%). "
            f"Avg {last} lasts {avg_duration.get(last, 0)} candles."
        ),
    }

# ── Volume Dominance & Thrust ──
def volume_thrust(h1_klines):
    """Analyze which side dominates volume and needs less effort."""
    if len(h1_klines) < 24: return {'error': 'insufficient data'}
    
    c = np.array([k['c'] for k in h1_klines])
    o = np.array([k['o'] for k in h1_klines])
    h = np.array([k['h'] for k in h1_klines])
    l = np.array([k['l'] for k in h1_klines])
    v = np.array([k['v'] for k in h1_klines], dtype=float)
    
    # Volume dominance
    buy_vol = sum(v[i] for i in range(len(c)) if c[i] >= o[i])
    sell_vol = sum(v[i] for i in range(len(c)) if c[i] < o[i])
    total_vol = buy_vol + sell_vol
    dom_side = 'buyers' if buy_vol > sell_vol else 'sellers'
    
    # Recent dominance (last 12)
    r_buy = sum(v[-12+i] for i in range(12) if c[-12+i] >= o[-12+i]) if len(c) >= 12 else buy_vol
    r_sell = sum(v[-12+i] for i in range(12) if c[-12+i] < o[-12+i]) if len(c) >= 12 else sell_vol
    r_total = r_buy + r_sell
    r_dom = 'buyers' if r_buy > r_sell else 'sellers'
    
    # Thrust (efficiency): how much price moves per unit volume
    returns = np.diff(c) / c[:-1] * 100
    abs_ret = np.abs(returns)
    buy_mask = returns > 0
    sell_mask = returns < 0
    buy_eff = float(np.mean(abs_ret[buy_mask] / v[1:][buy_mask])) if sum(buy_mask) > 0 else 0
    sell_eff = float(np.mean(abs_ret[sell_mask] / v[1:][sell_mask])) if sum(sell_mask) > 0 else 0
    thrust_side = 'buyers' if buy_eff > sell_eff else ('sellers' if sell_eff > buy_eff else 'neutral')
    
    # Recent thrust (last 24 candles)
    r_ret = returns[-24:] if len(returns) >= 24 else returns
    r_v = v[1:][-24:] if len(v[1:]) >= 24 else v[1:]
    r_abs = np.abs(r_ret)
    r_buy_m = r_ret > 0
    r_sell_m = r_ret < 0
    r_buy_e = float(np.mean(r_abs[r_buy_m] / r_v[r_buy_m])) if sum(r_buy_m) > 0 else 0
    r_sell_e = float(np.mean(r_abs[r_sell_m] / r_v[r_sell_m])) if sum(r_sell_m) > 0 else 0
    r_thrust = 'buyers' if r_buy_e > r_sell_e else ('sellers' if r_sell_e > r_buy_e else 'neutral')
    
    return {
        'volumeDominance': {
            'overall': dom_side,
            'overallRatio': round(buy_vol / total_vol * 100, 1) if total_vol > 0 else 0,
            'recent': r_dom,
            'recentRatio': round(r_buy / r_total * 100, 1) if r_total > 0 else 0,
        },
        'thrust': {
            'overall': thrust_side,
            'buyEfficiency': round(buy_eff * 10000, 2),
            'sellEfficiency': round(sell_eff * 10000, 2),
            'recent': r_thrust,
            'recentBuyEff': round(r_buy_e * 10000, 2),
            'recentSellEff': round(r_sell_e * 10000, 2),
            'interpretation': f'{thrust_side.upper()} need less volume to move price' if thrust_side != 'neutral' else 'Both sides equal',
        },
    }

# ── Whale Detection ──
def detect_whales(symbol):
    """Fetch recent trades and flag large ones."""
    try:
        url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=100"
        with urllib.request.urlopen(url, timeout=10) as r:
            trades = json.loads(r.read())
    except:
        return {'error': 'trade data unavailable'}
    
    if not trades: return {'error': 'no trades'}
    
    # Group by side
    prices = [float(t['price']) for t in trades]
    qtys = [float(t['qty']) for t in trades]
    quote_qtys = [float(t['quoteQty']) for t in trades]
    is_sell = [t['isBuyerMaker'] for t in trades]
    
    avg_qty = np.mean(qtys)
    std_qty = np.std(qtys)
    
    # Whales: trades > 3σ from mean
    whales = []
    for i, t in enumerate(trades):
        z = (qtys[i] - avg_qty) / max(std_qty, 1e-10)
        if z > 2.5:
            whales.append({
                'price': round(prices[i], 2),
                'qty': round(qtys[i], 4),
                'usd': round(quote_qtys[i], 0),
                'side': 'sell' if is_sell[i] else 'buy',
                'zScore': round(float(z), 1),
            })
    
    buy_whales = [w for w in whales if w['side'] == 'buy']
    sell_whales = [w for w in whales if w['side'] == 'sell']
    
    return {
        'totalTrades': len(trades),
        'whaleCount': len(whales),
        'buyWhales': len(buy_whales),
        'sellWhales': len(sell_whales),
        'netWhaleFlow': round(sum(w['usd'] for w in buy_whales) - sum(w['usd'] for w in sell_whales), 0),
        'recentWhales': whales[-5:] if whales else [],
        'threshold': {'avgQty': round(avg_qty, 4), 'stdQty': round(std_qty, 4), 'minWhaleQty': round(avg_qty + 2.5 * std_qty, 4)},
        'interpretation': f'{len(whales)} whale trades (>{avg_qty+2.5*std_qty:.4f} BTC). Net: {"🟢 buying" if sum(w["usd"] for w in buy_whales) > sum(w["usd"] for w in sell_whales) else "🔴 selling"}.',
    }

# ── Full Analysis ──
BUYER_SYMBOLS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'XAUT': 'XAUTUSDT'}

def analyze_all():
    result = {}
    for name, symbol in BUYER_SYMBOLS.items():
        try:
            # Fetch 1H data
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=200"
            with urllib.request.urlopen(url, timeout=10) as r:
                raw = json.loads(r.read())
            h1 = [{'c': float(k[4]), 'o': float(k[1]), 'h': float(k[2]), 'l': float(k[3]), 'v': float(k[5])} for k in raw]
            closes = np.array([k['c'] for k in h1])
            price = closes[-1]
            
            # Markov chain
            marks = markov_chain(closes)
            
            # Volume & thrust
            thrust = volume_thrust(h1)
            
            # Whales
            whales = detect_whales(symbol)
            
            result[name] = {
                'price': round(float(price), 2),
                'markov': marks,
                'volumeDominance': thrust['volumeDominance'] if isinstance(thrust, dict) else {},
                'thrust': thrust['thrust'] if isinstance(thrust, dict) else {},
                'whales': whales,
            }
        except Exception as e:
            result[name] = {'error': str(e)}
    return result

if __name__ == '__main__':
    print(json.dumps(analyze_all(), default=str))
