"""Technical & mathematical analysis per asset: RSI, MACD, BB, patterns, geometry."""
import sys, json, urllib.request, numpy as np, datetime

# ── Data ──
def fetch(symbol):
    d1 = json.loads(urllib.request.urlopen(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=200").read())
    h1 = json.loads(urllib.request.urlopen(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=168").read())
    def parse(data):
        return {'c': np.array([float(k[4]) for k in data]), 'h': np.array([float(k[2]) for k in data]),
                'l': np.array([float(k[3]) for k in data]), 'o': np.array([float(k[1]) for k in data]),
                'v': np.array([float(k[5]) for k in data])}
    return parse(d1), parse(h1)

# ── Technical Indicators ──
def rsi(c, period=14):
    deltas = np.diff(c)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gain[:period])
    avg_l = np.mean(loss[:period])
    rs = []
    for i in range(len(c)):
        if i < period:
            rs.append(None)
        elif i == period:
            rs.append(100 - 100 / (1 + avg_g / max(avg_l, 1e-10)))
        else:
            avg_g = (avg_g * (period - 1) + gain[i-1]) / period
            avg_l = (avg_l * (period - 1) + loss[i-1]) / period
            rs.append(100 - 100 / (1 + avg_g / max(avg_l, 1e-10)))
    return rs

def ema(data, period):
    r = [None] * (period - 1)
    mult = 2 / (period + 1)
    prev = sum(data[:period]) / period
    r.append(prev)
    for i in range(period, len(data)):
        prev = (data[i] - prev) * mult + prev
        r.append(prev)
    return r

def macd(c):
    e12 = ema(c, 12); e26 = ema(c, 26)
    macd_line = []
    for i in range(len(c)):
        if e12[i] is None or e26[i] is None:
            macd_line.append(None)
        else:
            macd_line.append(e12[i] - e26[i])
    valid = [x for x in macd_line if x is not None]
    sig_raw = ema(valid, 9)
    sig_vals = [x for x in sig_raw if x is not None]  # Remove EMA's leading Nones
    signal = [None] * len(macd_line)
    hist = [None] * len(macd_line)
    si = 0
    for i in range(len(macd_line)):
        if macd_line[i] is not None and si < len(sig_vals):
            signal[i] = sig_vals[si]
            hist[i] = macd_line[i] - sig_vals[si]
            si += 1
    return {'macd': macd_line, 'signal': signal, 'histogram': hist}

def composite_market_indicator(c, o, h, l, v, price):
    """
    Custom Composite Market Indicator (CMI): combines VWAP, volume dominance,
    CVD trend, RSI zone, and channel position into a single -100 to +100 score.
    Positive = bullish, Negative = bearish.
    """
    # 1. VWAP position
    cum_pv = np.cumsum(c * v)
    cum_v = np.cumsum(v)
    vwap = cum_pv / cum_v
    vwap_dist = (price - vwap[-1]) / vwap[-1] * 100
    vwap_score = max(-30, min(30, vwap_dist * 10))  # -30 to +30
    
    # 2. Volume dominance (last 48h)
    h48 = c[-48:] if len(c) >= 48 else c
    o48 = o[-48:] if len(o) >= 48 else o
    v48 = v[-48:] if len(v) >= 48 else v
    buy_v = sum(v48[i] for i in range(len(h48)) if h48[i] >= o48[i])
    sell_v = sum(v48[i] for i in range(len(h48)) if h48[i] < o48[i])
    vol_ratio = (buy_v - sell_v) / (buy_v + sell_v) * 100 if (buy_v + sell_v) > 0 else 0
    vol_score = max(-20, min(20, vol_ratio))  # -20 to +20
    
    # 3. CVD trend (cumulative volume delta direction)
    deltas = [(c[i] - o[i]) / (h[i] - l[i] + 1e-10) * v[i] for i in range(len(c))]
    cvd = np.cumsum(deltas)
    cvd_slope = (cvd[-1] - cvd[-10]) / max(abs(cvd[-10]), 1) * 100 if len(cvd) >= 10 else 0
    cvd_score = max(-20, min(20, cvd_slope))  # -20 to +20
    
    # 4. RSI zone
    rsi_vals = rsi(c, 14)
    rsi_last = [x for x in rsi_vals if x is not None]
    rsi_val = rsi_last[-1] if rsi_last else 50
    if rsi_val > 70: rsi_score = -15  # Overbought
    elif rsi_val > 60: rsi_score = -5   # Slightly overbought
    elif rsi_val > 55: rsi_score = 0
    elif rsi_val > 40: rsi_score = 0
    elif rsi_val > 30: rsi_score = 5
    else: rsi_score = 15  # Oversold
    rsi_score = max(-15, min(15, rsi_score))
    
    # 5. Channel position (where is price in recent range)
    recent_high = max(c[-20:]) if len(c) >= 20 else max(c)
    recent_low = min(c[-20:]) if len(c) >= 20 else min(c)
    ch_pos = (price - recent_low) / (recent_high - recent_low) * 100 if (recent_high - recent_low) > 0 else 50
    if ch_pos > 80: ch_score = -15  # Top of range
    elif ch_pos < 20: ch_score = 15  # Bottom of range
    else: ch_score = 0
    
    # Composite: -100 to +100
    total = vwap_score + vol_score + cvd_score + rsi_score + ch_score
    total = max(-100, min(100, total))
    
    # Interpretation
    if total > 40: signal = 'STRONG BUY'; color = '#26a69a'
    elif total > 15: signal = 'BUY'; color = '#26a69a'
    elif total > -15: signal = 'NEUTRAL'; color = '#f0c040'
    elif total > -40: signal = 'SELL'; color = '#ef5350'
    else: signal = 'STRONG SELL'; color = '#ef5350'
    
    return {
        'cmi': round(total, 1),
        'signal': signal,
        'signalColor': color,
        'components': {
            'vwapScore': round(vwap_score, 1),
            'volScore': round(vol_score, 1),
            'cvdScore': round(cvd_score, 1),
            'rsiScore': round(rsi_score, 1),
            'channelScore': round(ch_score, 1),
        },
        'vwapDist': round(vwap_dist, 2),
        'volRatio': round(vol_ratio, 1),
        'cvdSlope': round(cvd_slope, 1),
        'channelPos': round(ch_pos, 1),
    }

def volume_analysis(o, c, v):
    up_vol = sum(v[i] for i in range(len(c)) if c[i] >= o[i])
    dn_vol = sum(v[i] for i in range(len(c)) if c[i] < o[i])
    return {'buyRatio': round(up_vol/(up_vol+dn_vol)*100, 1) if (up_vol+dn_vol)>0 else 0,
            'sellRatio': round(dn_vol/(up_vol+dn_vol)*100, 1) if (up_vol+dn_vol)>0 else 0}

# ── Candlestick Pattern Detection ──
def detect_patterns(c, o, h, l):
    patterns = []
    for i in range(2, len(c)):
        body = abs(c[i]-o[i]); rg = h[i]-l[i]; prev_body = abs(c[i-1]-o[i-1]); prev_rg = h[i-1]-l[i-1]
        if rg == 0: continue
        body_ratio = body / rg
        up = c[i] > o[i]; dn = c[i] < o[i]
        # Doji
        if body_ratio < 0.1 and rg > 0:
            patterns.append({'type': 'Doji', 'dir': 'neutral', 'sig': 'low', 'price': round(c[i],2)})
        # Hammer
        if body_ratio < 0.4 and (l[i] < o[i] and l[i] < c[i]) and (h[i] - max(c[i], o[i])) < body * 0.3:
            patterns.append({'type': 'Hammer', 'dir': 'bullish', 'sig': 'high', 'price': round(c[i],2)})
        # Shooting Star
        if body_ratio < 0.4 and (h[i] > o[i] and h[i] > c[i]) and (min(c[i], o[i]) - l[i]) < body * 0.3:
            patterns.append({'type': 'Shooting Star', 'dir': 'bearish', 'sig': 'high', 'price': round(c[i],2)})
        # Engulfing
        if up and c[i-1] < o[i-1] and c[i] > o[i-1] and o[i] < c[i-1]:
            patterns.append({'type': 'Bullish Engulfing', 'dir': 'bullish', 'sig': 'high', 'price': round(c[i],2)})
        if dn and c[i-1] > o[i-1] and c[i] < o[i-1] and o[i] > c[i-1]:
            patterns.append({'type': 'Bearish Engulfing', 'dir': 'bearish', 'sig': 'high', 'price': round(c[i],2)})
        # Morning/Evening Star (3-candle)
        if i >= 2 and up and c[i-1] < o[i-1] and c[i-2] > o[i-2] and c[i] > (c[i-2]+o[i-2])/2:
            patterns.append({'type': 'Morning Star', 'dir': 'bullish', 'sig': 'very high', 'price': round(c[i],2)})
        if i >= 2 and dn and c[i-1] > o[i-1] and c[i-2] < o[i-2] and c[i] < (c[i-2]+o[i-2])/2:
            patterns.append({'type': 'Evening Star', 'dir': 'bearish', 'sig': 'very high', 'price': round(c[i],2)})
        # Harami
        if up and c[i-1] < o[i-1] and c[i] < o[i-1] and o[i] > c[i-1]:
            patterns.append({'type': 'Bullish Harami', 'dir': 'bullish', 'sig': 'moderate', 'price': round(c[i],2)})
        if dn and c[i-1] > o[i-1] and c[i] > o[i-1] and o[i] < c[i-1]:
            patterns.append({'type': 'Bearish Harami', 'dir': 'bearish', 'sig': 'moderate', 'price': round(c[i],2)})
    return patterns[-5:] if len(patterns) > 5 else patterns

# ── Geometry / Channel Analysis ──
def geometry(c, h, l):
    price = c[-1]
    # Trend angle (last 48 hours using 1H should be passed as daily)
    x = np.arange(len(c))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, c, rcond=None)[0]
    angle = float(np.degrees(np.arctan(slope)))
    
    # Channel (last 20 candles)
    ch = c[-20:] if len(c) >= 20 else c
    ch_x = np.arange(len(ch))
    ch_A = np.vstack([ch_x, np.ones(len(ch_x))]).T
    ch_slope, ch_int = np.linalg.lstsq(ch_A, ch, rcond=None)[0]
    ch_trend = ch_slope * ch_x + ch_int
    ch_upper = float(np.max(h[-20:] - ch_trend) if len(h) >= 20 else 0)
    ch_lower = float(np.min(l[-20:] - ch_trend) if len(l) >= 20 else 0)
    ch_width = ch_upper - ch_lower
    ch_width_pct = ch_width / price * 100 if price > 0 else 0
    
    # Fibonacci from recent swing
    recent_high = float(np.max(c[-30:]))
    recent_low = float(np.min(c[-30:]))
    fib_range = recent_high - recent_low
    fibs = {}
    for ratio, label in [(0.236,'23.6'),(0.382,'38.2'),(0.5,'50'),(0.618,'61.8')]:
        level = recent_high - fib_range * ratio
        fibs[label] = round(level, 2)
    
    return {
        'angle': round(angle, 2),
        'channelWidthPct': round(ch_width_pct, 2),
        'channelUpper': round(float(np.mean(ch_trend + ch_upper)), 2) if len(c) >= 20 else 0,
        'channelLower': round(float(np.mean(ch_trend + ch_lower)), 2) if len(c) >= 20 else 0,
        'fibonacci': fibs,
        'swing': {'high': round(recent_high, 2), 'low': round(recent_low, 2)},
    }

# ── Main ──
BUYER_SYMBOLS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'XAUT': 'XAUTUSDT'}

def analyze_all():
    result = {}
    for name, symbol in BUYER_SYMBOLS.items():
        try:
            d1, h1 = fetch(symbol)
            c, o, h, l, v = h1['c'], h1['o'], h1['h'], h1['l'], h1['v']
            price = float(c[-1])
            
            rsi_vals = [x for x in rsi(c, 14) if x is not None]
            macd_vals = macd(c)
            cmi = composite_market_indicator(c, o, h, l, v, price)
            vol = volume_analysis(o, c, v)
            patterns = detect_patterns(c, o, h, l)
            geo = geometry(c, h, l)
            
            # RSI/MACD divergence
            rsi_last = rsi_vals[-1] if rsi_vals else 50
            rsi_prev = rsi_vals[-5] if len(rsi_vals) >= 5 else rsi_last
            price_last = c[-1]; price_5 = c[-5] if len(c) >= 5 else price_last
            div = None
            if price_last > price_5 and rsi_last < rsi_prev - 5:
                div = 'bearish divergence (price up, RSI down)'
            elif price_last < price_5 and rsi_last > rsi_prev + 5:
                div = 'bullish divergence (price down, RSI up)'
            
            result[name] = {
                'price': round(price, 2),
                'rsi': round(rsi_last, 1),
                'macd': {'histogram': round(macd_vals['histogram'][-1], 2) if macd_vals['histogram'][-1] is not None else 0,
                         'signal': 0, 'macd': round(macd_vals['macd'][-1], 2) if macd_vals['macd'][-1] is not None else 0},
                'cmi': cmi,
                'volume': vol,
                'patterns': patterns,
                'geometry': geo,
                'divergence': div,
                'macdHist': {'value': round(macd_vals['histogram'][-1], 2) if macd_vals['histogram'][-1] is not None else 0,
                             'direction': 'rising' if len(macd_vals['histogram']) >= 2 and macd_vals['histogram'][-1] is not None and macd_vals['histogram'][-2] is not None and macd_vals['histogram'][-1] > macd_vals['histogram'][-2] else 'falling'},
            }
        except Exception as e:
            result[name] = {'error': str(e)}
    return result

if __name__ == '__main__':
    print(json.dumps(analyze_all(), default=str))
