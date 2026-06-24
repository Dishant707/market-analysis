"""F&O Analysis: Funding rates, basis, Black-Scholes IV, arbitrage strategies."""
import sys, json, urllib.request, numpy as np, datetime, math

FAPI = "https://fapi.binance.com/fapi/v1"

# ── Futures Data ──
def fetch_futures(symbol):
    """Fetch funding rate, mark price, open interest for perpetual futures."""
    result = {'symbol': symbol}
    try:
        # Premium index (funding rate + mark price)
        req = urllib.request.Request(f"{FAPI}/premiumIndex?symbol={symbol}", headers={'User-Agent':'Mozilla/5.0'})
        prem = json.loads(urllib.request.urlopen(req, timeout=10).read())
        result['markPrice'] = float(prem['markPrice'])
        result['indexPrice'] = float(prem['indexPrice'])
        result['fundingRate'] = float(prem['lastFundingRate']) * 100  # in percent
        result['nextFundingTime'] = prem['nextFundingTime']
        
        # Open interest
        req = urllib.request.Request(f"{FAPI}/openInterest?symbol={symbol}", headers={'User-Agent':'Mozilla/5.0'})
        oi = json.loads(urllib.request.urlopen(req, timeout=10).read())
        result['openInterest'] = float(oi['openInterest'])
        result['openInterestUsd'] = result['openInterest'] * result['markPrice']
        
        # Funding rate history (last 100)
        req = urllib.request.Request(f"{FAPI}/fundingRate?symbol={symbol}&limit=100", headers={'User-Agent':'Mozilla/5.0'})
        fr_hist = json.loads(urllib.request.urlopen(req, timeout=10).read())
        fr_vals = [float(fr['fundingRate']) * 100 for fr in fr_hist]
        result['fundingRateAvg'] = round(np.mean(fr_vals), 4)
        result['fundingRateStd'] = round(np.std(fr_vals), 4)
        result['fundingRateMin'] = round(min(fr_vals), 4)
        result['fundingRateMax'] = round(max(fr_vals), 4)
        
        # Basis: futures premium vs spot
        try:
            spot_req = urllib.request.Request(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.replace('USDT','USDT')}", headers={'User-Agent':'Mozilla/5.0'})
            spot = json.loads(urllib.request.urlopen(spot_req, timeout=5).read())
            spot_price = float(spot['price'])
            result['spotPrice'] = spot_price
            result['basis'] = round((result['markPrice'] - spot_price) / spot_price * 100, 4)
            result['annualizedBasis'] = round(result['basis'] * 365, 2)  # rough annualized
        except:
            result['spotPrice'] = result['indexPrice']
            result['basis'] = 0
            result['annualizedBasis'] = 0
            
    except Exception as e:
        result['error'] = str(e)
    return result

# ── Taker Buy/Sell Ratio (order flow) ──
def fetch_taker_flow(symbol):
    """Buy vs sell taker volume ratio — shows who's aggressive."""
    try:
        req = urllib.request.Request(f"{FAPI}/takerlongshortRatio?symbol={symbol}&period=1h&limit=24")
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
        ratios = [(float(item['buyVol']) / max(float(item['sellVol']), 1)) for item in d]
        return {
            'currentRatio': round(float(d[-1]['buyVol']) / max(float(d[-1]['sellVol']), 1), 3) if d else 0,
            'avgRatio': round(float(np.mean(ratios)), 3) if ratios else 0,
            'interpretation': 'buyers aggressive' if ratios and ratios[-1] > 1.1 else ('sellers aggressive' if ratios and ratios[-1] < 0.9 else 'neutral'),
        }
    except:
        return {'error': 'unavailable'}

# ── Black-Scholes (for options) ──
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes pricing model. S=spot, K=strike, T=time(years), r=rate, sigma=vol."""
    if T <= 0 or sigma <= 0:
        return 0, 0, 0, 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Cumulative normal (approximation)
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    if option_type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = -norm_cdf(-d1)
    
    gamma = norm_cdf(d1) / (S * sigma * math.sqrt(T)) if S > 0 and sigma > 0 and T > 0 else 0
    # Convert gamma from pdf: pdf(d1) / (S * sigma * sqrt(T))
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    gamma = pdf_d1 / (S * sigma * math.sqrt(T)) if S > 0 and sigma > 0 and T > 0 else 0
    
    return round(price, 2), round(delta, 4), round(gamma, 6), round(norm_cdf(d1), 4)

def implied_volatility(target_price, S, K, T, r, option_type='call'):
    """Find IV that makes BS price = target price (Newton's method)."""
    if target_price <= 0:
        return 0
    sigma = 0.5  # initial guess
    for _ in range(50):
        price, _, _, _ = black_scholes(S, K, T, r, sigma, option_type)
        vega = S * math.sqrt(T) * math.exp(-0.5 * ((math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T)))**2) / math.sqrt(2*math.pi)
        diff = price - target_price
        if abs(diff) < 0.0001:
            break
        if abs(vega) > 1e-10:
            sigma = sigma - diff / vega
        sigma = max(0.01, min(3.0, sigma))
    return round(sigma * 100, 1)  # return as percent

# ── Put-Call Parity Check ──
def put_call_parity(call_price, put_price, S, K, T, r):
    """P = C - S + K*e^(-rT). If prices deviate, arb exists."""
    parity_put = call_price - S + K * math.exp(-r * T)
    deviation = put_price - parity_put
    arb_pnl = abs(deviation)
    direction = 'short call + long put + long spot' if deviation > 0.001 else ('long call + short put + short spot' if deviation < -0.001 else 'none')
    return {
        'parityPutPrice': round(parity_put, 2),
        'actualPutPrice': round(put_price, 2),
        'deviation': round(deviation, 2),
        'arbPnl': round(arb_pnl, 2),
        'arbDirection': direction,
        'actionable': arb_pnl > 0.5,  # > $0.50 arb
    }

# ── Strategy Analysis ──
def analyze_strategies():
    symbols = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT'}
    results = {}
    
    for name, symbol in symbols.items():
        f = fetch_futures(symbol)
        tf = fetch_taker_flow(symbol)
        
        strategies = []
        
        # 1. Funding rate farming
        fr = f.get('fundingRate', 0)
        fr_avg = f.get('fundingRateAvg', 0)
        if abs(fr) > 0.005:  # > 0.005% per 8h = meaningful
            strategies.append({
                'name': 'Funding Rate Farm',
                'action': 'SHORT' if fr > 0 else 'LONG',
                'detail': f'FR {fr:.4f}%/{fr_avg:.4f}% avg. {"Longs paying shorts" if fr > 0 else "Shorts paying longs"}',
                'annualizedReturn': round(abs(fr) * 3 * 365, 1),  # 3 fundings per day
                'risk': 'mark-to-market loss > funding collected',
            })
        
        # 2. Basis trade (cash-and-carry)
        basis = f.get('basis', 0)
        if abs(basis) > 0.01:
            annual_basis = f.get('annualizedBasis', 0)
            strategies.append({
                'name': 'Cash-and-Carry (Basis)',
                'action': f'{"SHORT futures + LONG spot" if basis > 0 else "LONG futures + SHORT spot"}',
                'detail': f'Basis {basis:.4f}% (annualized {annual_basis:.1f}%)',
                'annualizedReturn': round(abs(f.get('annualizedBasis', 0)), 1),
                'risk': 'funding rate changes, liquidation if undercollateralized',
            })
        
        # 3. Taker flow divergence
        tf_ratio = tf.get('currentRatio', 1)
        if abs(tf_ratio - 1) > 0.15:
            strategies.append({
                'name': 'Order Flow Imbalance',
                'action': 'FOLLOW flow' if tf_ratio > 1 else 'CONTRARIAN',
                'detail': f'Taker ratio {tf_ratio:.2f}x ({(tf_ratio-1)*100:+.0f}% {"buy" if tf_ratio > 1 else "sell"} imbalance)',
                'annualizedReturn': 0,
                'risk': 'mean reversion / flow can persist',
            })
        
        # 4. Open Interest change
        oi = f.get('openInterest', 0)
        oi_usd = f.get('openInterestUsd', 0)
        if oi > 0:
            strategies.append({
                'name': 'OI Monitor',
                'action': 'WATCH',
                'detail': f'OI {oi:,.0f} BTC (${oi_usd/1e9:.2f}B)',
                'annualizedReturn': 0,
                'risk': 'rising OI + falling price = liquidation cascade risk',
            })
        
        results[name] = {
            'futures': {
                'markPrice': f.get('markPrice', 0),
                'fundingRate': f.get('fundingRate', 0),
                'fundingRateAvg': f.get('fundingRateAvg', 0),
                'basis': f.get('basis', 0),
                'annualizedBasis': f.get('annualizedBasis', 0),
                'openInterest': f.get('openInterest', 0),
                'openInterestUsd': f.get('openInterestUsd', 0),
            },
            'takerFlow': tf,
            'strategies': strategies,
            'bestStrategy': max(strategies, key=lambda s: abs(s['annualizedReturn'])) if strategies else None,
        }
    
    return results

if __name__ == '__main__':
    print(json.dumps(analyze_strategies(), default=str))
