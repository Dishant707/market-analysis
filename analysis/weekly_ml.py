"""Weekly BTC prediction model trained on 5-year data."""
import sys, json, urllib.request, numpy as np, datetime, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ── Data Fetching ──
def fetch_btc_5yr():
    """Fetch ~5 years of daily BTC data from Binance via pagination."""
    all_data = []
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    while len(all_data) < 2000:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000&endTime={end_time}"
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                batch = json.loads(r.read())
            if not batch or len(batch) <= 1:
                break
            all_data = [{'t': k[0], 'o': float(k[1]), 'h': float(k[2]), 'l': float(k[3]),
                         'c': float(k[4]), 'v': float(k[5])} for k in batch] + all_data
            end_time = batch[0][0] - 1
        except:
            break
    return all_data

# ── Feature Engineering ──
def engineer_weekly_features(data):
    """Convert daily data to weekly candles and build features."""
    if len(data) < 20:
        return None, None
    
    # Create weekly candles
    weekly = []
    for i in range(0, len(data), 7):
        week = data[i:min(i+7, len(data))]
        if len(week) < 3: continue
        weekly.append({
            'open': week[0]['o'],
            'high': max(k['h'] for k in week),
            'low': min(k['l'] for k in week),
            'close': week[-1]['c'],
            'volume': sum(k['v'] for k in week),
        })
    
    if len(weekly) < 10:
        return None, None
    
    # Build features for each week
    X, y = [], []
    prices = np.array([w['close'] for w in weekly])
    highs = np.array([w['high'] for w in weekly])
    lows = np.array([w['low'] for w in weekly])
    vols = np.array([w['volume'] for w in weekly])
    
    for i in range(8, len(weekly) - 1):
        feats = []
        w = weekly[i]
        price = w['close']
        
        # 1. Weekly return
        feats.append((w['close'] / w['open'] - 1) * 100)
        
        # 2. Weekly range
        feats.append((w['high'] - w['low']) / w['low'] * 100)
        
        # 3. Weekly volume vs average
        avg_vol = np.mean(vols[max(0,i-12):i+1])
        feats.append(w['volume'] / max(avg_vol, 1))
        
        # 4. Volume trend (last 4 weeks vs previous 4)
        vol_4 = np.mean(vols[i-3:i+1])
        vol_8 = np.mean(vols[max(0,i-7):i-3])
        feats.append(vol_4 / max(vol_8, 1))
        
        # 5-9. Previous week returns (mom 1-5)
        for lag in range(1, 6):
            prev_ret = (weekly[i-lag]['close'] / weekly[i-lag]['open'] - 1) * 100 if i >= lag else 0
            feats.append(prev_ret)
        
        # 10. Price vs 8-week MA
        ma8 = np.mean(prices[max(0,i-7):i+1])
        feats.append((price / ma8 - 1) * 100)
        
        # 11. Price vs 21-week MA
        ma21 = np.mean(prices[max(0,i-20):i+1])
        feats.append((price / max(ma21, 1) - 1) * 100)
        
        # 12. Week of month (seasonality)
        week_num = (i % 4) + 1
        feats.append(week_num / 4)  # 0.25, 0.5, 0.75, 1.0
        
        # 13. RSI weekly (14 weeks)
        if i >= 14:
            gains = [prices[j] - prices[j-1] for j in range(i-13, i+1) if prices[j] > prices[j-1]]
            losses = [prices[j-1] - prices[j] for j in range(i-13, i+1) if prices[j] < prices[j-1]]
            avg_g = np.mean(gains) if gains else 0
            avg_l = np.mean(losses) if losses else 0
            rsi = 100 - 100 / (1 + avg_g / max(avg_l, 1))
        else:
            rsi = 50
        feats.append(rsi / 100)  # 0-1 scale
        
        # 14. Volatility trend (current week vol vs 8-week avg)
        cur_vol = (w['high'] - w['low']) / w['low'] * 100
        hist_vol = np.mean([(highs[j] - lows[j]) / lows[j] * 100 for j in range(max(0,i-7), i+1)])
        feats.append(cur_vol / max(hist_vol, 0.01))
        
        # 15. Body ratio (close vs open position in range)
        body = abs(w['close'] - w['open'])
        wk_range = w['high'] - w['low']
        feats.append(body / max(wk_range, 1))
        
        # 16-19. Candle position (where close is relative to open)
        bull = 1 if w['close'] > w['open'] else 0
        feats.append(bull)
        
        # 20. Consecutive weeks up/down
        cons = 0
        for j in range(i-1, max(0, i-5), -1):
            if (weekly[j]['close'] > weekly[j]['open']) == bull:
                cons += 1
            else:
                break
        feats.append(cons / 5)
        
        # Target: next week direction (1 = up, 0 = down)
        next_chg = weekly[i+1]['close'] - weekly[i+1]['open']
        target = 1 if next_chg > 0 else 0
        
        X.append(feats)
        y.append(target)
    
    return np.array(X), np.array(y)

# ── Train Model ──
def train_model(data):
    X, y = engineer_weekly_features(data)
    if X is None or len(X) < 20:
        return {'error': 'insufficient data'}
    
    # Time series split — train on past, test on future
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train
    rf = RandomForestClassifier(n_estimators=1000, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    
    # Test
    train_acc = float(np.mean(rf.predict(X_train_s) == y_train)) * 100
    test_acc = float(np.mean(rf.predict(X_test_s) == y_test)) * 100
    
    # Current week features (last week)
    curr_feats = X[-1:].copy()
    curr_scaled = scaler.transform(curr_feats)
    next_prob = rf.predict_proba(curr_scaled)[0]
    next_pred = int(rf.predict(curr_scaled)[0])
    next_up = round(float(next_prob[1]) * 100, 1)
    next_down = round(float(next_prob[0]) * 100, 1)
    
    # Feature importance
    feature_names = ['weekly_ret', 'weekly_range', 'vol_ratio', 'vol_trend',
                     'mom_1', 'mom_2', 'mom_3', 'mom_4', 'mom_5',
                     'ma8_dist', 'ma21_dist', 'week_of_month', 'rsi', 'vol_trend_ratio',
                     'body_ratio', 'bullish', 'cons_weeks']
    importances = sorted(zip(feature_names[:len(rf.feature_importances_)], rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)[:8]
    
    return {
        'model': {
            'type': 'RandomForest (1000 trees)',
            'features': len(feature_names),
            'trainWeeks': len(X_train),
            'testWeeks': len(X_test),
            'trainAccuracy': round(train_acc, 1),
            'testAccuracy': round(test_acc, 1),
            'featureImportance': [{'feature': f, 'importance': round(i, 4)} for f, i in importances],
        },
        'prediction': {
            'nextWeek': 'UP' if next_pred == 1 else 'DOWN',
            'upProb': next_up,
            'downProb': next_down,
            'confidence': 'high' if max(next_up, next_down) > 70 else ('moderate' if max(next_up, next_down) > 55 else 'low'),
        },
        'interpretation': (
            f'Trained on {len(X_train)} weeks, tested on {len(X_test)} weeks. '
            f'Test accuracy: {test_acc:.1f}%. '
            f'Next week predicted: {"UP" if next_pred == 1 else "DOWN"} ({max(next_up, next_down):.0f}% confidence). '
            f'Top features: {importances[0][0]} ({importances[0][1]:.2f}), {importances[1][0]} ({importances[1][1]:.2f}).'
        ),
    }

if __name__ == '__main__':
    data = fetch_btc_5yr()
    result = train_model(data)
    result['dataPoints'] = len(data)
    result['weeksGenerated'] = len(data) // 7
    print(json.dumps(result, default=str))
