use rayon::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Write, BufRead};
use std::time::Instant;

// ─── Request/Response Types ───

#[derive(Deserialize)]
struct Request { action: String, data: serde_json::Value }

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Response {
    orderflow: Option<OrderFlowResult>,
    regimes: Option<RegimeResult>,
    features: Option<FeatureResult>,
    bayesian: Option<BayesianResult>,
    options: Option<OptionSellResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing: Option<f64>,
}

// ═══════════════════════════════════════════════
// MODULE 1: ORDER FLOW / MARKET MICROSTRUCTURE
// ═══════════════════════════════════════════════

#[derive(Deserialize)]
struct OrderFlowRequest {
    trades: Vec<TradeTick>,
    book_bids: Vec<[f64; 2]>,
    book_asks: Vec<[f64; 2]>,
    klines: Vec<Candle>,
    price: f64,
}

#[derive(Deserialize)]
struct TradeTick {
    price: f64,
    qty: f64,
    side: String, // "buy" or "sell"
    time: i64,
}

#[derive(Deserialize, Clone)]
struct Candle {
    o: f64, h: f64, l: f64, c: f64, v: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OrderFlowResult {
    cvd: f64,
    cvd_slope: f64,
    buy_volume_pct: f64,
    sell_volume_pct: f64,
    trade_imbalance: f64,
    aggression_ratio: f64,
    whale_trades: Vec<WhaleTrade>,
    absorption_candles: Vec<AbsorptionCandle>,
    book_imbalance: f64,
    book_pressure: HashMap<String, f64>,
    bid_walls: Vec<[f64; 2]>,
    ask_walls: Vec<[f64; 2]>,
    vwap: f64,
    vwap_distance_pct: f64,
    delta_divergence: bool,
    interpretation: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WhaleTrade { price: f64, qty: f64, usd: f64, side: String, zscore: f64 }

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AbsorptionCandle { idx: usize, vol_ratio: f64, range_pct: f64, side: String, strength: f64 }

fn analyze_orderflow(req: &OrderFlowRequest) -> OrderFlowResult {
    let trades = &req.trades;
    let bids = &req.book_bids;
    let asks = &req.book_asks;
    let klines = &req.klines;
    let price = req.price;

    // 1. CVD (Cumulative Volume Delta)
    let mut cvd = 0.0_f64;
    let mut buy_vol = 0.0_f64;
    let mut sell_vol = 0.0_f64;
    let mut trade_values: Vec<f64> = Vec::new();
    let mut cvd_history: Vec<f64> = Vec::new();

    for t in trades {
        let val = t.price * t.qty;
        trade_values.push(val);
        if t.side == "buy" { cvd += val; buy_vol += val; }
        else { cvd -= val; sell_vol += val; }
        cvd_history.push(cvd);
    }
    let total_vol = buy_vol + sell_vol;
    let buy_pct = if total_vol > 0.0 { buy_vol / total_vol * 100.0 } else { 50.0 };
    let trade_imb = if total_vol > 0.0 { (buy_vol - sell_vol) / total_vol * 100.0 } else { 0.0 };

    // CVD slope (last 20% of trades)
    let cvd_slope = if cvd_history.len() > 10 {
        let n = cvd_history.len();
        let start = cvd_history[n / 5 * 4].max(1.0);
        (cvd_history[n - 1] - start) / start * 100.0
    } else { 0.0 };

    // 2. Aggression ratio (how aggressive is the buying)
    let aggression = if trades.len() > 5 {
        let recent: Vec<&TradeTick> = trades.iter().rev().take(20).collect();
        let agg_buys = recent.iter().filter(|t| t.side == "buy").count();
        agg_buys as f64 / recent.len() as f64 * 100.0
    } else { 50.0 };

    // 3. Whale detection (trades > 3 sigma)
    let (whales, _) = if trade_values.len() > 5 {
        let mean = trade_values.iter().sum::<f64>() / trade_values.len() as f64;
        let var = trade_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / trade_values.len() as f64;
        let std = var.sqrt().max(1.0);
        let w: Vec<WhaleTrade> = trades.iter().zip(trade_values.iter())
            .filter(|(_, &v)| v > mean + 2.5 * std)
            .map(|(t, &v)| WhaleTrade {
                price: t.price, qty: t.qty, usd: v,
                side: t.side.clone(),
                zscore: ((v - mean) / std * 100.0).round() / 100.0,
            }).collect();
        (w, Some(mean))
    } else { (vec![], None) };

    // 4. Absorption detection
    let abs_candles = if klines.len() > 5 {
        let avg_vol = klines.iter().map(|k| k.v).sum::<f64>() / klines.len() as f64;
        let avg_range = klines.iter().map(|k| (k.h - k.l) / k.c).sum::<f64>() / klines.len() as f64;
        klines.iter().enumerate().filter_map(|(i, k)| {
            let vr = if avg_vol > 0.0 { k.v / avg_vol } else { 0.0 };
            let rp = if k.c > 0.0 { (k.h - k.l) / k.c * 100.0 } else { 0.0 };
            let arp = avg_range * 100.0;
            if vr > 1.5 && rp < arp * 0.5 {
                Some(AbsorptionCandle {
                    idx: i, vol_ratio: (vr * 100.0).round() / 100.0,
                    range_pct: (rp * 1000.0).round() / 1000.0,
                    side: if k.c >= k.o { "seller_absorption".into() } else { "buyer_absorption".into() },
                    strength: (vr * (1.0 - rp / arp.max(0.01)) * 100.0).round() / 100.0,
                })
            } else { None }
        }).collect()
    } else { vec![] };

    // 5. Order book imbalance at multiple depths
    let mut book_pressure = HashMap::new();
    for &depth_pct in &[0.1, 0.25, 0.5, 1.0, 2.0] {
        let dist = price * depth_pct / 100.0;
        let bd: f64 = bids.iter().filter(|b| b[0] >= price - dist).map(|b| b[0] * b[1]).sum();
        let ad: f64 = asks.iter().filter(|a| a[0] <= price + dist).map(|a| a[0] * a[1]).sum();
        let ratio = if ad > 0.0 { bd / ad } else { 1.0 };
        let imb = if bd + ad > 0.0 { (bd - ad) / (bd + ad) * 100.0 } else { 0.0 };
        book_pressure.insert(format!("{}%_bid_depth", depth_pct), (bd * 100.0).round() / 100.0);
        book_pressure.insert(format!("{}%_ask_depth", depth_pct), (ad * 100.0).round() / 100.0);
        book_pressure.insert(format!("{}%_ratio", depth_pct), (ratio * 1000.0).round() / 1000.0);
        book_pressure.insert(format!("{}%_imbalance", depth_pct), (imb * 100.0).round() / 100.0);
    }

    // 6. Book imbalance at top
    let top_bid_vol: f64 = bids.iter().take(10).map(|b| b[1]).sum();
    let top_ask_vol: f64 = asks.iter().take(10).map(|a| a[1]).sum();
    let book_imb = if top_bid_vol + top_ask_vol > 0.0 {
        (top_bid_vol - top_ask_vol) / (top_bid_vol + top_ask_vol) * 100.0
    } else { 0.0 };

    // 7. Walls
    let total_bid = bids.iter().map(|b| b[1]).sum::<f64>().max(1.0);
    let total_ask = asks.iter().map(|a| a[1]).sum::<f64>().max(1.0);
    let bid_walls: Vec<[f64; 2]> = bids.iter().filter(|b| b[1] > total_bid * 0.15).copied().take(3).collect();
    let ask_walls: Vec<[f64; 2]> = asks.iter().filter(|a| a[1] > total_ask * 0.15).copied().take(3).collect();

    // 8. VWAP
    let (vwap, vwap_dist) = if !klines.is_empty() {
        let mut cum_pv = 0.0_f64; let mut cum_v = 0.0_f64;
        for k in klines {
            let typical = (k.o + k.h + k.l + k.c) / 4.0;
            cum_pv += typical * k.v; cum_v += k.v;
        }
        let v = if cum_v > 0.0 { cum_pv / cum_v } else { price };
        let d = if v > 0.0 { (price - v) / v * 100.0 } else { 0.0 };
        ((v * 100.0).round() / 100.0, (d * 100.0).round() / 100.0)
    } else { (price, 0.0) };

    // 9. Delta divergence (CVD direction vs price direction)
    let price_chg = if klines.len() > 5 {
        (klines[klines.len() - 1].c - klines[klines.len() - 6].c) / klines[klines.len() - 6].c * 100.0
    } else { 0.0 };
    let delta_div = (price_chg > 0.5 && cvd_slope < -2.0) || (price_chg < -0.5 && cvd_slope > 2.0);

    // Interpretation
    let mut parts: Vec<String> = Vec::new();
    if book_imb > 10.0 { parts.push(format!("Bid-heavy book (+{:.0}% imbalance) — buyers defending", book_imb)); }
    else if book_imb < -10.0 { parts.push(format!("Ask-heavy book ({:.0}% imbalance) — sellers pressing", book_imb)); }
    else { parts.push(format!("Book balanced ({:.0}% imbalance)", book_imb)); }

    if !whales.is_empty() { parts.push(format!("{} whale trades detected", whales.len())); }
    if !abs_candles.is_empty() { parts.push(format!("{} absorption candles — smart money active", abs_candles.len())); }
    if delta_div { parts.push("⚠ CVD-Price divergence — trend weakening".into()); }

    let buy_aggression = aggression;
    if buy_aggression > 60.0 { parts.push(format!("Aggressive buying ({:.0}% of recent trades)", buy_aggression)); }
    else if buy_aggression < 40.0 { parts.push(format!("Aggressive selling ({:.0}% sells)", 100.0 - buy_aggression)); }

    let flow_interpretation = if parts.is_empty() { "Insufficient data for flow analysis".into() } else { parts.join(". ") };

    OrderFlowResult {
        cvd: (cvd * 100.0).round() / 100.0,
        cvd_slope: (cvd_slope * 100.0).round() / 100.0,
        buy_volume_pct: (buy_pct * 100.0).round() / 100.0,
        sell_volume_pct: ((100.0 - buy_pct) * 100.0).round() / 100.0,
        trade_imbalance: (trade_imb * 100.0).round() / 100.0,
        aggression_ratio: (aggression * 100.0).round() / 100.0,
        whale_trades: whales,
        absorption_candles: abs_candles,
        book_imbalance: (book_imb * 100.0).round() / 100.0,
        book_pressure,
        bid_walls,
        ask_walls,
        vwap,
        vwap_distance_pct: vwap_dist,
        delta_divergence: delta_div,
        interpretation: flow_interpretation,
    }
}

// ═══════════════════════════════════════════════
// MODULE 2: REGIME DETECTION (HIDDEN MARKOV)
// ═══════════════════════════════════════════════

#[derive(Deserialize)]
struct RegimeRequest {
    closes: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    volumes: Vec<f64>,
    num_regimes: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RegimeResult {
    current_regime: usize,
    regime_label: String,
    regime_probabilities: Vec<f64>,
    transition_matrix: Vec<Vec<f64>>,
    stationary_dist: Vec<f64>,
    persistence: f64,
    next_regime: usize,
    regime_history: Vec<usize>,
    volatility_by_regime: Vec<f64>,
    return_by_regime: Vec<f64>,
    interpretation: String,
}

fn detect_regimes(req: &RegimeRequest) -> RegimeResult {
    let n = req.closes.len();
    let k = req.num_regimes.unwrap_or(3).max(2).min(5);
    let r: Vec<f64> = req.closes.windows(2).map(|w| (w[1] - w[0]) / w[0] * 100.0).collect();
    let m = r.len();
    if m < 50 {
        return RegimeResult {
            current_regime: 0, regime_label: "INSUFFICIENT_DATA".into(),
            regime_probabilities: vec![], transition_matrix: vec![],
            stationary_dist: vec![], persistence: 0.0, next_regime: 0,
            regime_history: vec![], volatility_by_regime: vec![],
            return_by_regime: vec![],
            interpretation: "Need at least 50 data points".into(),
        };
    }

    // Feature extraction for regime classification
    let window = 20usize;
    let mut features: Vec<[f64; 4]> = Vec::new();
    for i in window..m {
        let slice_r = &r[i - window..i];
        let mean_r = slice_r.iter().sum::<f64>() / window as f64;
        let var_r = slice_r.iter().map(|x| (x - mean_r).powi(2)).sum::<f64>() / window as f64;
        let std_r = var_r.sqrt();

        let avg_range = if i < req.highs.len().min(req.lows.len()) {
            let h = &req.highs[i - window..i];
            let l = &req.lows[i - window..i];
            if h.len() == window && l.len() == window {
                h.iter().zip(l.iter()).map(|(h, l)| (h - l) / l.max(0.01) * 100.0).sum::<f64>() / window as f64
            } else { std_r * 2.0 }
        } else { std_r * 2.0 };

        let vol_ratio = if i < req.volumes.len() {
            let end = i.min(req.volumes.len());
            let start = i - window.min(i);
            let len = (end - start).max(1);
            let slice_v_avg = req.volumes[start..end].iter().sum::<f64>() / len as f64;
            if end > window + 20 && req.volumes.len() > 20 {
                let prev_start = (i - window - 20).max(0).min(req.volumes.len());
                let prev = req.volumes[prev_start..end - window].iter().sum::<f64>() / 20.0_f64.max(1.0);
                slice_v_avg / prev.max(0.01)
            } else { 1.0 }
        } else { 1.0 };

        features.push([mean_r, std_r * 100.0, avg_range, vol_ratio]);
    }

    let f_n = features.len();

    // K-means clustering (simplified — 5 iterations)
    let mut centroids: Vec<[f64; 4]> = (0..k).map(|i| features[i * f_n / k]).collect();
    for _ in 0..10 {
        let mut sums: Vec<[f64; 4]> = vec![[0.0; 4]; k];
        let mut counts = vec![0usize; k];
        for f in &features {
            let mut best = 0usize;
            let mut best_d = f64::MAX;
            for (j, c) in centroids.iter().enumerate() {
                let d = f.iter().zip(c.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                if d < best_d { best_d = d; best = j; }
            }
            for dim in 0..4 { sums[best][dim] += f[dim]; }
            counts[best] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                for dim in 0..4 { centroids[j][dim] = sums[j][dim] / counts[j] as f64; }
            }
        }
    }

    // Assign regimes to each point
    let mut assignments: Vec<usize> = features.iter().map(|f| {
        let mut best = 0usize;
        let mut best_d = f64::MAX;
        for (j, c) in centroids.iter().enumerate() {
            let d = f.iter().zip(c.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
            if d < best_d { best_d = d; best = j; }
        }
        best
    }).collect();

    // Label regimes by volatility (0 = lowest vol, k-1 = highest vol)
    let mut vol_by_regime: Vec<f64> = (0..k).map(|j| {
        let indices: Vec<usize> = assignments.iter().enumerate().filter(|(_, &a)| a == j).map(|(i, _)| i).collect();
        if indices.is_empty() { return 0.0; }
        indices.iter().map(|&i| features[i][1]).sum::<f64>() / indices.len() as f64
    }).collect();

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| vol_by_regime[a].partial_cmp(&vol_by_regime[b]).unwrap());

    let mut label_map: Vec<usize> = vec![0; k];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        label_map[old_idx] = new_idx;
    }
    for a in assignments.iter_mut() { *a = label_map[*a]; }
    vol_by_regime.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Transition matrix
    let mut trans = vec![vec![0.0_f64; k]; k];
    for i in 1..assignments.len() {
        trans[assignments[i - 1]][assignments[i]] += 1.0;
    }
    for i in 0..k {
        let row_sum: f64 = trans[i].iter().sum();
        if row_sum > 0.0 { for j in 0..k { trans[i][j] /= row_sum; } }
    }

    // Stationary distribution (eigenvector iteration)
    let mut pi = vec![1.0 / k as f64; k];
    for _ in 0..100 {
        let mut pi_new = vec![0.0; k];
        for i in 0..k { for j in 0..k { pi_new[j] += pi[i] * trans[i][j]; } }
        let diff: f64 = pi.iter().zip(pi_new.iter()).map(|(a, b)| (a - b).abs()).sum();
        pi = pi_new;
        if diff < 1e-6 { break; }
    }

    // Current regime & persistence
    let current = assignments[assignments.len() - 1];
    let mut persistence = 0.0_f64;
    for i in (1..assignments.len()).rev() {
        if assignments[i] == current && assignments[i - 1] == current { persistence += 1.0; }
        else { break; }
    }
    let persistence_pct = if assignments.len() > 20 { persistence / 20.0 * 100.0 } else { 0.0 };

    // Next most likely regime
    let next = (0..k).max_by(|&a, &b| trans[current][a].partial_cmp(&trans[current][b]).unwrap()).unwrap_or(current);

    // Return by regime
    let ret_by_reg: Vec<f64> = (0..k).map(|j| {
        let indices: Vec<usize> = assignments.iter().enumerate().filter(|(_, &a)| a == j).map(|(i, _)| i).collect();
        if indices.is_empty() { return 0.0; }
        indices.iter().map(|&i| r[i + window]).sum::<f64>() / indices.len() as f64
    }).collect();

    let labels = ["RANGE_LOW_VOL", "RANGE", "HIGH_VOL_TRENDING"];
    let current_label = if current < labels.len() { labels[current].into() } else { format!("REGIME_{}", current) };

    let interp = format!(
        "Current regime: {} (regime {}). Persistence: {:.0} candles ({:.0}%). Volatility: {:.3}%. Next most likely: regime {}. Stationary: {:.1}%/{:.1}%/{}.",
        current_label, current, persistence, persistence_pct, vol_by_regime[current],
        next,
        pi[0] * 100.0, pi.get(1).unwrap_or(&0.0) * 100.0,
        if k > 2 { format!("{:.1}%", pi[2] * 100.0) } else { "".into() }
    );

    RegimeResult {
        current_regime: current,
        regime_label: current_label,
        regime_probabilities: pi.iter().map(|p| (p * 10000.0).round() / 100.0).collect(),
        transition_matrix: trans.iter().map(|row| row.iter().map(|v| (v * 1000.0).round() / 1000.0).collect()).collect(),
        stationary_dist: pi.iter().map(|p| (p * 10000.0).round() / 100.0).collect(),
        persistence: (persistence_pct * 100.0).round() / 100.0,
        next_regime: next,
        regime_history: assignments.iter().skip(assignments.len().saturating_sub(50)).copied().collect(),
        volatility_by_regime: vol_by_regime.iter().map(|v| (v * 10000.0).round() / 10000.0).collect(),
        return_by_regime: ret_by_reg.iter().map(|v| (v * 1000.0).round() / 1000.0).collect(),
        interpretation: interp,
    }
}

// ═══════════════════════════════════════════════
// MODULE 3: FEATURE ENGINEERING
// ═══════════════════════════════════════════════

#[derive(Deserialize)]
struct FeatureRequest {
    closes: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    volumes: Vec<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FeatureResult {
    features: HashMap<String, f64>,
    top_features: Vec<FeatureImportance>,
    entropy: f64,
    fractal_dim: f64,
    efficiency_ratio: f64,
    herfindahl: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FeatureImportance { name: String, value: f64, category: String }

fn engineer_features(req: &FeatureRequest) -> FeatureResult {
    let n = req.closes.len();
    let c = &req.closes;
    let h = &req.highs;
    let l = &req.lows;
    let v = &req.volumes;
    let mut feats: HashMap<String, f64> = HashMap::new();

    if n < 20 { return FeatureResult {
        features: HashMap::new(), top_features: vec![], entropy: 0.0, fractal_dim: 0.0,
        efficiency_ratio: 0.0, herfindahl: 0.0,
    }; }

    // Returns-based features
    let r: Vec<f64> = c.windows(2).map(|w| (w[1] - w[0]) / w[0] * 100.0).collect();
    let mean_r = r.iter().sum::<f64>() / r.len() as f64;
    let var_r = r.iter().map(|x| (x - mean_r).powi(2)).sum::<f64>() / r.len() as f64;
    let std_r = var_r.sqrt();

    let recent_r: Vec<f64> = r.iter().rev().take(20).cloned().collect();
    let recent_mean = recent_r.iter().sum::<f64>() / recent_r.len() as f64;

    feats.insert("return_mean_20".into(), (mean_r * 10000.0).round() / 100.0);
    feats.insert("return_std_20".into(), (std_r * 10000.0).round() / 100.0);
    feats.insert("return_skew".into(), {
        let s: f64 = r.iter().map(|x| ((x - mean_r) / std_r.max(0.001)).powi(3)).sum::<f64>() / r.len() as f64;
        (s * 1000.0).round() / 1000.0
    });
    feats.insert("return_kurtosis".into(), {
        let k: f64 = r.iter().map(|x| ((x - mean_r) / std_r.max(0.001)).powi(4)).sum::<f64>() / r.len() as f64;
        (k * 1000.0).round() / 1000.0
    });
    feats.insert("recent_return".into(), (recent_mean * 10000.0).round() / 100.0);
    feats.insert("negative_ratio".into(), {
        let neg = r.iter().filter(|&&x| x < 0.0).count();
        (neg as f64 / r.len() as f64 * 10000.0).round() / 100.0
    });
    feats.insert("positive_ratio".into(), {
        let pos = r.iter().filter(|&&x| x > 0.0).count();
        (pos as f64 / r.len() as f64 * 10000.0).round() / 100.0
    });

    // Volatility features
    let ranges: Vec<f64> = (0..n).map(|i| (h[i] - l[i]) / c[i].max(0.01) * 100.0).collect();
    let avg_range = ranges.iter().sum::<f64>() / ranges.len() as f64;
    feats.insert("avg_candle_range_pct".into(), (avg_range * 1000.0).round() / 1000.0);

    let recent_ranges: Vec<f64> = ranges.iter().rev().take(10).cloned().collect();
    let recent_avg_range = recent_ranges.iter().sum::<f64>() / recent_ranges.len() as f64;
    feats.insert("recent_range_vs_avg".into(), ((recent_avg_range / avg_range.max(0.001)) * 1000.0).round() / 1000.0);

    // Body ratio
    let bodies: Vec<f64> = (0..n).map(|i| (c[i] - req.closes.get(i.saturating_sub(1)).copied().unwrap_or(c[i])).abs() / (h[i] - l[i]).max(0.01)).collect();
    let avg_body = bodies.iter().sum::<f64>() / bodies.len() as f64;
    feats.insert("avg_body_ratio".into(), (avg_body * 1000.0).round() / 1000.0);

    // Volume features
    let avg_vol = v.iter().sum::<f64>() / v.len() as f64;
    feats.insert("avg_volume".into(), (avg_vol * 100.0).round() / 100.0);
    let vol_std = v.iter().map(|x| (x - avg_vol).powi(2)).sum::<f64>() / v.len() as f64;
    feats.insert("volume_cv".into(), ((vol_std.sqrt() / avg_vol.max(0.01)) * 1000.0).round() / 1000.0);

    if v.len() > 20 {
        let recent_v: Vec<f64> = v.iter().rev().take(10).cloned().collect();
        let older_v: Vec<f64> = v.iter().rev().skip(10).take(10).cloned().collect();
        let recent_avg = recent_v.iter().sum::<f64>() / recent_v.len() as f64;
        let older_avg = older_v.iter().sum::<f64>() / older_v.len() as f64;
        feats.insert("volume_trend_10v10".into(), ((recent_avg / older_avg.max(0.01)) * 1000.0).round() / 1000.0);
    }

    // Volume-weighted features
    let mut cum_v = 0.0_f64;
    let mut cum_pv = 0.0_f64;
    for i in 0..n { let typical = (h[i] + l[i] + c[i]) / 3.0; cum_pv += typical * v[i]; cum_v += v[i]; }
    let vwap = if cum_v > 0.0 { cum_pv / cum_v } else { c[n - 1] };
    feats.insert("vwap_distance".into(), ((c[n - 1] - vwap) / vwap.max(0.01) * 10000.0).round() / 100.0);

    if n > 20 {
        // Price position in range
        let hi20 = c.iter().rev().take(20).cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo20 = c.iter().rev().take(20).cloned().fold(f64::INFINITY, f64::min);
        let pos = if hi20 > lo20 { (c[n - 1] - lo20) / (hi20 - lo20) * 100.0 } else { 50.0 };
        feats.insert("position_in_20d_range".into(), (pos * 100.0).round() / 100.0);

        // MA distances
        let ma10: f64 = c.iter().rev().take(10).sum::<f64>() / 10.0;
        let ma50: f64 = if n >= 50 { c.iter().rev().take(50).sum::<f64>() / 50.0 } else { c[n - 1] };
        feats.insert("price_vs_ma10".into(), ((c[n - 1] / ma10 - 1.0) * 10000.0).round() / 100.0);
        feats.insert("price_vs_ma50".into(), ((c[n - 1] / ma50 - 1.0) * 10000.0).round() / 100.0);
        feats.insert("ma10_vs_ma50".into(), ((ma10 / ma50 - 1.0) * 10000.0).round() / 100.0);
    }

    // Serial correlation
    if r.len() > 5 {
        let lag1: Vec<f64> = r[..r.len() - 1].to_vec();
        let lag0: Vec<f64> = r[1..].to_vec();
        let m1 = lag1.iter().sum::<f64>() / lag1.len() as f64;
        let m0 = lag0.iter().sum::<f64>() / lag0.len() as f64;
        let num: f64 = lag0.iter().zip(lag1.iter()).map(|(a, b)| (a - m0) * (b - m1)).sum();
        let d0: f64 = lag0.iter().map(|x| (x - m0).powi(2)).sum();
        let d1: f64 = lag1.iter().map(|x| (x - m1).powi(2)).sum();
        let ac = if d0 > 0.0 && d1 > 0.0 { num / (d0 * d1).sqrt() } else { 0.0 };
        feats.insert("autocorrelation_1".into(), (ac * 1000.0).round() / 1000.0);
    }

    // Entropy (Shannon)
    let bins = 20;
    let min_r = r.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_r = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_w = (max_r - min_r).max(0.001) / bins as f64;
    let mut hist = vec![0usize; bins];
    for &val in &r {
        let idx = ((val - min_r) / bin_w).floor() as usize;
        hist[idx.min(bins - 1)] += 1;
    }
    let total = r.len() as f64;
    let entropy: f64 = hist.iter().filter(|&&c| c > 0).map(|&c| {
        let p = c as f64 / total;
        -p * p.log2()
    }).sum();
    feats.insert("entropy".into(), (entropy * 1000.0).round() / 1000.0);
    let max_entropy = (bins as f64).log2();
    let norm_entropy = if max_entropy > 0.0 { entropy / max_entropy } else { 0.5 };
    feats.insert("normalized_entropy".into(), (norm_entropy * 1000.0).round() / 1000.0);

    // Efficiency ratio (net move / total movement)
    let total_move: f64 = r.iter().map(|x| x.abs()).sum();
    let net_move = (c[n - 1] - c[0]) / c[0] * 100.0;
    let er = if total_move > 0.0 { (net_move / total_move).abs() } else { 0.0 };
    feats.insert("efficiency_ratio".into(), (er * 10000.0).round() / 100.0);
    feats.insert("trend_strength".into(), (er * 10000.0).round() / 100.0);

    // Herfindahl index (volume concentration)
    let vol_sum: f64 = v.iter().sum();
    let herf: f64 = if vol_sum > 0.0 {
        v.iter().map(|vi| (vi / vol_sum).powi(2)).sum::<f64>()
    } else { 1.0 / n as f64 };
    feats.insert("volume_concentration".into(), (herf * 10000.0).round() / 100.0);

    // Build top features list
    let mut feat_list: Vec<FeatureImportance> = feats.iter().map(|(k, v)| FeatureImportance {
        name: k.clone(), value: *v,
        category: if k.contains("return") || k.contains("autocorr") { "MOMENTUM".into() }
            else if k.contains("vol") || k.contains("range") || k.contains("body") { "VOLATILITY".into() }
            else if k.contains("volume") || k.contains("concentration") { "VOLUME".into() }
            else if k.contains("entropy") || k.contains("efficiency") || k.contains("trend") { "STRUCTURE".into() }
            else { "PRICE".into() },
    }).collect();
    feat_list.sort_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());

    let fractal = if n > 10 {
        // Simplified fractal dimension via box-counting
        let range = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - c.iter().cloned().fold(f64::INFINITY, f64::min);
        if range > 0.0 {
            let n1 = (range / (range / 4.0)).ln();
            let n2 = (c.len() as f64).ln();
            if n2 > 0.0 { n1 / n2 } else { 1.5 }
        } else { 1.5 }
    } else { 1.5 };

    FeatureResult {
        features: feats,
        top_features: feat_list.into_iter().take(15).collect(),
        entropy: (entropy * 1000.0).round() / 1000.0,
        fractal_dim: (fractal * 1000.0).round() / 1000.0,
        efficiency_ratio: (er * 10000.0).round() / 100.0,
        herfindahl: (herf * 10000.0).round() / 100.0,
    }
}

// ═══════════════════════════════════════════════
// MODULE 4: BAYESIAN PROBABILISTIC MODELS
// ═══════════════════════════════════════════════

#[derive(Deserialize)]
struct BayesianRequest {
    closes: Vec<f64>,
    price: f64,
    targets: Vec<f64>, // price levels to compute probability for
    horizon_hours: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BayesianResult {
    price_distribution: Vec<LevelProbability>,
    expected_return: f64,
    expected_volatility: f64,
    prob_up_1d: f64,
    prob_down_1d: f64,
    prob_up_1w: f64,
    prob_down_1w: f64,
    kalman_filter: KalmanState,
    bayesian_prior: BayesianPrior,
    interpretation: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LevelProbability { level: f64, distance_pct: f64, prob_reach: f64, prob_exceed: f64 }

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct KalmanState { level: f64, slope: f64, slope_smoothed: f64, std_dev: f64, innovation: f64 }

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BayesianPrior { mean_return: f64, std_return: f64, skew: f64, tail_risk: f64, is_normal: bool }

fn run_bayesian(req: &BayesianRequest) -> BayesianResult {
    let n = req.closes.len();
    let price = req.price;
    if n < 20 {
        return BayesianResult {
            price_distribution: vec![], expected_return: 0.0, expected_volatility: 0.0,
            prob_up_1d: 50.0, prob_down_1d: 50.0, prob_up_1w: 50.0, prob_down_1w: 50.0,
            kalman_filter: KalmanState { level: price, slope: 0.0, slope_smoothed: 0.0, std_dev: 0.0, innovation: 0.0 },
            bayesian_prior: BayesianPrior { mean_return: 0.0, std_return: 0.0, skew: 0.0, tail_risk: 0.0, is_normal: true },
            interpretation: "Insufficient data".into(),
        };
    }

    let c = &req.closes;

    // 1. Return statistics
    let r: Vec<f64> = c.windows(2).map(|w| (w[1] - w[0]) / w[0] * 100.0).collect();
    let m = r.len() as f64;
    let mean_r = r.iter().sum::<f64>() / m;
    let var_r = r.iter().map(|x| (x - mean_r).powi(2)).sum::<f64>() / m;
    let std_r = var_r.sqrt().max(0.001);

    // Skew & tail risk
    let skew = r.iter().map(|x| ((x - mean_r) / std_r).powi(3)).sum::<f64>() / m;
    let kurt = r.iter().map(|x| ((x - mean_r) / std_r).powi(4)).sum::<f64>() / m;
    let tail_5pct: Vec<f64> = {
        let mut sorted = r.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (sorted.len() as f64 * 0.05) as usize;
        sorted[..=idx.min(sorted.len() - 1)].to_vec()
    };
    let cvar = if !tail_5pct.is_empty() { tail_5pct.iter().sum::<f64>() / tail_5pct.len() as f64 } else { mean_r };

    let jb = m / 6.0 * (skew.powi(2) + (kurt - 3.0).powi(2) / 4.0);
    let is_normal = jb < 5.99;

    // 2. Kalman filter for dynamic trend
    let mut level = c[0];
    let mut trend = 0.0_f64;
    let mut P_ll = 1.0;
    let mut P_lg = 0.0;
    let mut P_gg = 1.0;
    let var_obs = var_r * price * price / 10000.0;
    let var_proc = var_obs * 0.01;

    let mut innovations: Vec<f64> = Vec::new();
    for i in 1..n {
        // Predict
        let level_pred = level + trend;
        let trend_pred = trend;
        let P_ll_pred = P_ll + 2.0 * P_lg + P_gg + var_proc;
        let P_lg_pred = P_lg + P_gg;
        let P_gg_pred = P_gg + var_proc * 0.1;

        // Update
        let innovation = c[i] - level_pred;
        let kalman_gain_l = P_ll_pred / (P_ll_pred + var_obs);
        let kalman_gain_g = P_lg_pred / (P_ll_pred + var_obs);

        level = level_pred + kalman_gain_l * innovation;
        trend = trend_pred + kalman_gain_g * innovation;
        P_ll = (1.0 - kalman_gain_l) * P_ll_pred;
        P_lg = (1.0 - kalman_gain_l) * P_lg_pred;
        P_gg = P_gg_pred - kalman_gain_g * P_lg_pred;

        innovations.push(innovation);
    }

    let innov_std = if innovations.len() > 1 {
        let mean_inn = innovations.iter().sum::<f64>() / innovations.len() as f64;
        (innovations.iter().map(|x| (x - mean_inn).powi(2)).sum::<f64>() / innovations.len() as f64).sqrt()
    } else { 0.0 };

    // 3. Bayesian probability for each target level
    // P(price reaches X) using first-passage time probability
    let mu_hourly = mean_r / 100.0; // convert to decimal
    let sigma_hourly = std_r / 100.0;
    let annual_factor = 365.0 * 24.0;
    let mu_annual = mu_hourly * annual_factor;
    let sigma_annual = sigma_hourly * annual_factor.sqrt();

    let horizon = req.horizon_hours.unwrap_or(24.0).max(1.0);

    let targets = if req.targets.is_empty() {
        let mut t = Vec::new();
        let step = price * 0.01;
        for i in -10..=10 { t.push(price + i as f64 * step); }
        t
    } else { req.targets.clone() };

    // Monte Carlo simulation for finite-horizon probabilities
    let num_paths = 10000;
    let steps_per_hour = 4; // 15-min steps
    let total_steps = (horizon * steps_per_hour as f64) as usize;

    let mut rng = rand::thread_rng();
    let mut end_prices = Vec::with_capacity(num_paths);
    let mut min_prices = Vec::with_capacity(num_paths);
    let mut max_prices = Vec::with_capacity(num_paths);

    for _ in 0..num_paths {
        let mut p = price;
        let mut p_min = p;
        let mut p_max = p;
        for _ in 0..total_steps {
            let z: f64 = {
                let u1: f64 = rng.gen_range(0.0001_f64..0.9999);
                let u2: f64 = rng.gen_range(0.0001_f64..0.9999);
                (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            let step_ret = mu_hourly / steps_per_hour as f64 + sigma_hourly / (steps_per_hour as f64).sqrt() * z;
            p *= (1.0 + step_ret).max(0.5).min(1.5);
            if p < p_min { p_min = p; }
            if p > p_max { p_max = p; }
        }
        end_prices.push(p);
        min_prices.push(p_min);
        max_prices.push(p_max);
    }

    let levels: Vec<LevelProbability> = targets.iter().map(|&tp| {
        let dist_pct = (tp - price) / price * 100.0;

        // Probability price reaches this level at any point (first-passage)
        let reach_count = if tp <= price {
            min_prices.iter().filter(|&&m| m <= tp).count()
        } else {
            max_prices.iter().filter(|&&m| m >= tp).count()
        };
        let prob_reach = reach_count as f64 / num_paths as f64 * 100.0;

        // Probability price ends below/above this level
        let exceed_count = if tp <= price {
            end_prices.iter().filter(|&&e| e <= tp).count()
        } else {
            end_prices.iter().filter(|&&e| e >= tp).count()
        };
        let prob_exceed = exceed_count as f64 / num_paths as f64 * 100.0;

        LevelProbability {
            level: (tp * 100.0).round() / 100.0,
            distance_pct: (dist_pct * 100.0).round() / 100.0,
            prob_reach: (prob_reach * 100.0).round() / 100.0,
            prob_exceed: (prob_exceed * 100.0).round() / 100.0,
        }
    }).collect();

    // 4. Up/down probabilities for 1 day and 1 week
    let daily_vol = std_r * (24.0_f64).sqrt();
    let weekly_vol = std_r * (168.0_f64).sqrt();
    let daily_drift = mean_r * 24.0;
    let weekly_drift = mean_r * 168.0;

    let prob_up_1d = norm_cdf(daily_drift / daily_vol.max(0.01));
    let prob_down_1d = 1.0 - prob_up_1d;
    let prob_up_1w = norm_cdf(weekly_drift / weekly_vol.max(0.01));
    let prob_down_1w = 1.0 - prob_up_1w;

    let interp = format!(
        "{:.1}% up / {:.1}% down next 24h. Kalman trend: {:.4}/day. Skew: {:.2} ({}). Tail CVaR: {:.2}%. {}.",
        prob_up_1d * 100.0, prob_down_1d * 100.0,
        trend,
        skew,
        if skew.abs() > 0.5 { "asymmetric" } else { "symmetric" },
        cvar,
        if !is_normal { "Non-normal distribution — fat tails likely" } else { "Returns near-normal" }
    );

    BayesianResult {
        price_distribution: levels,
        expected_return: (daily_drift * 100.0).round() / 100.0,
        expected_volatility: (daily_vol * 100.0).round() / 100.0,
        prob_up_1d: (prob_up_1d * 10000.0).round() / 100.0,
        prob_down_1d: (prob_down_1d * 10000.0).round() / 100.0,
        prob_up_1w: (prob_up_1w * 10000.0).round() / 100.0,
        prob_down_1w: (prob_down_1w * 10000.0).round() / 100.0,
        kalman_filter: KalmanState {
            level: (level * 100.0).round() / 100.0,
            slope: (trend * 100000.0).round() / 100000.0,
            slope_smoothed: 0.0,
            std_dev: (innov_std * 100.0).round() / 100.0,
            innovation: *innovations.last().unwrap_or(&0.0),
        },
        bayesian_prior: BayesianPrior {
            mean_return: (mean_r * 10000.0).round() / 100.0,
            std_return: (std_r * 10000.0).round() / 100.0,
            skew: (skew * 1000.0).round() / 1000.0,
            tail_risk: (cvar.abs() * 1000.0).round() / 1000.0,
            is_normal,
        },
        interpretation: interp,
    }
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + (x / 1.41421356237).tanh())
}

// ═══════════════════════════════════════════════
// MODULE 5: OPTION SELLING ANALYSIS
// ═══════════════════════════════════════════════

#[derive(Deserialize)]
struct OptionSellRequest {
    closes: Vec<f64>,
    price: f64,
    strikes: Vec<f64>,       // strike prices to analyze
    days_to_expiry: f64,
    option_type: Option<String>, // "call" or "put" — if None, analyze both
    premium_percent: Option<f64>, // premium as % of strike (for ROC calc)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OptionSellResult {
    strikes: Vec<StrikeAnalysis>,
    max_pain: f64,
    expected_move: f64,
    expected_move_pct: f64,
    best_strike: String,
    iv_estimate: f64,
    realized_vol: f64,
    vol_percentile: f64,
    interpretation: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StrikeAnalysis {
    strike: f64,
    distance_pct: f64,
    prob_itm: f64,
    prob_touch: f64,
    delta_est: f64,
    gamma_est: f64,
    theta_per_day: f64,
    premium_collected: f64,
    max_loss: f64,
    risk_reward: f64,
    roc: f64,
    sigmas_away: f64,
    margin_required: f64,      // collateral needed to hold position
    notional_exposure: f64,    // notional value of 1 contract
    capital_efficiency: f64,   // premium / margin * 100
    roi_if_wins: f64,          // return on margin if trade wins
    roi_if_loses: f64,         // loss as % of margin if trade loses
    days_to_expiry: f64,
    theta_per_margin: f64,     // daily return on margin
    recommendation: String,
}

fn analyze_options(req: &OptionSellRequest) -> OptionSellResult {
    let n = req.closes.len();
    let price = req.price;
    let dte = req.days_to_expiry.max(1.0).min(365.0);
    if n < 20 {
        return OptionSellResult {
            strikes: vec![], max_pain: price, expected_move: 0.0, expected_move_pct: 0.0,
            best_strike: "INSUFFICIENT_DATA".into(), iv_estimate: 0.0, realized_vol: 0.0,
            vol_percentile: 50.0, interpretation: "Insufficient data".into(),
        };
    }

    // Compute return statistics
    let r: Vec<f64> = req.closes.windows(2).map(|w| (w[1] - w[0]) / w[0] * 100.0).collect();
    let m = r.len() as f64;
    let mean_r = r.iter().sum::<f64>() / m;
    let var_r = r.iter().map(|x| (x - mean_r).powi(2)).sum::<f64>() / m;
    let std_r = var_r.sqrt().max(0.001);

    // Realized volatility (annualized)
    let realized_vol = std_r * (365.0_f64).sqrt() / 100.0; // as decimal

    // IV estimate (simplified: recent vol percentile vs historical)
    let vol_percentile = {
        // Compare recent 30d vol vs all-time vol
        let recent = if r.len() > 30 { &r[r.len()-30..] } else { &r };
        let recent_var = recent.iter().map(|x| (x - mean_r).powi(2)).sum::<f64>() / recent.len().max(1) as f64;
        let recent_vol = recent_var.sqrt();
        let all_vol = std_r;
        if all_vol > 0.0 {
            (recent_vol / all_vol).min(2.0) * 50.0
        } else { 50.0 }
    };

    // IV is typically 1.2-1.5x realized vol for BTC
    let iv_estimate = realized_vol * (1.2 + vol_percentile / 250.0);

    // Annualized IV for pricing
    let sigma = iv_estimate; // decimal
    let sigma_daily = sigma / (365.0_f64).sqrt();

    // Monte Carlo simulation for probabilities (10K paths)
    let num_paths = 10000;
    let steps_per_day = 4;
    let total_steps = (dte * steps_per_day as f64) as usize;
    let mu_daily = mean_r / 100.0; // daily drift (decimal)

    let mut rng = rand::thread_rng();
    let mut end_prices = Vec::with_capacity(num_paths);
    let mut min_prices = Vec::with_capacity(num_paths);
    let mut max_prices = Vec::with_capacity(num_paths);

    for _ in 0..num_paths {
        let mut p = price;
        let mut p_min = p;
        let mut p_max = p;
        for _ in 0..total_steps {
            let z: f64 = {
                let u1: f64 = rng.gen_range(0.0001_f64..0.9999);
                let u2: f64 = rng.gen_range(0.0001_f64..0.9999);
                (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            let step_ret = mu_daily / steps_per_day as f64 + sigma_daily / (steps_per_day as f64).sqrt() * z;
            p *= (1.0 + step_ret).max(0.5).min(1.5);
            if p < p_min { p_min = p; }
            if p > p_max { p_max = p; }
        }
        end_prices.push(p);
        min_prices.push(p_min);
        max_prices.push(p_max);
    }

    // Expected move
    let mut end_sorted = end_prices.clone();
    end_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let expected_move = (end_sorted[num_paths / 2] - price).abs();
    let expected_move_pct = expected_move / price * 100.0;

    // Max pain: find price where most options expire worthless
    // Simplified: simulate call/put payoffs at expiry for all strikes
    let strikes = if req.strikes.is_empty() {
        let step = price * 0.05;
        let mut s = Vec::new();
        for i in -10..=10 { s.push(price + i as f64 * step); }
        s
    } else { req.strikes.clone() };

    let mut max_pain = price;
    let mut min_pain = f64::MAX;
    for &sp in &strikes {
        let mut total_pain = 0.0_f64;
        for &ep in &end_prices {
            // Call payoff if above strike
            if ep > sp { total_pain += (ep - sp) * 100.0; } // 1 contract = 0.1 BTC
            // Put payoff if below strike
            if ep < sp { total_pain += (sp - ep) * 100.0; }
        }
        let avg_pain = total_pain / num_paths as f64;
        if avg_pain < min_pain { min_pain = avg_pain; max_pain = sp; }
    }

    // Analyze each strike
    let options_type = req.option_type.as_deref().unwrap_or("both");
    let premium_pct = req.premium_percent.unwrap_or(2.0); // default 2% premium

    let mut strikes_result: Vec<StrikeAnalysis> = strikes.iter().map(|&strike| {
        let dist_pct = (strike - price) / price * 100.0;

        // Probability ITM (expire in the money)
        let itm_count = if strike >= price {
            // Call: ITM if price ends above strike
            end_prices.iter().filter(|&&ep| ep >= strike).count()
        } else {
            // Put: ITM if price ends below strike
            end_prices.iter().filter(|&&ep| ep <= strike).count()
        };
        let prob_itm = itm_count as f64 / num_paths as f64 * 100.0;

        // Probability of touching (for barrier risk)
        let touch_count = if strike >= price {
            max_prices.iter().filter(|&&mp| mp >= strike).count()
        } else {
            min_prices.iter().filter(|&&mp| mp <= strike).count()
        };
        let prob_touch = touch_count as f64 / num_paths as f64 * 100.0;

        // Delta estimate (simplified — normal CDF of distance/sigma)
        let z = dist_pct / 100.0 / (sigma_daily * dte.sqrt());
        let delta = if strike >= price {
            norm_cdf(z) // call delta
        } else {
            -norm_cdf(-z) // put delta
        };

        // Gamma (simplified — peak near ATM)
        let gamma = (1.0 / (sigma * (dte / 365.0).sqrt() * price * 2.0 * std::f64::consts::PI.sqrt())).exp();
        let gamma = gamma.min(0.1);

        // Premium collected (as % of strike)
        let premium = premium_pct / 100.0 * strike;

        // Margin required (realistic exchange margin model)
        // Base margin: 15% of strike, reduced by 0.5% per % OTM, min 8%
        let otm_pct = dist_pct.abs() / 100.0;
        let margin_rate = (0.15 - otm_pct * 0.5).max(0.08).min(0.25);
        let margin = strike * margin_rate;

        // Notional exposure (1 BTC contract)
        let notional = strike;

        // Max loss for naked put/call at expiry (strike reached)
        let max_loss_naked = if dist_pct > 0.0 {
            // Call: max loss if BTC goes to 0 is premium (capped), 
            // but realistic max loss if BTC spikes is unlimited
            // Use 2x margin as worst-case
            margin * 2.0
        } else {
            // Put: max loss = strike - premium (BTC goes to 0)
            (strike - premium).max(margin * 2.0)
        };

        // Theta (daily time decay)
        let theta_per_day = premium / dte;

        // ROC (return on margin)
        let roc = if margin > 0.0 { premium / margin * 100.0 } else { 0.0 };

        // Risk/reward
        let rr = if max_loss_naked > 0.0 { premium / max_loss_naked } else { 0.0 };

        // Capital efficiency
        let cap_eff = if margin > 0.0 { premium / margin * 100.0 } else { 0.0 };

        // ROI scenarios
        let roi_wins = roc; // if wins, keep premium vs margin
        let roi_loses = if margin > 0.0 { -(max_loss_naked / margin * 100.0) } else { 0.0 };

        // Theta per unit margin (daily return efficiency)
        let theta_per_margin = if margin > 0.0 { theta_per_day / margin * 100.0 } else { 0.0 };

        // Sigmas away from current price
        let sigmas = dist_pct.abs() / (sigma_daily * dte.sqrt() * 100.0);

        // Recommendation
        let rec = if prob_itm > 25.0 { "HIGH RISK".into() }
        else if prob_itm > 15.0 { "MODERATE".into() }
        else if prob_itm > 5.0 { "GOOD".into() }
        else { "SAFE".into() };

        StrikeAnalysis {
            strike: (strike * 100.0).round() / 100.0,
            distance_pct: (dist_pct * 100.0).round() / 100.0,
            prob_itm: (prob_itm * 100.0).round() / 100.0,
            prob_touch: (prob_touch * 100.0).round() / 100.0,
            delta_est: (delta * 10000.0).round() / 100.0,
            gamma_est: (gamma * 100000.0).round() / 100000.0,
            theta_per_day: (theta_per_day * 100.0).round() / 100.0,
            premium_collected: (premium * 100.0).round() / 100.0,
            max_loss: (max_loss_naked * 100.0).round() / 100.0,
            risk_reward: (rr * 100.0).round() / 100.0,
            roc: (roc * 100.0).round() / 100.0,
            sigmas_away: (sigmas * 100.0).round() / 100.0,
            margin_required: (margin * 100.0).round() / 100.0,
            notional_exposure: (notional * 100.0).round() / 100.0,
            capital_efficiency: (cap_eff * 100.0).round() / 100.0,
            roi_if_wins: (roi_wins * 100.0).round() / 100.0,
            roi_if_loses: (roi_loses * 100.0).round() / 100.0,
            days_to_expiry: (dte * 100.0).round() / 100.0,
            theta_per_margin: (theta_per_margin * 10000.0).round() / 100.0,
            recommendation: rec,
        }
    }).collect();

    // Sort by distance from price
    strikes_result.sort_by(|a, b| a.distance_pct.abs().partial_cmp(&b.distance_pct.abs()).unwrap());

    // Find best strike (OTM with best risk/reward and low prob ITM)
    let best = strikes_result.iter()
        .filter(|s| s.prob_itm > 1.0 && s.prob_itm < 20.0)
        .max_by(|a, b| a.roc.partial_cmp(&b.roc).unwrap())
        .map(|s| if s.strike >= price { format!("SELL CALL ${:.0}", s.strike) } else { format!("SELL PUT ${:.0}", s.strike) })
        .unwrap_or_else(|| format!("ATM STRADDLE ${:.0}", price));

    let interp = format!(
        "Expected move: {:.1}% (${:.0}). Max pain: ${:.0}. IV: {:.1}% ({}th percentile). \
         Best: {}. Prob ITM for ATM: {:.1}%.",
        expected_move_pct, expected_move, max_pain, iv_estimate * 100.0, vol_percentile as u64,
        best, strikes_result.first().map(|s| s.prob_itm).unwrap_or(0.0)
    );

    OptionSellResult {
        strikes: strikes_result,
        max_pain: (max_pain * 100.0).round() / 100.0,
        expected_move: (expected_move * 100.0).round() / 100.0,
        expected_move_pct: (expected_move_pct * 100.0).round() / 100.0,
        best_strike: best,
        iv_estimate: (iv_estimate * 10000.0).round() / 100.0,
        realized_vol: (realized_vol * 10000.0).round() / 100.0,
        vol_percentile: (vol_percentile * 100.0).round() / 100.0,
        interpretation: interp,
    }
}

// ═══════════════════════════════════════════════
// MASTER HANDLER
// ═══════════════════════════════════════════════

fn handle_action(req: &Request) -> Response {
    let start = Instant::now();

    match req.action.as_str() {
        "orderflow" => {
            let d = &req.data;
            let trades: Vec<TradeTick> = d["trades"].as_array().map(|a| a.iter().map(|v| TradeTick {
                price: v["price"].as_f64().unwrap_or(0.0),
                qty: v["qty"].as_f64().unwrap_or(0.0),
                side: v["side"].as_str().unwrap_or("buy").to_string(),
                time: v["time"].as_i64().unwrap_or(0),
            }).collect()).unwrap_or_default();
            let bids: Vec<[f64; 2]> = d["book_bids"].as_array().map(|a| a.iter().map(|v| {
                [v[0].as_f64().unwrap_or(0.0), v[1].as_f64().unwrap_or(0.0)]
            }).collect()).unwrap_or_default();
            let asks: Vec<[f64; 2]> = d["book_asks"].as_array().map(|a| a.iter().map(|v| {
                [v[0].as_f64().unwrap_or(0.0), v[1].as_f64().unwrap_or(0.0)]
            }).collect()).unwrap_or_default();
            let klines: Vec<Candle> = d["klines"].as_array().map(|a| a.iter().map(|v| Candle {
                o: v["o"].as_f64().unwrap_or(0.0), h: v["h"].as_f64().unwrap_or(0.0),
                l: v["l"].as_f64().unwrap_or(0.0), c: v["c"].as_f64().unwrap_or(0.0),
                v: v["v"].as_f64().unwrap_or(0.0),
            }).collect()).unwrap_or_default();
            let price = d["price"].as_f64().unwrap_or(0.0);

            let ofr = OrderFlowRequest { trades, book_bids: bids, book_asks: asks, klines, price };
            Response {
                orderflow: Some(analyze_orderflow(&ofr)),
                regimes: None, features: None, bayesian: None,
            options: None,
                error: None, timing: Some(start.elapsed().as_secs_f64() * 1000.0),
            }
        }
        "regimes" => {
            let d = &req.data;
            let rr = RegimeRequest {
                closes: d["closes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                highs: d["highs"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                lows: d["lows"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                volumes: d["volumes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                num_regimes: d["num_regimes"].as_u64().map(|v| v as usize),
            };
            Response {
                regimes: Some(detect_regimes(&rr)),
                orderflow: None, features: None, bayesian: None,
            options: None,
                error: None, timing: Some(start.elapsed().as_secs_f64() * 1000.0),
            }
        }
        "features" => {
            let d = &req.data;
            let fr = FeatureRequest {
                closes: d["closes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                highs: d["highs"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                lows: d["lows"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                volumes: d["volumes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
            };
            Response {
                features: Some(engineer_features(&fr)),
                orderflow: None, regimes: None, bayesian: None,
            options: None,
                error: None, timing: Some(start.elapsed().as_secs_f64() * 1000.0),
            }
        }
        "bayesian" => {
            let d = &req.data;
            let targets: Vec<f64> = d["targets"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
            let br = BayesianRequest {
                closes: d["closes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                price: d["price"].as_f64().unwrap_or(0.0),
                targets,
                horizon_hours: d["horizon_hours"].as_f64(),
            };
            Response {
                bayesian: Some(run_bayesian(&br)),
                orderflow: None, regimes: None, features: None,
                options: None,
                error: None, timing: Some(start.elapsed().as_secs_f64() * 1000.0),
            }
        }
        "options" => {
            let d = &req.data;
            let strikes: Vec<f64> = d["strikes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
            let osr = OptionSellRequest {
                closes: d["closes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
                price: d["price"].as_f64().unwrap_or(0.0),
                strikes,
                days_to_expiry: d["days_to_expiry"].as_f64().unwrap_or(7.0),
                option_type: d["option_type"].as_str().map(|s| s.to_string()),
                premium_percent: d["premium_percent"].as_f64(),
            };
            Response {
                options: Some(analyze_options(&osr)),
                orderflow: None, regimes: None, features: None, bayesian: None,
                error: None, timing: Some(start.elapsed().as_secs_f64() * 1000.0),
            }
        }
        "all" => {
            // Run all analyses in sequence
            let d = &req.data;
            let price = d["price"].as_f64().unwrap_or(0.0);

            let closes: Vec<f64> = d["closes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
            let highs: Vec<f64> = d["highs"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
            let lows: Vec<f64> = d["lows"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
            let vols: Vec<f64> = d["volumes"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();

            let regimes = detect_regimes(&RegimeRequest {
                closes: closes.clone(), highs: highs.clone(), lows: lows.clone(), volumes: vols.clone(),
                num_regimes: Some(3),
            });
            let features = engineer_features(&FeatureRequest {
                closes: closes.clone(), highs: highs.clone(), lows: lows.clone(), volumes: vols.clone(),
            });
            let bayesian = run_bayesian(&BayesianRequest {
                closes, price, targets: vec![], horizon_hours: Some(24.0),
            });

            Response {
                regimes: Some(regimes),
                features: Some(features),
                bayesian: Some(bayesian),
                orderflow: None,
                options: None,
                error: None, timing: Some(start.elapsed().as_secs_f64() * 1000.0),
            }
        }
        _ => Response {
            orderflow: None, regimes: None, features: None, bayesian: None,
            options: None,
            error: Some(format!("unknown action: {}", req.action)),
            timing: None,
        },
    }
}

fn main() {
    let mut buffer = String::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();

    let ready = serde_json::json!({"type":"ready","engine":"modern_market_analysis","version":"3.0"});
    let _ = serde_json::to_string(&ready).map(|s| println!("{}", s));
    let _ = io::stdout().flush();

    loop {
        buffer.clear();
        match handle.read_line(&mut buffer) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = buffer.trim();
                if trimmed.is_empty() || trimmed == "exit" { break; }
                let req: Result<Request, _> = serde_json::from_str(trimmed);
                let resp = match req {
                    Ok(r) => handle_action(&r),
                    Err(e) => Response {
                        orderflow: None, regimes: None, features: None, bayesian: None,
            options: None,
                        error: Some(format!("parse error: {}", e)), timing: None,
                    },
                };
                let _ = serde_json::to_string(&resp).map(|s| println!("{}", s));
                let _ = io::stdout().flush();
            }
            Err(_) => break,
        }
    }
}
