// ──────────────────────────────────────────────────
//  FINNHUB.MJS — News & Social Sentiment
//  Free tier: 60 calls/min, no credit card needed
//  Get free API key at: https://finnhub.io/register
// ──────────────────────────────────────────────────

const FH = 'https://finnhub.io/api/v1';

// Set your key in .env: FINNHUB_KEY=your_key
function getKey() {
  try {
    const fs = require('fs');
    const env = fs.existsSync('.env') ? fs.readFileSync('.env', 'utf8') : '';
    const match = env.match(/FINNHUB_KEY=(.+)/);
    return match ? match[1].trim() : '';
  } catch (_) { return ''; }
}
const API_KEY = getKey() || process.env.FINNHUB_KEY || 'demo';

// ─── Crypto News ────────────────────────────────
export async function fetchCryptoNews() {
  try {
    const res = await fetch(`${FH}/news?category=crypto&token=${API_KEY}`);
    const data = await res.json();
    if (!Array.isArray(data)) return null;

    // Analyze sentiment from headlines
    const headlines = data.slice(0, 20);
    const sentiment = analyzeHeadlines(headlines);

    return {
      count: headlines.length,
      sentiment,
      top3: headlines.slice(0, 3).map(n => ({
        headline: n.headline,
        source: n.source,
        url: n.url,
      })),
      score: sentiment.bullish - sentiment.bearish,
    };
  } catch (_) {
    return null;
  }
}

function analyzeHeadlines(news) {
  const bullish = ['surge', 'rally', 'bull', 'breakout', 'gain', 'rise', 'up', 'high', 'record', 'green',
    'accumulat', 'buy', 'long', 'bullish', 'soar', 'jump', 'moon', 'pump', 'recovery', 'bounce'];
  const bearish = ['crash', 'drop', 'fall', 'down', 'low', 'red', 'bear', 'bearish', 'sell', 'short',
    'decline', 'dump', 'panic', 'fear', 'correction', 'capitulat', 'liquidat', 'risk', 'warn', 'crisis'];

  let bull = 0, bear = 0;
  for (const n of news) {
    const text = (n.headline + ' ' + n.summary).toLowerCase();
    for (const w of bullish) if (text.includes(w)) bull++;
    for (const w of bearish) if (text.includes(w)) bear++;
  }

  const total = bull + bear || 1;
  return {
    bullish: round(bull / total * 100, 1),
    bearish: round(bear / total * 100, 1),
    neutral: round(100 - (bull + bear) / total * 100, 1),
    headline: total > 5 ? `${bull > bear ? 'Bullish' : 'Bearish'} headlines (${bull}B/${bear}S)` : 'Neutral news',
  };
}

// ─── Crypto Social Sentiment ────────────────────
export async function fetchSocialSentiment(symbol = 'BTC') {
  try {
    // Reddit + Twitter sentiment from Finnhub
    const reddit = await fetch(`${FH}/stock/social-sentiment?symbol=BTC&token=${API_KEY}`);
    const rd = await reddit.json();

    if (rd.error) return null;

    return {
      reddit: {
        mention: rd.reddit?.mention || 0,
        positiveMention: rd.reddit?.positiveMention || 0,
        negativeMention: rd.reddit?.negativeMention || 0,
        score: rd.reddit ? round((rd.reddit.positiveMention - rd.reddit.negativeMention) / Math.max(rd.reddit.mention, 1) * 100, 1) : 0,
      },
      twitter: {
        mention: rd.twitter?.mention || 0,
        positiveMention: rd.twitter?.positiveMention || 0,
        negativeMention: rd.twitter?.negativeMention || 0,
        score: rd.twitter ? round((rd.twitter.positiveMention - rd.twitter.negativeMention) / Math.max(rd.twitter.mention, 1) * 100, 1) : 0,
      },
      compositeScore: rd.reddit || rd.twitter
        ? round((((rd.reddit?.positiveMention || 0) - (rd.reddit?.negativeMention || 0)) +
                 ((rd.twitter?.positiveMention || 0) - (rd.twitter?.negativeMention || 0))) /
                 Math.max((rd.reddit?.mention || 0) + (rd.twitter?.mention || 0), 1) * 100, 1)
        : 0,
    };
  } catch (_) {
    return null;
  }
}

// ─── Fetch all sentiment ────────────────────────
export async function fetchAllSentiment() {
  const [news, social] = await Promise.allSettled([
    fetchCryptoNews(),
    fetchSocialSentiment('BTC'),
  ]);

  const newsData = news.status === 'fulfilled' ? news.value : null;
  const socialData = social.status === 'fulfilled' ? social.value : null;

  // Composite sentiment score (-100 to +100)
  let score = 0, count = 0;
  if (newsData?.score !== undefined) { score += newsData.score * 2; count++; }
  if (socialData?.compositeScore !== undefined) { score += socialData.compositeScore; count++; }

  return {
    news: newsData,
    social: socialData,
    compositeScore: count > 0 ? round(score / count, 1) : 0,
    signal: getSentimentSignal(count > 0 ? round(score / count, 1) : 0),
  };
}

function getSentimentSignal(score) {
  if (score > 30) return 'BULLISH — positive news/social sentiment';
  if (score > 10) return 'MILDLY BULLISH';
  if (score > -10) return 'NEUTRAL';
  if (score > -30) return 'MILDLY BEARISH';
  return 'BEARISH — negative news/social sentiment';
}

// ─── Format for alerts ──────────────────────────
export function formatSentimentAlert(data) {
  if (!data) return '';
  let msg = '';
  if (data.news?.sentiment?.headline) msg += `News: ${data.news.sentiment.headline} `;
  if (data.social?.reddit?.score !== undefined) msg += `Reddit:${data.social.reddit.score > 0 ? '+' : ''}${data.social.reddit.score} `;
  msg += `| Score:${data.compositeScore > 0 ? '+' : ''}${data.compositeScore}`;
  return msg;
}

function round(n, d = 2) { return Math.round(n * 10 ** d) / 10 ** d; }

// CLI test
const isMain = process.argv[1]?.includes('finnhub.mjs');
if (isMain) {
  const result = await fetchAllSentiment();
  console.log(JSON.stringify(result, null, 2));
  console.log(formatSentimentAlert(result));
}
