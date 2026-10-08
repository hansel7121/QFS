# Delta Hedging TW — warrant issuer hedging flow as a tradable signal

**Status:** parked, ready to resume. Data sources verified, collector written and smoke-tested, full pull not yet run.
**Parked on:** 2026-09-23

---

## The idea in one paragraph

Taiwanese warrant issuers are required to run delta-neutral books. When a new **call** warrant is listed, the issuer must **buy the underlying stock** to hedge. When an in-the-money warrant expires, it is cash-settled and the issuer must **sell** the hedge. Both flows are mechanical, sizeable, and — critically — their timing is **known in advance**, because TWSE announces new warrant listings before trading starts and expiry dates are fixed at issuance. That is a predictable, non-informational flow hitting a market where we have already measured that flows move prices.

## Why this one is worth doing

Two earlier Taiwan ideas died for reasons this one structurally avoids:

| Previous attempt | Why it failed | Why this avoids it |
|---|---|---|
| Granular IV / demand multiplier | Effect was **contemporaneous**. Lagged signal gave t=0.26. Nothing to trade. | Listing dates are known **before** the flow arrives |
| Institutional flow → next-day | Real signal (+38.5bp, t=14.1) but **killed by Taiwan's 30bp transaction tax** | Same tax problem — but see the futures escape hatch below |

The GIV study did establish something useful that carries over: **Taiwanese equity demand is inelastic.** 1% of market cap flowing in moves the market ~18% (bootstrap CI 12.1–23.1%), versus ~5% for the US. So mechanical flow in this market has unusually large price impact. That is the economic case for why a hedging-flow signal should be detectable here rather than in the US.

## The academic anchor

**"The impact of derivatives hedging on the stock market: Evidence from Taiwan's covered warrants market"**, *Journal of Banking & Finance* (2014).
<https://www.sciencedirect.com/science/article/abs/pii/S0378426614000417>

Key findings:
- Call warrant introduction produces a **significantly positive price effect** on the underlying
- Cumulative abnormal return averages **+0.904%** over the 20 days around the warrant announcement (≈10.85% annualised)
- Significantly positive abnormal trading volume **before** announcement dates
- For calls expiring **in the money**, negative price impact at expiry — warrants are cash-settled in TWSE, so issuers must liquidate the hedge portfolio, creating selling pressure

Supporting: <http://www.fin.ntu.edu.tw/~conference2002/proceding/12-2.pdf> (warrant issuer trading behaviour), and the Hong Kong analogue <https://www.sciencedirect.com/science/article/abs/pii/S0378426600001382>.

---

## Data — all free, all verified working

### Endpoints confirmed live on 2026-09-23

| What | Endpoint | Notes |
|---|---|---|
| Daily warrant quotes — **calls** | `https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date=YYYYMMDD&type=0999&response=json` | ~27,500 rows/day, full history back to 2020 |
| Daily warrant quotes — **puts** | same URL, `type=0999P` | ~4,200 rows/day |
| Warrant **master** | `https://openapi.twse.com.tw/v1/opendata/t187ap37_L` | 40,951 live warrants |
| Underlying name → stock id | `https://openapi.twse.com.tw/v1/opendata/t187ap03_L` | company short names |

Warrant master fields: `權證代號`, `權證簡稱`, `權證類型` (認購/認售), `標的證券/指數`, `履約開始日`, `最後交易日`, `履約截止日`, **`發行單位數量(仟單位)`** (issue size — median 5,000k units), `最新履約價格`.

### The survivorship problem and its fix

The master file only holds warrants **still alive**, so `履約開始日` reaches back only to 2024-08. Using it alone for a 2020–2026 study would be survivorship-biased — every warrant that listed and expired inside the sample would be missing.

**Fix:** a warrant's *first appearance* in the daily quote file **is** its listing date. Walk days forward holding a set of codes already seen and emit only new ones. This is survivorship-free and collapses ~50 million rows into ~250k.

Measured on a 3-day smoke test:

```
20240102   31,693 warrants   <- BURN-IN: standing universe, NOT new listings
20240103      401 new
20240104      312 new
```

**~350 new listings per day** across ~450 underlying stocks. Over 1,600 trading days that is on the order of half a million events. Power will not be the constraint.

### Underlying recovery

Warrant short names are prefixed with the underlying's short name:
`台積電凱基3B購01` → `台積電` → `2330`

Prefix map is learned from the master file (which states the underlying explicitly), sorted longest-name-first so a short name cannot shadow a longer one. **Maps 77% of warrants** to a listed common stock; the remainder are index and ETF warrants (臺股指數, 元大台灣50, leveraged ETFs) which should be excluded anyway.

Top underlyings by live warrant count: 台積電 1,372 · 鴻海 677 · 奇鋐 633 · 緯穎 614 · 南亞科 609.

### Data already on disk

In `New Pitch/data/`, from the Taiwan GIV work:

| File | Contents |
|---|---|
| `inst_flows.csv` | 1,551,483 rows. Per-stock daily institutional flows, 2020-03-02 → 2026-09-23. **Includes the `dealer_hedge_net` column — this is the issuers' hedging flow directly** |
| `prices.csv` | 1,596,068 rows. Per-stock OHLC, volume, trade value, same period |
| `holdings.csv` | Daily shares issued + foreign holdings per stock (MI_QFIIS). **Pull was at 1,172/1,601 when parked — resume with `python3 collect_holdings.py`** |
| `universe.csv` | Top 300 by market cap with index weights |

**This is the unusual advantage.** `dealer_hedge_net` (自營商避險 in TWSE's T86 report) is the issuers' hedging flow, per stock, per day, already collected. Almost nobody looks at this column. It lets you verify the *mechanism*, not just the correlation.

---

## Code written

In `New Pitch/`:

| File | Purpose |
|---|---|
| `collect_warrants.py` | The warrant listing collector. Written, smoke-tested, **not yet run in full** |
| `twse_client.py` | Throttled TWSE client with backoff. Reused |
| `panel.py` | Panel construction, returns, weights |
| `collect_twse.py` | The T86 + MI_INDEX pull (complete) |
| `collect_holdings.py` | MI_QFIIS pull (incomplete, see above) |

`collect_warrants.py` outputs `data/warrant_listings.csv` with `first_date, warrant_id, warrant_name, kind, volume, underlying_name, stock_id, is_burnin`, checkpointed per day so it survives restarts.

**To run:** `python3 collect_warrants.py --interval 2.0` — roughly 1.8 hours.

⚠️ **Do not run it concurrently with `collect_holdings.py`.** Two processes at a 2s interval means TWSE sees one request per second, which risks throttling or an IP block. Finish holdings first.

---

## Test plan

### Stage 1 — does issuance actually cause hedge buying?
Regress `dealer_hedge_net` for stock *i* on day *t* against warrant issuance in stock *i* around *t*. If issuers hedge as regulation requires, this must be strongly positive.

**This is the cheapest possible falsification — one afternoon.** If it fails, the premise is wrong and nothing else matters. Do this first.

### Stage 2 — does hedge flow move the price?
Already partly measured. Decile sort on dealer-hedge flow → next-day return:

```
dealer hedge   +8.2 bp/day   t = +3.09
```

Positive and significant, but **the weakest of the four flow categories** (investment trust +38.5bp t=14.1, all institutions +30.8bp t=12.3, foreign +25.6bp t=10.4). Be honest about this upfront — it is the main reason the pitch might not clear costs.

### Stage 3 — is it tradable?
1. **Check the lag structure before building anything.** Ten lines. This single test is what would have saved the two previous Taiwan attempts.
2. **Then the cost test.** Taiwan's securities transaction tax is 30bp on sale, and it kills most Taiwanese equity strategies. Nobody models it. Model it.

### The expiry leg — possibly the better half
Chan et al. find *negative* pressure when in-the-money cash-settled warrants expire and issuers dump the hedge. Expiry dates are in the master file and known **months** in advance. That is a short signal on a pre-announced calendar, likely less crowded than the issuance leg. Worth testing in parallel, not as an afterthought.

---

## The cost problem, and the escape hatch

```
securities transaction tax   30 bp   (on SALE, unavoidable in cash equities)
commission                    5 bp   (institutional)
spread                       10 bp
                           -------
round trip                   45 bp
```

Any daily-rebalanced Taiwanese cash-equity strategy needs >45bp/day gross to survive. The institutional-flow signal, at 38.5bp gross, does **not** clear it — net −6.5bp/day.

**Escape hatches, in order of promise:**
1. **TAIFEX single-stock futures.** Transaction tax ~0.002% ≈ **0.2bp** versus 30bp on cash. ~250 underlyings listed. *Liquidity has not been verified — this is the first thing to check, and it determines whether the whole pitch is investable.*
2. **Day-trade tax is halved to 15bp** (since 2017, repeatedly extended).
3. Longer holding periods — **but only if the signal persists**, which for institutional flow it does not (entire edge on t+1, nothing after). Measure decay directly; never amortise cost over an assumed holding period.

---

## Lessons carried from the two failed attempts

1. **Check that the signal precedes the return before building anything.** The GIV multiplier was real, well-identified, robust to every test — and completely contemporaneous, so untradable. One lag regression at the start would have revealed that in minutes.
2. **A result that beats the literature by a wide margin is a bug until proven otherwise.** An implied multiplier ~6× the most extreme published estimate turned out to be reverse causality. Separately, a units error (log return read as a percentage, off by 100×) briefly inverted the entire conclusion. Sanity-check magnitudes against a published benchmark.
3. **Model Taiwan's transaction tax from the start.** It is 30bp, it applies to every sale, and it is the single most common reason Taiwanese equity backtests do not survive contact with reality.
4. **Cross-sectional demeaning only removes a term that is genuinely common across units.** The stock-level GIV failed because the price term was stock-specific and did not demean away.

---

## Next actions when resuming

1. Finish `collect_holdings.py` (at 1,172/1,601)
2. Run `collect_warrants.py` (~1.8h, only after holdings completes)
3. **Stage 1 test** — issuance → `dealer_hedge_net`. Cheapest falsification
4. Check TAIFEX single-stock-futures liquidity. Determines whether any of this is investable
5. Only then build the event study, and test issuance and expiry legs separately
