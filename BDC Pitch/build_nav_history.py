"""Assemble a point-in-time quarterly NAV history for listed BDCs, 2012 onward.

For each BDC and fiscal quarter-end:
  value  : XBRL NAV per share (bdc_nav.csv) where tagged in the original filing; otherwise the 10-Q balance sheet (current-period column);
           for fiscal year-ends (10-K quarters) the prior-period column of the next 10-Q, because the 10-K balance
           sheet sits too deep in the document to read reliably.
  public : the earliest moment the NAV was public: the first earnings release (8-K Item 2.02) quoting a NAV within
           100 days after the quarter-end, else the 10-Q / 10-K acceptance time.
All NAVs are put on Yahoo's split-adjusted basis (data/splits.csv), so price / NAV is consistent across reverse splits.

Output: data/nav_history.csv  ticker, period, nav_ps (split-adjusted), nav_raw, source, public (US Eastern timestamp),
        public_source, release_nav (NAV quoted in that release, raw), release_adjusted, clean (False near reverse splits)
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


def eastern(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)


def main():
    fil = pd.read_csv(DATA / "filing_nav.csv", parse_dates=["period"])
    fil["accepted"] = eastern(fil.accepted)
    fil["c0"] = pd.to_numeric(fil.cands.str.split(";").str[0], errors="coerce")
    fil["c1"] = pd.to_numeric(fil.cands.str.split(";").str[1], errors="coerce")
    rows = []
    for t, g in fil.sort_values("period").groupby("ticker"):
        q = g[g.form == "10-Q"]
        for _, r in q.iterrows():
            rows.append({"ticker": t, "period": r.period, "nav_raw": r.c0, "source": "10-Q", "filing_public": r.accepted})
        for _, k in g[g.form == "10-K"].iterrows():
            nxt = q[q.period > k.period].head(1)            # first 10-Q after the fiscal year-end: its prior-period column
            val = nxt.c1.iloc[0] if len(nxt) and (nxt.period.iloc[0] - k.period).days < 120 else np.nan
            rows.append({"ticker": t, "period": k.period, "nav_raw": val, "source": "next 10-Q prior column", "filing_public": k.accepted})
    H = pd.DataFrame(rows)
    # XBRL values (2022+) take precedence where present
    x = pd.read_csv(DATA / "bdc_nav.csv", parse_dates=["period_end", "filed"]).dropna(subset=["nav_ps"])
    # only original filings: values first tagged years later come from post-split comparatives (restated per share)
    x = x[(x.filed - x.period_end).dt.days <= 120]
    x = x.rename(columns={"period_end": "period", "nav_ps": "nav_xbrl"})[["ticker", "period", "nav_xbrl"]]
    H = H.merge(x, on=["ticker", "period"], how="left")
    H["source"] = np.where(H.nav_xbrl.notna(), "XBRL", H.source)
    H["nav_raw"] = H.nav_xbrl.fillna(H.nav_raw)
    # public timestamp: first earnings release with a NAV within 100 days after period-end
    pr = pd.read_csv(DATA / "press_nav.csv").dropna(subset=["nav_ps"])
    pr["accepted"] = eastern(pr.accepted)
    pub = []
    for _, r in H.iterrows():
        c = pr[(pr.ticker == r.ticker) & (pr.accepted > r.period) & (pr.accepted <= r.period + pd.Timedelta(days=100))].sort_values("accepted")
        c = c[~c.estimate.astype(str).eq("True")]       # preliminary estimates are not the reported NAV
        if len(c) and c.accepted.iloc[0] <= r.filing_public:
            pub.append((c.accepted.iloc[0], "earnings release", c.nav_ps.iloc[0], bool(c.adjusted.iloc[0])))
        else:
            pub.append((r.filing_public, "10-Q/10-K", np.nan, False))
    H[["public", "public_source", "release_nav", "release_adjusted"]] = pd.DataFrame(pub, index=H.index)
    # split adjustment: divide by the product of split ratios that occurred after the NAV was public
    sp = pd.read_csv(DATA / "splits.csv", parse_dates=["date"])
    def factor(t, when):
        s = sp[(sp.ticker == t) & (sp.date > when)]
        return float(np.prod(s.ratio)) if len(s) else 1.0
    H["nav_ps"] = [n / factor(t, p) for n, t, p in zip(H.nav_raw, H.ticker, H.public)]
    H = H.dropna(subset=["nav_ps"]).sort_values(["ticker", "period"]).drop_duplicates(["ticker", "period"])
    # Filings near a reverse split mix pre- and post-split per-share bases (and Yahoo's split dates are approximate),
    # so NAVs within 270 days of a split are flagged and excluded downstream. GECC before 2018 is a merger-era mix.
    big = sp[(sp.ratio < 0.9) | (sp.ratio > 1.1)]
    near = [((big.ticker == t) & ((big.date - p).abs().dt.days <= 270)).any() for t, p in zip(H.ticker, H.period)]
    H["clean"] = ~np.array(near) & ~((H.ticker == "GECC") & (H.period < "2018-01-01"))
    H.drop(columns=["nav_xbrl"]).to_csv(DATA / "nav_history.csv", index=False)
    print(H.groupby(H.period.dt.year).agg(n=("ticker", "size"), bdcs=("ticker", "nunique")).to_string())
    print(H.source.value_counts().to_dict(), H.public_source.value_counts().to_dict())


if __name__ == "__main__":
    main()
