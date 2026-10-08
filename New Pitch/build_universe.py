"""
Taiwan GIV data build: shares outstanding and index weights.

TAIEX is a full market-cap weighted index (發行量加權), so a stock's index weight
is (shares issued x price) / (sum over constituents) -- total issued shares, not
free float. TWSE's company-profile open data carries shares issued directly, for
every listed company, in a single free call.

Caveat worth carrying into the analysis: this is a *current* snapshot. Taiwanese
companies issue stock dividends heavily, so shares issued drifts over the sample
and weights computed from today's share count are only approximate for 2020.
build_shares_history.py backfills the weekly series; this file is the fast path
that lets the first-pass Herfindahl and weights be computed immediately.

Writes data/universe.csv.
"""

import os

import pandas as pd
import requests

from twse_client import HEADERS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT_CSV = os.path.join(DATA, "universe.csv")

PROFILE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"

UNIVERSE_N = 300     # granular units i for the GIV


def fetch_profiles():
    r = requests.get(PROFILE_URL, headers=HEADERS, timeout=90)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df.rename(columns={
        "公司代號": "stock_id",
        "公司簡稱": "stock_name",
        "產業別": "industry",
        "已發行普通股數或TDR原股發行股數": "shares_issued",
        "實收資本額": "paid_in_capital",
        "普通股每股面額": "par_value",
        "上市日期": "listing_date",
    })
    keep = ["stock_id", "stock_name", "industry", "shares_issued",
            "paid_in_capital", "par_value", "listing_date"]
    return df[[c for c in keep if c in df.columns]]


SNAPSHOT_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"


def latest_prices():
    """Close for every listed stock, from TWSE's live snapshot.

    Uses the exchange snapshot rather than the CSV being written by
    collect_twse.py, so the universe can be built at any point during the pull
    and always reflects current prices against current share counts.
    """
    r = requests.get(SNAPSHOT_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    snap = pd.DataFrame(r.json()).rename(columns={"Code": "stock_id"})
    snap["close"] = pd.to_numeric(snap.ClosingPrice.str.replace(",", ""), errors="coerce")
    asof = snap.Date.iloc[0] if "Date" in snap.columns else "latest"
    return snap[["stock_id", "close"]].dropna(), asof


def main():
    prof = fetch_profiles()
    prof["shares_issued"] = pd.to_numeric(prof.shares_issued, errors="coerce")

    # Fall back to paid-in capital / par value where shares issued is blank.
    cap = pd.to_numeric(prof.get("paid_in_capital"), errors="coerce")
    par = pd.to_numeric(prof.get("par_value"), errors="coerce").replace(0, pd.NA)
    prof["shares_issued"] = prof.shares_issued.fillna(cap / par)
    prof = prof.dropna(subset=["shares_issued"])
    prof = prof[prof.shares_issued > 0]

    px, asof = latest_prices()
    uni = prof.merge(px, on="stock_id", how="inner")
    uni["market_cap"] = uni.close * uni.shares_issued
    uni = uni.nlargest(UNIVERSE_N, "market_cap").reset_index(drop=True)
    uni["weight"] = uni.market_cap / uni.market_cap.sum()
    uni["asof"] = asof

    os.makedirs(DATA, exist_ok=True)
    uni.to_csv(OUT_CSV, index=False)

    herf = (uni.weight ** 2).sum()
    print(f"wrote {OUT_CSV}: {len(uni)} stocks, prices as of {asof}")
    print(f"  total cap  TWD {uni.market_cap.sum()/1e12:,.1f} tn")
    print("  top 10 weights:")
    for r in uni.head(10).itertuples():
        print(f"    {r.stock_id} {r.stock_name:12s} {r.weight:7.2%}")
    print(f"\n  Herfindahl H = {herf:.4f}  ->  1/H = {1/herf:.2f} effective names")
    print("  (GIV power needs a high H; compare with 1/H = 2.07 for the 5-sector file)")


if __name__ == "__main__":
    main()
