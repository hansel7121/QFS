"""Daily prices for stocks that have been among the 50 largest in the S&P 500 at some point since 2006, plus SPY.

COR1M / COR3M are built from the top 50 S&P 500 names, so realized correlation needs the same kind of basket.
Outputs:
  data/stocks.csv  date, ticker, close (split-adjusted), adj_close (dividend-adjusted)
  data/shares.csv  ticker, shares (current shares outstanding, used to approximate market cap = close x shares)
Names that were delisted or acquired (e.g. Celgene, Monsanto, Time Warner) are not on Yahoo; see the EDA caveats.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
TICKERS = sorted(set("""
AAPL MSFT AMZN GOOGL META NVDA TSLA BRK-B JPM JNJ XOM V PG UNH HD MA CVX PFE KO PEP MRK ABBV BAC WMT DIS CSCO INTC
VZ T ORCL CMCSA ADBE CRM NFLX NKE MCD COST TMO ABT AVGO LLY TXN QCOM DHR WFC C GS MS AMGN GILD BMY MDT UNP UPS HON
IBM MMM BA CAT GE SBUX LOW NEE LIN PM MO AMD ACN AMAT INTU NOW ISRG BKNG SPGI BLK AXP RTX LMT DE ELV CVS SCHW PYPL
ADP MDLZ CL TGT USB COP SLB OXY EOG PLTR UBER ANET MU LRCX KLAC PANW HPQ F GM FCX EMR DUK SO D EXC MET AIG COF
BK PNC CB MMC ADI SYK BSX VRTX REGN CI HUM KHC KMB GD NOC FDX TJX MCK CAH APD DD DOW TRV ALL WBA APH ETN TT
""".split()))


def main():
    raw = yf.download(TICKERS + ["SPY"], start="2005-06-01", auto_adjust=False, progress=False, threads=True)
    df = pd.concat([raw["Close"].stack().rename("close"), raw["Adj Close"].stack().rename("adj_close")], axis=1)
    df = df.reset_index()
    df.columns = ["date", "ticker", "close", "adj_close"]
    df.dropna(subset=["close"]).to_csv(DATA / "stocks.csv", index=False)
    print("prices:", df.ticker.nunique(), "tickers,", df.date.min().date(), "..", df.date.max().date())
    sh = []
    for t in TICKERS:
        try:
            sh.append({"ticker": t, "shares": yf.Ticker(t).fast_info["shares"]})
        except Exception as e:  # missing share count: name is dropped from the market-cap ranking
            print("no shares:", t, e)
    pd.DataFrame(sh).to_csv(DATA / "shares.csv", index=False)
    print("shares:", len(sh))


if __name__ == "__main__":
    main()
