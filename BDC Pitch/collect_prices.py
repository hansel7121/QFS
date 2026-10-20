"""Daily prices from Yahoo Finance for listed BDCs and market proxies.

Outputs data/prices.csv: date, ticker, open, close (raw, for price/NAV), adj_close (dividend-adjusted, for total return), volume.
Proxies: BKLN / SRLN (public leveraged loans), HYG (high yield), BIZD (BDC ETF), SPY, ^IRX (3m T-bill yield).
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
LISTED = ("ARCC OBDC FSK BXSL GBDC PSEC MAIN HTGC OCSL TSLX GSBD NMFC BBDC CGBD SLRC CSWC TCPC PFLT PNNT MFIC "
          "BCSF TRIN FDUS GAIN GLAD CCAP KBDC MSDL NCDL PSBD OFS SAR WHF HRZN TPVG RWAY GECC ICMB OXSQ SCM CION "
          "BCIC").split()
PROXIES = ["BKLN", "SRLN", "HYG", "BIZD", "SPY", "^IRX"]


def main():
    tickers = LISTED + PROXIES
    raw = yf.download(tickers, start="2012-01-01", auto_adjust=False, progress=False, group_by="column", threads=True)
    df = pd.concat([raw[k].stack().rename(v) for k, v in
                    [("Open", "open"), ("Close", "close"), ("Adj Close", "adj_close"), ("Volume", "volume")]], axis=1).reset_index()
    df.columns = ["date", "ticker", "open", "close", "adj_close", "volume"]
    df = df.dropna(subset=["close"]).sort_values(["ticker", "date"])
    df.to_csv(DATA / "prices.csv", index=False)
    cov = df.groupby("ticker").date.agg(["min", "max", "count"])
    print(cov.to_string())
    missing = sorted(set(tickers) - set(df.ticker))
    print("missing:", missing)


if __name__ == "__main__":
    main()
