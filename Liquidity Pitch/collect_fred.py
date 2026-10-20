"""Fed balance-sheet liquidity series from FRED (weekly H.4.1, Wednesday levels, published Thursday 16:30 ET).

WALCL = Fed total assets ($m), WTREGEN = Treasury General Account ($bn), RRPONTSYD = overnight reverse repo ($bn, daily).
Output: data/fred.csv  date, WALCL, WTREGEN, RRPONTSYD
"""

import io
from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"

if __name__ == "__main__":
    out = []
    for s in ["WALCL", "WTREGEN", "RRPONTSYD"]:
        d = pd.read_csv(io.StringIO(requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", timeout=60).text))
        d.columns = ["date", s]; d["date"] = pd.to_datetime(d.date); d[s] = pd.to_numeric(d[s], errors="coerce")
        out.append(d.set_index("date"))
    pd.concat(out, axis=1).to_csv(DATA / "fred.csv")
