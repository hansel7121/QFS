"""WTI and Henry Hub futures, contracts 1-4, daily settlements from the EIA (free .xls history, ends 2024-04).

Output: data/RCLC{1..4}.xls, data/RNGC{1..4}.xls  (read with pandas + xlrd, sheet 1, skiprows=2)
"""

from pathlib import Path
import requests

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

if __name__ == "__main__":
    for k in range(1, 5):
        for code, area in [("RCLC", "pet"), ("RNGC", "ng")]:
            r = requests.get(f"https://www.eia.gov/dnav/{area}/hist_xls/{code}{k}d.xls", headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            (DATA / f"{code}{k}.xls").write_bytes(r.content)
            print(code, k, r.status_code, len(r.content))
