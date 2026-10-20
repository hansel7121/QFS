"""Every Treasury note and bond auction since 1990, from the TreasuryDirect auction API.

Outputs:
  data/auctions.csv  auction_date, announce_date, issue_date, type, term (original term, e.g. 10-Year), reopening,
                     tips, frn, offering (USD), high_yield, bid_to_cover, dealer_share, indirect_share, direct_share,
                     tail_bp (high yield - when-issued yield is not public; avg/median yield gap is kept instead)
"""

from pathlib import Path
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
URL = "https://www.treasurydirect.gov/TA_WS/securities/search"
UA = {"User-Agent": "Mozilla/5.0"}


def main():
    rows = []
    for typ in ("Note", "Bond"):
        for y0 in range(1990, 2027, 3):
            r = requests.get(URL, params={"format": "json", "type": typ, "startDate": f"{y0}-01-01",
                                          "endDate": f"{y0 + 2}-12-31"}, headers=UA, timeout=60).json()
            rows += r
            print(typ, y0, len(r), flush=True)
    d = pd.DataFrame(rows).drop_duplicates(["cusip", "auctionDate"])
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    acc = num("totalAccepted").where(num("totalAccepted") > 0, num("offeringAmount"))
    out = pd.DataFrame({
        "auction_date": pd.to_datetime(d.auctionDate).dt.normalize(),
        "announce_date": pd.to_datetime(d.announcementDate, errors="coerce").dt.normalize(),
        "issue_date": pd.to_datetime(d.issueDate, errors="coerce").dt.normalize(),
        "type": d.type, "term": d.originalSecurityTerm.where(d.originalSecurityTerm.ne(""), d.securityTerm),
        "reopening": d.reopening.eq("Yes"), "tips": d.tips.eq("Yes"), "frn": d.floatingRate.eq("Yes"),
        "offering": num("offeringAmount"), "high_yield": num("highYield"), "median_yield": num("averageMedianYield"),
        "bid_to_cover": num("bidToCoverRatio"),
        "dealer_share": num("primaryDealerAccepted") / acc, "indirect_share": num("indirectBidderAccepted") / acc,
        "direct_share": num("directBidderAccepted") / acc,
    }).sort_values("auction_date")
    out.to_csv(DATA / "auctions.csv", index=False)
    nom = out[~out.tips & ~out.frn]
    print(f"{len(out)} auctions, {len(nom)} nominal coupons, {out.auction_date.min().date()}..{out.auction_date.max().date()}")
    print(nom.term.value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
