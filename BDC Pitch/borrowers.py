"""Match the same borrower across BDCs whose schedules of investments describe positions in different formats.

Some BDCs start each identifier with the borrower ("Acme Corp | First lien senior secured loan"); others bury it
("Debt Investments Software Acme Corp First Lien Term Loan SOFR+6.00% ..."). Approach:
  1. Split every identifier on "|" and ",", normalise each segment (lower case, punctuation and legal suffixes removed).
  2. Vocabulary = segments that look like a company: they label only a few positions in any one filing (generic labels
     such as "first lien" or "software" label hundreds) and are not tranche/industry words.
  3. A position's borrower = the longest vocabulary entry found as a whole-word n-gram anywhere in its identifier.
"""

import re
import pandas as pd

SUFFIX = re.compile(r"\b(incorporated|inc|llc|lp|llp|corp|corporation|co|company|ltd|limited|holdings?|holdco|group|plc|"
                    r"gmbh|sarl|sa|bv|nv|ag|ab|as|the|parent|intermediate|topco|midco|bidco|buyer|acquisition|acquisitions|"
                    r"acquisitionco|purchaser|borrower|us|usa|newco|opco|aggregator|investor|investors|lux)\b")
GENERIC = re.compile(r"\b(lien|secured|senior|loan|loans|debt|investment|investments|affiliated|issuer|equity|warrant|"
                     r"warrants|preferred|common|units?|notes?|bond|bonds|revolv\w*|delayed|draw|term|tranche|unitranche|"
                     r"subordinated|mezzanine|sofr|libor|prime|floor|spread|maturity|industry|interest|rate|pik|cash|"
                     r"services|software|healthcare|technology|financial|insurance|other|clo|total|non|controlled|"
                     r"first|second|1st|2nd|last|out|funded|unfunded|related|party|united|states|kingdom|canada|january|february|march|"
                     r"april|may|june|july|august|september|october|november|december|portfolio|companies|industries|"
                     r"products|equipment|retail|real|estate|health|care|media|energy|chemicals|consumer|commercial|"
                     r"business|professional|capital|goods|materials|transportation|telecommunications|construction)\b")


def clean(s):
    s = re.sub(r"\((f/k/a|fka|d/b/a|dba|aka|formerly)[^)]*\)?", " ", s, flags=re.I)
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.lower().replace("&", " and ").replace(".", "").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = SUFFIX.sub(" ", s)
    s = re.sub(r"\b\d+(\s\d+)*\b", " ", s)       # numbers: tranche counters, percentages, dates
    s = re.sub(r"\band\b\s*$|^\s*and\b", " ", s)
    return " ".join(s.split())


def segments(ident):
    return [clean(x) for x in re.split(r"[|,;]", ident)]


def build_vocab(pos, max_per_filing=12):
    """Return {name: number of BDCs whose identifiers produced it}."""
    seg = pos[["ticker", "acc", "identifier"]].drop_duplicates().copy()
    seg["seg"] = seg.identifier.map(segments)
    seg = seg.explode("seg")
    seg = seg[seg.seg.str.len() >= 4]
    per_filing = seg.groupby(["acc", "seg"]).identifier.nunique().groupby("seg").max()
    src = seg.groupby("seg").ticker.nunique()
    keep = [x for x in per_filing[per_filing <= max_per_filing].index
            if not GENERIC.search(x) and (len(x.split()) >= 2 or len(x) >= 6)]
    return src[keep].to_dict()


def match(ident, vocab, max_n=7):
    """Among all vocabulary n-grams in the identifier, take the one used by the most BDCs, then the longest."""
    toks = clean(ident).split()
    best, key = None, (0, 0)
    for n in range(min(max_n, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i:i + n])
            k = (vocab.get(g, 0), n)
            if k[0] and k > key:
                best, key = g, k
    return best


def assign(pos, max_ids=15, rounds=3):
    """Borrower per position. Names that end up matching more than max_ids positions in one filing are generic
    labels, not companies: drop them from the vocabulary and match again."""
    vocab = build_vocab(pos)
    uniq = pos[["acc", "identifier"]].drop_duplicates()
    for _ in range(rounds):
        cache = {s: match(s, vocab) for s in uniq.identifier.unique()}
        b = uniq.identifier.map(cache)
        cnt = uniq.assign(b=b.values).groupby(["acc", "b"]).identifier.nunique().groupby("b").max()
        bad = cnt[cnt > max_ids].index
        if len(bad) == 0:
            break
        for x in bad:
            vocab.pop(x, None)
    return pos.identifier.map(cache), vocab
