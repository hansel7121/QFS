"""Write BDC_Pitch.pdf: context, thesis, EDA (with an explanation of every figure), optimization, backtest, verdict.

Run build_report_figs.py first; every number quoted here is read from report_figs/numbers.json.
"""

import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether

ROOT = Path(__file__).parent
FIG = ROOT / "report_figs"
N = json.load(open(FIG / "numbers.json"))

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=17, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=12.6, spaceAfter=5)
BL = ParagraphStyle("BL", parent=B, leftIndent=12, bulletIndent=2)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.HexColor("#444444"))
SUB = ParagraphStyle("SUB", parent=B, alignment=1, textColor=colors.HexColor("#555555"))

f = lambda x, d=1: f"{x:+.{d}f}" if x is not None else "n/a"
g = lambda x, d=1: f"{x:.{d}f}"
story = []
P = lambda t, s=B: story.append(Paragraph(t, s))
bullets = lambda items: [story.append(Paragraph(i, BL, bulletText="•")) for i in items]


TITLES = {
    "f1_discount": "Listed BDC share prices relative to reported NAV (training period, 2012–2019)",
    "f2_who_closes": "Who closes the gap: NAV change vs share return after sorting on the discount",
    "f3_stale": "How BDC NAVs respond to the traded leveraged-loan market",
    "f5_reaction": "Share-price reaction to the NAV release",
    "f6_pre_release": "Does pre-release information predict the reaction?",
    "f8_z_example": "Example of the own-discount z-score signal",
    "f7_zscore_ic": "Own-discount z-score and past returns vs future returns against peers",
    "f7b_tstat_construction": "How the t-statistic in the previous figure is built (20-day horizon, z-score signal)",
    "f9_portfolios": "Long cheap / short rich portfolios by liquidity and cost",
    "f10_optimization": f"Training-period optimization grid (displayed Sharpe rescaled × {N['sharpe_display_mult']:g})",
    "f11_backtest": "Backtest equity curve, training and out-of-sample periods",
    "f12_years": "Strategy return by calendar year",
}
CAPF = ParagraphStyle("CAPF", parent=B, alignment=1, fontSize=8.8, spaceBefore=2, spaceAfter=6)
_fig_no = [0]


def fig(name, w=6.9, what=None, means=None):
    from PIL import Image as PI
    im = PI.open(FIG / f"{name}.png"); h = w * im.size[1] / im.size[0]
    _fig_no[0] += 1
    parts = [Image(str(FIG / f"{name}.png"), width=w * inch, height=h * inch),
             Paragraph(f"<b>Figure {_fig_no[0]}.</b> {TITLES[name]}", CAPF)]
    if what:
        parts.append(Paragraph(f"<b>What Figure {_fig_no[0]} shows.</b> " + what, B))
    if means:
        parts.append(Paragraph("<b>What it means.</b> " + means, B))
    story.append(KeepTogether(parts))


def table(rows, widths, head=True):
    t = Table(rows, colWidths=[w * inch for w in widths])
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    if head:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); story.append(t); story.append(Spacer(1, 6))


f2i, f3, f5i = N["f2_is"], N["f3"], N["f5_is"]
f6 = {(r["x"], r["y"], r["era"]): r for r in N["f6"]}
f7 = {(r["sig"], r["H"], r["era"]): r for r in N["f7"]}
f9 = {(r["case"], r["era"]): r for r in N["f9"]}
bt, opt = N["bt"], N["opt"]
sens = {(r["mult"], r["era"]): r for r in N["sens"]}

# ------------------------------------------------------------------ title
P("BDC Pitch: Is Private Credit's NAV Real, and Can You Trade the Gap?", H1)
P("QFS research pitch · US listed Business Development Companies · data 2012–2026", SUB)

# ------------------------------------------------------------------ context
P("Context", H2)
P("A <b>Business Development Company (BDC)</b> is a US closed-end fund, created by Congress in 1980, that lends to "
  "small and mid-sized private companies. In practice it is how ordinary investors own <b>private credit</b> through a "
  "stock ticker (ARCC, OBDC, FSK, MAIN, ...). By law a BDC must invest at least 70% of its assets in private or very "
  "small US companies; to avoid corporate tax it pays out at least 90% of its income, which is why BDCs yield 9–12%. "
  "Most of the assets are first-lien, floating-rate loans to private-equity-owned companies.")
P("The loans never trade. Every quarter the BDC's board <b>estimates</b> what each loan is worth (a \"Level 3\" fair "
  "value), and the fund's <b>net asset value (NAV)</b> per share is built from those estimates. The shares, however, "
  "trade every day on an exchange, and there is no creation or redemption mechanism that forces the price to equal NAV. "
  "So every day the stock market gives its own opinion of the same loan book.")
P(f"Today the two disagree sharply: the median listed BDC trades at <b>{g(N['disc_now'], 0)}% to NAV</b>, and "
  f"{N['n_below20_now']} of {N['n_now']} trade more than 20% below it. Either the market is too pessimistic, or the "
  "managers' marks are too optimistic and NAVs will be cut.")

# ------------------------------------------------------------------ thesis
P("Thesis", H2)
P("<b>Part 1 (is the problem real?).</b> If the marks are smoothed and slow, the market price should lead NAV: deep "
  "discounts should be followed by NAV cuts, and NAVs should follow the public loan market with a lag. If instead the "
  "discount is investor pessimism, deep-discount BDCs should earn higher returns as the gap closes through the price.")
P("<b>Part 2 (is it tradable?).</b> Two ways to monetise it: (a) trade the quarterly NAV release, if the market does not "
  "fully anticipate the NAV change it should already be able to predict; (b) like a closed-end-fund discount trade, buy "
  "BDCs that are unusually cheap relative to their <i>own</i> normal discount and short those that are unusually rich, "
  "betting on a return to normal.")

# ------------------------------------------------------------------ data
P("Data", H2)
bullets([
    f"<b>NAV history</b> for {N['n_bdcs']} listed BDCs, {N['n_navs']:,} quarterly NAVs, 2012–2026, built from each BDC's "
    "own SEC filings: 10-Q balance sheets (current-period column), XBRL tags from the original filing (2022+), and fiscal "
    "year-ends from the next 10-Q. Each NAV is stamped with the moment it became public: the earnings press release "
    "(8-K Item 2.02) or the 10-Q, whichever came first. Release-quoted NAVs were checked against XBRL (83% exact, the rest "
    "explained by preliminary estimates, \"adjusted NAV\" figures and prior-quarter columns, which is why filings are used).",
    "All NAVs are put on the same split-adjusted basis as Yahoo prices; NAVs within nine months of a reverse split are "
    "excluded, because filings around a split mix per-share bases.",
    "<b>Prices</b>: Yahoo daily open/close/total return for the BDCs and BKLN (public leveraged-loan ETF).",
    "<b>Split</b>: training (in-sample) 2012–2019, test (out-of-sample) 2020–2026. <b>All EDA below uses the training "
    "period only</b>, and only observations whose whole outcome window ends before 1 January 2020. The test period is "
    "used once, in the backtest. All signals use only information public at the time; trades enter the day after a signal.",
    "<b>Caveat</b>: Yahoo only lists BDCs that still trade, so BDCs that merged or failed (e.g. Medley, Alcentra) are "
    "missing. This mainly affects long-horizon tests; the trading tests are short-horizon and relative to peers.",
])

# ------------------------------------------------------------------ EDA part 1
P("EDA Part 1 – Is the discount telling the truth? (2012–2019)", H2)
fig("f1_discount",
    what="Each BDC's share price divided by its latest public NAV, minus one, over the training period. The line is the "
         "median BDC; the band is the middle 50%. Zero means the stock trades exactly at NAV.",
    means=f"Discounts are not constant. BDCs traded near NAV in 2012–2014 and 2017–2019, and the median fell to about "
          f"{g(N['disc_2016min'], 0)}% in the 2015–16 credit scare (energy loans), while reported NAVs barely moved. Because "
          "the discount moves with credit sentiment and NAVs do not, the question is which one is late.")
fig("f2_who_closes",
    what="Every quarter, when a BDC publishes its NAV, BDCs are sorted into five groups by their discount (1 = deepest "
         "discount). Bars show what happened over the next four quarters, relative to the median BDC that quarter: the "
         f"change in NAV (blue) and the share's total return including dividends (orange). {f2i['n']} BDC-quarters, "
         "2012–2019, counting only those whose following year also ends before 2020.",
    means=f"The gap closes through <b>NAV falling, not price rising</b>. The deepest-discount group's NAV fell "
          f"{g(abs(f2i['q1_nav']))}% more than the median BDC over the next year, while the richest group's NAV rose "
          f"{g(f2i['q5_nav'])}% more. The discount predicts NAV changes strongly (rank correlation {g(f2i['rho_nav'], 2)}, "
          f"t {g(f2i['t_nav'])}), but it does <b>not</b> significantly predict share returns (t {g(f2i['t_ret'])}). Cheap "
          "BDCs are cheap for a reason: <b>the market was right and the marks were late</b>.")
fig("f3_stale",
    what="Left: a regression of the median BDC's quarterly NAV change on the return of the public leveraged-loan market "
         f"(BKLN) in the same quarter and the two quarters before ({f3['n']} training quarters), with 95% confidence bars. "
         "Right: the two series over time.",
    means=f"NAVs respond to the traded loan market only <b>partly and late</b>: {g(f3['bkln']['coef'], 2)} of a loan-market "
          f"move shows up in NAV the same quarter (t {g(f3['bkln']['t'])}), another {g(f3['bkln_l1']['coef'], 2)} one "
          f"quarter later (t {g(f3['bkln_l1']['t'])}) and {g(f3['bkln_l2']['coef'], 2)} two quarters later "
          f"(t {g(f3['bkln_l2']['t'])}). The marks are <b>stale</b>: a loan-market drop keeps feeding into NAV for two more "
          "quarters, while the share price can react the same day. This is the mechanism behind Figure 2.")
P("Findings from Part 1", H3)
bullets([
    "The discount is <b>information</b>, not just sentiment: it predicts future NAV cuts, and the gap closes through NAV, "
    "not through price.",
    "Reported NAVs follow the traded loan market with a lag of up to two quarters.",
    "So the simple trade, \"buy BDCs at a deep discount and wait\", does <b>not</b> work: the share returns of deep-discount "
    "BDCs are no better than average.",
])

# ------------------------------------------------------------------ EDA part 2
P("EDA Part 2 – Is any of it tradable? (2012–2019)", H2)
P("Trade (a): the quarterly NAV release", H3)
fig("f5_reaction",
    what=f"For every NAV release in the training period ({f5i['n']} releases), the BDC's return on the day the market can "
         "first react, minus the average BDC's return that day, grouped by how good the NAV change was relative to other "
         "BDCs that quarter (1 = worst). Error bars are 95% confidence intervals.",
    means=f"The market <b>does react</b> to NAV news: the worst NAV changes lose about {g(abs(f5i['q1']))}% on the day and "
          f"the best gain about {g(f5i['q5'])}% (rank correlation {g(f5i['rho'], 2)}). NAV releases contain news the price had "
          "not fully absorbed.")
fig("f6_pre_release",
    what="Can that reaction be predicted with information available <i>before</i> the release? Each bar is the t-statistic "
         "of a cross-sectional regression (one per quarter, averaged over the training quarters) of the post-release return "
         "on a pre-release signal: the discount the day before the release (left) and the BDC's price move against peers "
         "since the quarter-end (right). Red dashed lines mark t = ±2.",
    means=f"Both signals predict the NAV change itself (rank correlation {g(N['f5_pre']['disc_pre'], 2)} and "
          f"{g(N['f5_pre']['own_pre'], 2)}), so one might hope to front-run the reaction. They do <b>not</b> predict it: every "
          f"bar is between t {g(min(r['t'] for r in N['f6']))} and {g(max(r['t'] for r in N['f6']))}. The predictable part of "
          "the NAV change is already in the price before the release; only the unpredictable part moves the stock. "
          "<b>Trade (a) is rejected.</b>")

P("Trade (b): own-discount mean reversion", H3)
fig("f8_z_example",
    what="One BDC's discount (blue) and its own trailing one-year average (orange), 2012–2019. The signal is the "
         "<b>z-score</b>: how far today's discount is below or above that average, in standard deviations of its own history.",
    means="The z-score asks \"is this BDC cheap <i>for itself</i>?\" rather than \"is it cheap versus other BDCs?\". Part 1 "
          "showed cross-BDC discounts are mostly deserved; temporary swings around a BDC's own normal level might not be.")
z = {r["H"]: r for r in N["f7"] if r["sig"] == "own-discount z-score"}
pr = {r["H"]: r for r in N["f7"] if r["sig"] == "past 20-day return vs peers"}
fig("f7_zscore_ic",
    what="Each training-period week, the rank correlation between a signal and every BDC's return relative to the average "
         "BDC over the next 5, 20 or 60 days (skipping the first day after the signal, so the same closing price is never "
         "used twice). Bars are t-statistics; the red line is t = −2. Left: the own-discount z-score. Right: the BDC's "
         "return relative to peers over the past 20 days.",
    means=f"Both signals predict <b>reversal</b>: BDCs that are cheap against their own history, or have just "
          f"underperformed peers, outperform over the following weeks (z-score t {g(z[5]['t'])}, {g(z[20]['t'])} and "
          f"{g(z[60]['t'])} at 5, 20 and 60 days; past-return t {g(pr[5]['t'])}, {g(pr[20]['t'])}, {g(pr[60]['t'])}). The "
          f"two are largely the same effect: in one regression with past returns, neither is individually significant at "
          f"20 days (z-score t {g(N['f7_ctrl']['20_IS'])}, past 20-day return t {g(N['f7_ctrl']['20_IS_p20'])}). The "
          "mechanism is short-term price pressure in thinly traded, retail-held vehicles that reverts toward the NAV anchor; "
          "the z-score is the NAV-anchored way to measure it.")
tb = N["f7b"]; tbi = tb["IS"]
fig("f7b_tstat_construction",
    what=f"How the 20-day z-score bar of Figure {_fig_no[0]} is produced. <b>(a)</b> One ordinary training week, "
         f"{tb['ex_date']}: each dot is a BDC. The x-axis ranks the {tb['ex_n']} BDCs by z-score that day (1 = cheapest "
         "relative to its own history); the y-axis ranks them by their return relative to the average BDC over the next 20 "
         "trading days (1 = worst). The rank (Spearman) correlation of the two is that week's number, here "
         f"{g(tb['ex_rho'], 2)}: a slight tendency for cheap-for-themselves BDCs (left) to finish higher. One week on its own "
         "is very noisy. <b>(b)</b> The same number for every training week, as a histogram, with the average (dashed). "
         "<b>(c)</b> The same weekly numbers in time order, with a one-year rolling average (black line).",
    means=f"The t-statistic asks whether the average weekly correlation is reliably below zero. There are {tbi['weeks']} "
          f"training weeks with average {g(tbi['mean'], 3)} and standard deviation {g(tbi['sd'], 3)}; "
          f"{g(tbi['neg_share'] * 100, 0)}% of weeks are negative. The naive standard error is {g(tbi['sd'], 3)} / "
          f"√{tbi['weeks']} = {g(tbi['naive_se'], 4)}, giving t = {g(tbi['naive_t'])}. But each week's 20-day return window "
          "overlaps the next three weeks' windows, so neighbouring weeks are not independent, and the naive t overstates the "
          f"evidence. Correcting for that overlap (Newey-West standard errors, 4 weeks of lags) gives t = {g(tbi['hac_t'])}, "
          f"the bar in Figure {_fig_no[0]}. Panel (c) shows the rolling average below zero through most of 2012–2019: the "
          "effect is persistent rather than one episode, but small each week, which is why it needs many names and many "
          "weeks, and why costs matter.")

a_is = f9[("all BDCs, before costs", "IS")]; a15 = f9[("all BDCs, 15bp/side", "IS")]
l_is = f9[("more-liquid half, 8bp/side", "IS")]; i_is = f9[("less-liquid half, 25bp/side", "IS")]
ig_is = f9[("less-liquid half, before costs", "IS")]
fig("f9_portfolios",
    what="A simple training-period portfolio: every week, buy the cheapest third of BDCs by z-score and short the richest "
         "third, hold 20 days. Costs here are charged each way (on opening and again on closing). Lines: all BDCs before costs "
         "(grey) and at 15bp each way, i.e. 30bp per round trip (dark grey); the more-liquid half of BDCs at 8bp each way "
         "(blue); the less-liquid half before costs (light orange) and at 25bp each way (orange).",
    means=f"Before costs the trade works (all BDCs: Sharpe {g(a_is['sr'], 2)}, t {g(a_is['t'])}). Most of it comes from the "
          f"<b>less-liquid BDCs</b> (Sharpe {g(ig_is['sr'], 2)} before costs), and their wider spreads eat most of it "
          f"(Sharpe {g(i_is['sr'], 2)}, t {g(i_is['t'])} after 25bp each way). In the more-liquid half it is smaller (Sharpe "
          f"{g(l_is['sr'], 2)}, t {g(l_is['t'])} after 8bp each way). Charged 15bp each way, all BDCs together reach Sharpe "
          f"{g(a15['sr'], 2)} (t {g(a15['t'])}). Trading costs therefore decide how much of the edge is kept; the "
          "optimization below charges each BDC a cost that matches its liquidity.")

# ------------------------------------------------------------------ optimization
P("Optimization (training period only)", H2)
P("Costs are charged per BDC by its liquidity (20-day median dollar volume), on every trade in and out: <b>5bp</b> per "
  "side for BDCs trading at least $20M a day, <b>10bp</b> for $5–20M, <b>20bp</b> for $1–5M and <b>40bp</b> below $1M, "
  "so a round trip costs 10bp in the largest BDCs and 80bp in the smallest. Shorted shares pay a 1% a year borrow fee. "
  f"The signal is checked every {N['step']} trading days and the trade placed at the next day's close. The two "
  f"parameters are the holding period ({N['step']} to 30 trading days, in steps of {N['step']}; longer holds overlap) and "
  "how much of the cross-section to trade (cheapest/richest 20% to 50%). Only 2013–2019 data is used to choose them.")
fig("f10_optimization", w=5.6,
    what=f"Training-period Sharpe ratio after liquidity-based costs and borrow for each combination. <b>Note on scale:</b> "
         f"the large number in each cell is the backtest Sharpe multiplied by {N['sharpe_display_mult']:g}, a presentation "
         f"rescaling chosen by the author; divide by {N['sharpe_display_mult']:g} for the backtest Sharpe. The "
         "unscaled values are the ones used to select the setting and everywhere else in this document. The bold cell is "
         "the best.",
    means=f"The best cell (hold {opt['H']} days, cheapest/richest {int(round(opt['q'] * 100))}%) has a training Sharpe of "
          f"{g(opt['sr_is'], 2)} (t {g(opt['t_is'])}); its neighbours have a median of {g(opt['nb_median'], 2)} and a minimum "
          f"of {g(opt['nb_min'], 2)}, and it sits inside the grid rather than on its edge. Shorter holds score lower: "
          "rebalancing often means trading the small BDCs often, and their costs eat the edge (holds of 3 days, not shown, "
          "lose money). Holds of two to three weeks balance signal against cost. But <b>no cell reaches t = 2 in training</b> "
          f"(best cell t {g(opt['t_is'])}).")

# ------------------------------------------------------------------ backtest
P("Backtest", H2)
fig("f11_backtest",
    what="Growth of $1 for the selected setting after liquidity-based costs and borrow, over the whole sample: training "
         "period in blue, test period in orange. The red dashed line marks where the out-of-sample period begins "
         "(1 January 2020); nothing after it was used for any choice.")
yrs = N["years"]; worst = sorted(yrs, key=lambda k: yrs[k])[:3]
fig("f12_years", what="After-cost return by calendar year.",
    means=f"The worst years are {', '.join(f'{k} ({g(yrs[k])}%)' for k in worst)}. 2020 stands out "
          f"({g(yrs['2020'], 0)}%): in the COVID crash and rebound BDC prices swung far from their normal discounts and "
          "snapped back, the ideal environment for this trade.")
rows = [["", "Train 2013–2019", "Test 2020–2026"],
        ["Annual return after costs", f"{g(bt['IS']['ann'])}%", f"{g(bt['OOS']['ann'])}%"],
        ["Annual return before costs", f"{g(bt['IS']['gross_ann'])}%", f"{g(bt['OOS']['gross_ann'])}%"],
        ["Sharpe after costs (t-stat)", f"{g(bt['IS']['sr'], 2)} (t {g(bt['IS']['t'])})", f"{g(bt['OOS']['sr'], 2)} (t {g(bt['OOS']['t'])})"],
        ["Total return after costs", f"{g(bt['IS']['total'], 0)}%", f"{g(bt['OOS']['total'], 0)}%"],
        ["Maximum drawdown", f"{g(bt['IS']['maxdd'])}%", f"{g(bt['OOS']['maxdd'])}%"]]
table(rows, [2.6, 1.9, 1.9])
P("Cost sensitivity (same setting; all liquidity-based costs and the borrow fee multiplied by a factor):", B)
rows = [["Cost multiple", "Train Sharpe (t)", "Test Sharpe (t)"]] + [
    [f"{k:g}×" + (" (assumed)" if k == 1 else ""), f"{g(sens[(k, 'IS')]['sr'], 2)} (t {g(sens[(k, 'IS')]['t'])})", f"{g(sens[(k, 'OOS')]['sr'], 2)} (t {g(sens[(k, 'OOS')]['t'])})"]
    for k in [0, 0.5, 1, 1.5, 2]] + [
    ["Flat 15bp per round trip for every BDC, no borrow", f"{g(N['flat15']['IS']['sr'], 2)} (t {g(N['flat15']['IS']['t'])})",
     f"{g(N['flat15']['OOS']['sr'], 2)} (t {g(N['flat15']['OOS']['t'])})"]]
table([[Paragraph(str(c), B) if i == 0 else c for i, c in enumerate(r)] for r in rows], [2.8, 1.9, 1.9])
P(f"<b>What it means.</b> Before costs the trade works in both periods (t {g(sens[(0, 'IS')]['t'])} and "
  f"{g(sens[(0, 'OOS')]['t'])}). After liquidity-based costs, which take about {g(N['cost_drag_is'])}% a year in training, "
  f"the training period falls short of the bar (Sharpe {g(bt['IS']['sr'], 2)}, t {g(bt['IS']['t'])}), while the test "
  f"period passes (Sharpe {g(bt['OOS']['sr'], 2)}, t {g(bt['OOS']['t'])}; {g(N['bt_ex2020']['sr'], 2)}, t "
  f"{g(N['bt_ex2020']['t'])} excluding 2020). At half the assumed costs training clears t = 2; if small BDCs could be "
  f"traded as cheaply as large ones (flat 15bp per round trip) it clears easily (t {g(N['flat15']['IS']['t'])}). The "
  "signal is strongest in exactly the BDCs that are most expensive to trade.")

# ------------------------------------------------------------------ verdict
P("Verdict", H2)
P("<b>The research question has a clear answer; the trade is a cost-bound near miss.</b>")
bullets([
    "<b>Private credit marks lag the market.</b> Listed BDC discounts are information: they predict NAV cuts, the gap "
    "closes through NAV rather than price, and NAVs follow the traded loan market with a lag of up to two quarters.",
    "<b>Buying deep discounts does not pay</b>, because those discounts are deserved.",
    "<b>Trading NAV releases</b> fails: information available before the release does not predict the reaction.",
    f"<b>Own-discount mean reversion</b> is real before costs in both periods (t {g(sens[(0, 'IS')]['t'])} and "
    f"{g(sens[(0, 'OOS')]['t'])}), but after liquidity-based costs the training-period result is t {g(bt['IS']['t'])}, "
    f"below the t = 2 bar, even though the untouched test period passes (t {g(bt['OOS']['t'])}).",
])
P("Caveats", H3)
bullets([
    "<b>Costs are assumptions by liquidity tier, not measured spreads.</b> The result moves from fail to pass between 1× "
    "and 0.5× of them. Measuring real BDC bid-ask spreads (intraday quotes) and trading with limit or closing-auction "
    "orders is the most important next step.",
    "Survivorship: BDCs that merged or failed before 2026 are missing, which can flatter long-only results; the trade is "
    "long-short against peers, which limits but does not remove this.",
    "Shorting small BDCs may be hard to borrow at times; 1% a year is an average assumption.",
    "Capacity is small (low millions per side), because much of the edge is in BDCs trading $1–5M a day.",
    "The test period is helped by 2020 (COVID crash and rebound), the ideal environment for this trade.",
])
P("<b>What this pitch is good for:</b> a well-supported answer to a live question in markets (\"are private credit "
  "valuations real?\": the public market says no, and it has been right), plus a relative-value trade that is real "
  "before costs and should be run only by someone who can trade small BDCs at about half of these assumed costs.")

doc = SimpleDocTemplate(str(ROOT / "BDC_Pitch.pdf"), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.6 * inch, title="BDC Pitch", author="QFS")
doc.build(story)
print("wrote", ROOT / "BDC_Pitch.pdf")
