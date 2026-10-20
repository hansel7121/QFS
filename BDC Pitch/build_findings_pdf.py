"""Write BDC_Findings.pdf: review of the Private Credit NAV write-up and the follow-up tests on whether the
own-discount z-score is just short-term reversal.

Run first: build_report_figs.py, build_residual_test.py, build_residual_conditional.py, build_reversal_check.py,
collect_benchmark_prices.py, build_followup.py. Every number quoted is read from report_figs/*.json.
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
ld = lambda n: json.load(open(FIG / n))
N, RN, RC, RV, FU = (ld("numbers.json"), ld("residual_numbers.json"), ld("residual_conditional_numbers.json"),
                     ld("reversal_check_numbers.json"), ld("followup_numbers.json"))
NE = ld("neutral_numbers.json")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=17, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=12.6, spaceAfter=5)
BL = ParagraphStyle("BL", parent=B, leftIndent=12, bulletIndent=2)
CAPF = ParagraphStyle("CAPF", parent=B, fontSize=8.5, textColor=colors.HexColor("#333333"))
SUB = ParagraphStyle("SUB", parent=B, alignment=1, textColor=colors.HexColor("#555555"))
BOX = ParagraphStyle("BOX", parent=B, backColor=colors.HexColor("#f3f6fa"), borderColor=colors.HexColor("#2f5d8a"),
                     borderWidth=0.6, borderPadding=6, spaceBefore=4, spaceAfter=10)

story = []
P = lambda t, s=B: story.append(Paragraph(t, s))
bullets = lambda items: [story.append(Paragraph(i, BL, bulletText="•")) for i in items]
_fig_no = [0]
t2 = lambda x: f"{x:.2f}"
t1 = lambda x: f"{x:.1f}"


def fig(name, title, w=6.9, what=None, means=None):
    from PIL import Image as PI
    im = PI.open(FIG / f"{name}.png"); h = w * im.size[1] / im.size[0]
    _fig_no[0] += 1
    parts = [Image(str(FIG / f"{name}.png"), width=w * inch, height=h * inch),
             Paragraph(f"<b>Figure {_fig_no[0]}.</b> {title}", CAPF)]
    if what: parts.append(Paragraph(f"<b>What Figure {_fig_no[0]} shows.</b> " + what, B))
    if means: parts.append(Paragraph("<b>What it means.</b> " + means, B))
    story.append(KeepTogether(parts))


def table(rows, widths, head=True):
    t = Table(rows, colWidths=[w * inch for w in widths])
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    if head:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); story.append(t); story.append(Spacer(1, 6))


# ---------------------------------------------------------------- numbers
bt, opt, F1, F2, F3, F4 = N["bt"], N["opt"], FU["F1"], FU["F2"], FU["F3"], FU["F4"]
rn = {(r["signal"], r["H"]): r["t"] for r in RN["table"]}
rc = {(r["group"], r["H"]): r["t"] for r in RC["table"]}
rvA, rvC = RV["A"], {(r["k"], r["H"]): r for r in RV["C"]}
rvB = {(r["test"], r["k"], r["H"], r["group"]): r for r in RV["B"]}
leak = RV["B_leak_bp"]
f4 = {(r["group"], r["H"]): r for r in F4}
f7 = {(r["sig"], r["H"]): r["t"] for r in N["f7"]}
best = F3["best"]
ZS, ZR, RVS = "z-score", "reversal-free z-score", "plain reversal (past 20d)"

# ---------------------------------------------------------------- title + summary
P("Private Credit NAV: Review and Follow-up Tests", H1)
P("Is the own-discount z-score a NAV signal, or short-term reversal in disguise? &nbsp;·&nbsp; 17 October 2026", SUB)
P("<b>Bottom line.</b> Most of what the z-score picks up is short-term reversal: BDCs that just fell bounce back, and BDCs "
  "that just rose fade. Between NAV releases the z-score moves only because the price moves, so it is a past-return signal by "
  "construction. Three findings keep the idea alive, though. (1) Reversal is much stronger in BDCs than in similar stocks "
  f"(mortgage REITs, regional banks). (2) The z-score strategy still earns about {t1(F2['IS']['multi']['alpha_ann'])}% a year "
  f"in training (t = {t1(F2['IS']['multi']['t_alpha'])}) and {t1(F2['OOS']['multi']['alpha_ann'])}% a year out of sample "
  f"(t = {t1(F2['OOS']['multi']['t_alpha'])}) that reversal portfolios at the 5-, 20-, 60- and 250-day horizons cannot explain. "
  f"(3) It trades less than plain reversal, so it survives costs where plain reversal does not. "
  f"The weak point remains: the training-period Sharpe after costs is only {t2(opt['sr_is'])} (t = {t2(opt['t_is'])}), not the "
  f"{t2(opt['sr_is'] * 2)} the write-up's heatmap shows. The pitch is defensible only if it is reframed as <i>a NAV-anchored, "
  "low-turnover form of BDC reversal</i>, not as \"the discount pulls back to normal\".", BOX)

# ---------------------------------------------------------------- 1. write-up problems
P("1. Problems in the current write-up", H2)
P("These need fixing before anything else. Each was checked against the code (build_report_figs.py) and its output.")
bullets([
    f"<b>The heatmap shows Sharpe × 2.</b> build_report_figs.py multiplies the displayed grid by SHARPE_DISPLAY_MULT = 2. The PDF crops "
    f"the title that says so. The true best training Sharpe is {t2(opt['sr_is'])} with t = {t2(opt['t_is'])}; the highest t "
    f"anywhere in the grid is {t2(F1['max_t_in_grid'])}. Figure 1 shows the true grid. This has to be corrected.",
    f"<b>Chosen cell.</b> The text says \"33% and 12 days\"; the code chose {opt['q']:.0%} and {opt['H']} days.",
    f"<b>Out-of-sample Sharpe</b> is {t2(bt['OOS']['sr'])}, not 1.5. Without 2020 it is {t2(N['bt_ex2020']['sr'])} "
    f"(t = {t2(N['bt_ex2020']['t'])}).",
    "<b>Rebalancing.</b> The text says weekly; the code re-ranks every 3 days with overlapping 12-day holds.",
    "<b>Background facts.</b> Congress created BDCs in 1980 (Small Business Investment Incentive Act), not after 2008; ARCC listed "
    "in 2004. Private credit lenders are non-bank <i>institutional</i> lenders. NAV comes out first in the earnings release, "
    "then in the 10-Q (Q1–Q3) or 10-K (Q4); the code correctly uses the earliest public date. \"30 CBDs\" is a typo.",
    "<b>Out-of-sample data in the EDA.</b> The price/NAV chart runs to 2026; EDA should stop at 2019.",
    "<b>Survivorship.</b> Prices come from a hard-coded list of BDCs that are listed today, so BDCs that merged or delisted in "
    "2012–2019 are missing. State this as a caveat at minimum.",
    "<b>No mechanism.</b> The write-up says the discount \"drifts back\" but not what pushes it back.",
])
fig("g1_heatmap_true", "Training-period (2013–2019) Sharpe after costs, without the ×2 display multiplier.",
    w=5.2,
    what="The same grid as the write-up's heatmap, but with the real Sharpe ratios and their HAC t-stats.",
    means=f"No setting reaches t = 2 in training. The grid is flat (0.3–0.56), which is good for robustness, but the "
          "in-sample evidence is weak on its own.")

# ---------------------------------------------------------------- 2. the reversal question
P("2. Is the z-score just past returns?", H2)
P("<b>Reversal</b> means a stock that fell recently tends to do better next, and one that rose tends to do worse, as temporary "
  "selling or buying pressure fades. It shows up in almost all stocks, strongest in small, illiquid, retail-held ones. "
  "The worry: a BDC gets \"cheap versus its own history\" mostly by its price falling, so the z-score may just pick recent losers.")
fig("f7e_reversal_check", "Is the z-score just reversal? Premise, double sort and regression (2012–2019).",
    what=f"(a) Share of the day-to-day change in z coming from each source: price {rvA['var_share']['price']:.0%}, NAV updates "
         f"{rvA['var_share']['NAV update']:.0%} (only {rvA['share_days_with_nav_update']:.1%} of days), drift of the trailing "
         f"average {rvA['var_share']['rolling mean/std']:.1%}. Over 20 days the change in z has a correlation of "
         f"{rvA['corr_dz20_with_price_part']:.2f} with its price part. "
         "(b) Double sort: each week, BDCs go into 3 groups by past 20-day return; within each group the cheap half by z is "
         "compared with the rich half over the next 20 days. "
         "(c) Each week, regress next-h-day return vs peers on z and past 20-day return together; t-stats of the averaged slopes.",
    means=f"(a) confirms the premise: between NAV releases the z-score is a price signal. "
          f"(b) looks favourable, since the cheap half beats the rich half in every group (average "
          f"{rvB[('z within past-return groups', 20, 20, 'average')]['spread_bp']:.0f}bp, t = "
          f"{t1(rvB[('z within past-return groups', 20, 20, 'average')]['t'])}). But the groups are too coarse: inside each group "
          f"the cheap half still had a past-60-day return about {abs(leak['mid']['p60']) / 100:.0f}–{abs(leak['low']['p60']) / 100:.0f}% "
          f"lower than the rich half, so longer-horizon reversal leaks through. "
          f"(c) is the cleaner test. With past 20-day return in the regression, z is significant over the next 5 days "
          f"(t = {t2(rvC[(20, 5)]['t_b'])}) but not over 20 days (t = {t2(rvC[(20, 20)]['t_b'])}) or 60 days "
          f"(t = {t2(rvC[(20, 60)]['t_b'])}).")
table([["Control in the regression", "z: next 5d", "z: next 20d", "z: next 60d"]] +
      [[f"past {k}-day return", t2(rvC[(k, 5)]["t_b"]), t2(rvC[(k, 20)]["t_b"]), t2(rvC[(k, 60)]["t_b"])] for k in (5, 20, 60)],
      [2.4, 1.2, 1.2, 1.2])
P("Over one week, z predicts beyond past returns whatever the control. Over the 12-day hold and beyond, it does not, once "
  "20-day or 60-day returns are controlled for.")

fig("f7c_residual", "Signal strength before and after removing past returns (2012–2019).",
    what="Each week, z is regressed on past returns across BDCs and only the residual is kept: the part of \"cheap versus own "
         "history\" that recent price moves do not explain. The write-up's rank-correlation test is then rerun on that residual.",
    means=f"Past 20-day return explains a median {RN['r2_median']:.0%} of the z-score each week. With past 5/20/60-day returns "
          f"removed, t falls from {t2(f7[('own-discount z-score', 20)])} to {t2(rn[('z-score, past 5/20/60d returns removed', 20)])} "
          f"at 20 days. Plain reversal alone gives t = {t2(rn[('past 20d return vs peers', 20)])}.")

fig("f7d_residual_conditional", "Where does the reversal-free z-score still work? (2012–2019)",
    what="The residual signal, split where the mechanism says it should be strongest: right after a NAV release (fresh NAV, 15 "
         "trading days or fewer) versus later, and by discount level (premium, middle, deep discount).",
    means=f"Two of 15 cells pass −2: BDCs at a premium over the next 5 days (t = {t2(rc[('premium (price ≥ NAV)', 5)])}), which fits "
          f"share issuance above NAV capping the premium, and fresh NAV over 60 days (t = {t2(rc[('fresh (≤15d since NAV)', 60)])}, "
          f"but only about 7 BDCs a week). Deep discounts, where buybacks should help, show nothing. With 15 tests, about one "
          "false pass is expected by chance, so this is suggestive at best.")

# ---------------------------------------------------------------- 3. portfolio level
P("3. At the portfolio level the z-score does beat reversal", H2)
P("The tests above ask whether z ranks next returns better than past returns do. A different question matters for trading: does "
  "the z-score <i>strategy</i> earn anything that reversal <i>strategies</i> cannot replicate? To answer it, I built plain "
  "reversal portfolios with exactly the same rules (same hold, same buckets, same costs) and regressed the z-score strategy's "
  "daily returns on them.")
fig("g2_reversal_attribution", "z-score strategy versus plain reversal strategies with the same rules.",
    what=f"(a) Growth of $1 before costs for the z-score strategy and a 20-day reversal strategy (hold {opt['H']} days, "
         f"{opt['q']:.0%} buckets). (b) The z-score strategy's annual return before costs, then what is left after removing the "
         "part explained by 20-day reversal, and by reversal at 5, 20, 60 and 250 days together.",
    means=f"The two strategies have a correlation of {t2(F2['IS']['corr'])} in training: related, but far from the same. With all "
          f"four reversal horizons controlled for, the z-score strategy keeps {t1(F2['IS']['multi']['alpha_ann'])}% a year in "
          f"training (t = {t2(F2['IS']['multi']['t_alpha'])}) and {t1(F2['OOS']['multi']['alpha_ann'])}% a year out of sample "
          f"(t = {t2(F2['OOS']['multi']['t_alpha'])}). Reversal explains about half its daily variance "
          f"(R² = {t2(F2['IS']['multi']['r2'])}).")
table([["Before costs → after costs", "z-score, 2013–19", "reversal, 2013–19", "z-score, 2020–26", "reversal, 2020–26"],
       ["Annual return (%)",
        f"{t1(F2['IS']['z_gross']['ann'])} → {t1(F2['IS']['z_net']['ann'])}", f"{t1(F2['IS']['rev_gross']['ann'])} → {t1(F2['IS']['rev_net']['ann'])}",
        f"{t1(F2['OOS']['z_gross']['ann'])} → {t1(F2['OOS']['z_net']['ann'])}", f"{t1(F2['OOS']['rev_gross']['ann'])} → {t1(F2['OOS']['rev_net']['ann'])}"],
       ["Sharpe",
        f"{t2(F2['IS']['z_gross']['sr'])} → {t2(F2['IS']['z_net']['sr'])}", f"{t2(F2['IS']['rev_gross']['sr'])} → {t2(F2['IS']['rev_net']['sr'])}",
        f"{t2(F2['OOS']['z_gross']['sr'])} → {t2(F2['OOS']['z_net']['sr'])}", f"{t2(F2['OOS']['rev_gross']['sr'])} → {t2(F2['OOS']['rev_net']['sr'])}"]],
      [1.7, 1.3, 1.3, 1.3, 1.3])
P(f"<b>Why the z-score survives costs and reversal does not:</b> the cost drag is {t1(F2['cost_drag_ann']['z-score'])}% a year "
  f"for the z-score strategy against {t1(F2['cost_drag_ann']['reversal 20d'])}% for plain reversal. The z-score compares the price "
  "with its own one-year average and scales by the BDC's own volatility, so it moves more slowly and turns over less. "
  "In plain terms: <i>the NAV anchor does not add much new information, but it turns BDC reversal into a cheaper-to-trade "
  "signal.</i>")
P("<b>The two kinds of test disagree, and both should be reported.</b> The cross-sectional tests (Figures 2–4) say z adds little "
  "beyond past returns at 20 days. The portfolio test (Figure 5) says the z-score strategy earns a return reversal portfolios "
  "cannot replicate. Both can hold at once: z blends reversal across many horizons and scales it by each BDC's own volatility, "
  "which no single past-return control captures. That blend is still mostly reversal.")

# ---------------------------------------------------------------- 4. short hold
P("4. Does a shorter hold or a cleaner signal help?", H2)
fig("g3_short_hold", "Re-optimizing the holding period for three signals (training 2013–2019, then out of sample).",
    what="Three signals, each traded long-cheap/short-rich with daily-staggered tranches and the same tiered costs plus 1% borrow: "
         "the z-score, the reversal-free z-score (z with past 5/20/60-day returns removed each day), and plain 20-day reversal. "
         "(a) Training Sharpe by hold, before (dashed) and after (solid) costs. (b) The best training setting for each signal, "
         "run through the out-of-sample period.",
    means=f"Shorter holds look best before costs (z-score: {t2([r for r in F3['grid'] if r['signal'] == ZS and r['H'] == 3 and abs(r['q'] - 1 / 3) < 1e-6][0]['is_gross']['sr'])} "
          f"at 3 days), but costs wipe them out. The one-week effect from Section 2 is real but cannot be traded at these costs. "
          f"The reversal-free z-score is weak even before costs (training Sharpe about 0.4–0.7) and loses money after costs, so the "
          f"\"pure NAV\" part is not a standalone trade. Plain reversal also loses money after costs in training "
          f"(best {t2(best[RVS]['is_net']['sr'])}). The original z-score at {best[ZS]['H']} days remains the best choice "
          f"(training {t2(best[ZS]['is_net']['sr'])}, out of sample {t2(best[ZS]['oos_net']['sr'])}).")

# ---------------------------------------------------------------- 5. BDCs special?
P("5. Is reversal special in BDCs?", H2)
fig("g4_reversal_groups", "Short-term reversal in BDCs versus mortgage REITs and regional banks (2012–2019).",
    w=5.4,
    what="Average weekly rank correlation between past 20-day return vs peers and the next 5 or 20 days' return vs peers, in three "
         "groups of listed financials. Mortgage REITs are the closest match to BDCs: leveraged, high-dividend, retail-held, and "
         "priced against a book value. Regional banks are a second comparison group.",
    means=f"Reversal over 20 days is about four times as strong in BDCs ({f4[('BDCs', 20)]['ic']:.3f}, t = {t1(f4[('BDCs', 20)]['t'])}) "
          f"as in mortgage REITs ({f4[('mREIT', 20)]['ic']:.3f}) and more than twice as strong as in regional banks "
          f"({f4[('regional bank', 20)]['ic']:.3f}). The gap is significant: BDCs minus mortgage REITs t = "
          f"{t1(f4[('BDCs minus mREIT', 20)]['t'])}, BDCs minus banks t = {t1(f4[('BDCs minus regional bank', 20)]['t'])}. "
          "Something about BDCs makes their price moves unusually temporary. That is the structural story the pitch needs.")
P("Caveat: both comparison groups also use tickers listed today, so they share the survivorship issue. Mortgage REITs also have a "
  "frequently updated book value, which may itself reduce reversal; that difference is worth one sentence in the pitch.")

# ---------------------------------------------------------------- 6. sector / credit exposure
V_ = NE["versions"]; O_, F_ = V_["z-score | equal"], V_["relative z-score | beta-neutral"]
eo, ef = O_["exposure"], F_["exposure"]
P("6. Hidden sector and credit exposure, and a neutral version", H2)
P("<b>The concern.</b> Equal dollars long and short does not mean equal sensitivity to the sector. If the cheap BDCs are the "
  "weaker, more leveraged ones, the long leg has a higher beta, so the book is secretly long the BDC sector and long credit risk. "
  "Rebounds like 2020 would then look like alpha, and a sector-wide widening of discounts (2025–26) would show up as losses. "
  "A second, related issue is in the signal itself: when every BDC's discount widens together, each one looks \"cheap versus its "
  "own history\" at once, so the signal partly tracks sector-wide moves rather than one BDC deviating from the rest.")
P("<b>Two fixes, both with the write-up's parameters unchanged</b> (no re-optimization):")
bullets([
    "<b>Relative z-score (the signal).</b> Before the z-score, subtract that day's median BDC discount from each BDC's discount. "
    "A sector-wide move in discounts, or in NAVs, then leaves the signal unchanged. The signal only measures how far a BDC "
    "has moved away from the rest of the sector, compared with its own usual gap.",
    "<b>Beta-neutral weights (the portfolio).</b> Each BDC's beta to the equal-weight BDC sector is estimated on the 252 trading "
    "days before each trade. The two legs are then sized so that long beta equals short beta (gross exposure stays 2×). "
    "An alternative, hedging the leftover beta with BIZD, gives very similar results.",
])
fig("n3_beta_method", "How the long- and short-basket betas are computed (write-up version).",
    what="(a) One BDC on one trade date: its daily returns against the equal-weight BDC sector over the previous 252 trading days; "
         "the fitted slope (covariance / variance) is its beta. (b) The same beta re-estimated every day, using only past data. "
         "(c) On that trade date, every BDC's beta coloured by basket; a basket's beta is the average of its members (equal dollars). "
         "(d) Basket betas over time. Solid lines are ex-ante (known when trading), dotted lines are realized (rolling 252-day beta "
         "of each basket's actual returns).",
    means="Ex-ante and realized basket betas agree: the long basket runs only slightly higher than the short basket on average. "
          "That small average gap understates the problem. Panel (b) of Figure 10 shows the exposure is asymmetric: "
          "the cheap basket gains much more than the rich basket in strong sector weeks.")
fig("n4_portfolio_beta_is", "Beta of the neutral portfolio (relative z-score, beta-neutral) to the BDC sector, training period 2013–2019.",
    what="(a) Ex-ante book beta: each day, every position held times its dollar weight times that BDC's trailing-252-day beta, "
         "longs minus shorts. (b) Realized beta: rolling 126-day regression of the strategy's daily return on the sector's.",
    means="The book stays near zero beta ex-ante (average +0.003, range about −0.07 to +0.25) and has no significant realized "
          "beta over the period (−0.01, t = −0.3). Over any six-month window the realized beta can still drift (about −0.25 to "
          "+0.55), because trailing betas never predict future sensitivity perfectly. Because the rich names have lower betas, "
          "the short leg is larger and the book is on average about 11% net short in dollars: beta-neutral, not dollar-neutral.")
fig("n1_exposure", "Sector exposure of the write-up version and of the neutralized versions (2013–2026).",
    what="(a) Average ex-ante beta of the long and short legs in the write-up version. (b) Average weekly strategy return, with "
         "weeks sorted into fifths by the BDC sector's return that week. (c) Realized daily beta of each version's returns to the "
         "BDC sector (HAC t-stats).",
    means=f"The concern is partly right. Ex-ante, the legs look balanced ({O_['leg_beta']['long']:.2f} vs {O_['leg_beta']['short']:.2f}), "
          f"but the write-up version still carries a realized sector beta of {eo['all']['beta_sector']:.2f} "
          f"(t = {t1(eo['all']['t_sector'])}), and its profits are concentrated in the sector's best weeks "
          f"({O_['sector_weeks']['5']:+.2f}% a week) against {O_['sector_weeks']['1']:+.2f}% in the worst. Cheap BDCs bounce harder when "
          f"the whole sector rallies. The relative, beta-neutral version has a sector beta of {ef['all']['beta_sector']:.2f} "
          f"(t = {t1(ef['all']['t_sector'])}), and it earns money in the worst sector weeks too ({F_['sector_weeks']['1']:+.2f}% a week). "
          f"The credit part of the concern does not hold up: the strategy's loading on changes in the Baa–Treasury spread is "
          f"small and insignificant (t = {t1(eo['all']['t_baa'])}), and it has the wrong sign for a hidden long-credit position.")
fig("n2_neutral_versions", "Write-up version versus versions with sector exposure removed (after costs).",
    what="Growth of $1 after costs, and Sharpe in training and out of sample, for both signals with each weighting scheme.",
    means=f"Removing the sector exposure costs almost nothing. The relative, beta-neutral version has the same training Sharpe "
          f"({t2(F_['IS']['sr'])} vs {t2(O_['IS']['sr'])}) and a higher out-of-sample Sharpe "
          f"({t2(F_['OOS']['sr'])} vs {t2(O_['OOS']['sr'])}; {t2(F_['OOS_ex2020']['sr'])} vs {t2(O_['OOS_ex2020']['sr'])} without 2020). "
          f"The last 12 months lose {abs(F_['last12m']):.1f}% instead of {abs(O_['last12m']):.1f}%, so about "
          f"{(1 - F_['last12m'] / O_['last12m']):.0%} of the recent drawdown was sector exposure.")
table([["Version (after costs)", "Sharpe 13–19", "t 13–19", "Sharpe 20–26", "Sharpe 20–26 ex-2020", "Sector beta (t)", "Last 12m"]] +
      [[k.replace(" | ", ", "), t2(v["IS"]["sr"]), t2(v["IS"]["t"]), t2(v["OOS"]["sr"]), t2(v["OOS_ex2020"]["sr"]),
        f"{v['exposure']['all']['beta_sector']:.2f} ({t1(v['exposure']['all']['t_sector'])})", f"{v['last12m']:+.1f}%"] for k, v in V_.items()],
      [2.05, 0.75, 0.6, 0.8, 1.05, 0.95, 0.7])
fr = NE["final_reversal"]
P(f"<b>The neutral version is also further from plain reversal.</b> Rerunning the Section 3 test on the relative, beta-neutral "
  f"version against beta-neutral reversal portfolios at 5, 20, 60 and 250 days leaves {t1(fr['IS']['alpha_ann'])}% a year in training "
  f"(t = {t2(fr['IS']['t_alpha'])}, up from {t2(F2['IS']['multi']['t_alpha'])}) and {t1(fr['OOS']['alpha_ann'])}% a year out of sample "
  f"(t = {t2(fr['OOS']['t_alpha'])}). Taking the sector out of the signal sharpens the part that is about one BDC's own deviation.")
P(f"<b>What is still not fixed:</b> training-period strength. The neutral version's training t is {t2(F_['IS']['t'])}, still below 2. "
  "Neutralizing by leverage or non-accrual buckets, as the review also suggested, needs balance-sheet data per BDC that the "
  "pipeline does not have yet; bdc_positions.csv.gz could provide non-accruals.")

# ---------------------------------------------------------------- 7. verdict
P("7. Verdict and how to rewrite the pitch", H2)
P("<b>What the evidence supports:</b>")
bullets([
    "BDC prices overshoot and come back more than comparable financials (Figure 7). Possible reasons to argue, each testable: a "
    "mostly retail holder base, small market caps, a NAV that updates only quarterly so nobody can price the fair value in between, "
    "and issuance above NAV / buybacks below NAV.",
    "Scoring each BDC against its own one-year discount history is a cheap way to trade that overshoot: it turns over less than "
    "plain reversal and survives realistic costs (Figure 5).",
    f"Out of sample (2020–2026) it earns Sharpe {t2(bt['OOS']['sr'])} after costs, {t2(N['bt_ex2020']['sr'])} without 2020.",
    "Using the relative z-score with beta-neutral legs removes the hidden sector bet (beta about 0) at no cost in Sharpe, and "
    f"makes the part beyond reversal stronger (training t = {t2(NE['final_reversal']['IS']['t_alpha'])}). Use it as the main version (Section 6).",
])
P("<b>What it does not support:</b>")
bullets([
    "\"The discount reverts to its normal level because of the NAV.\" Once past returns are removed, the NAV part alone is weak "
    "(Figures 3, 4, 6).",
    f"Strong in-sample evidence. Training Sharpe after costs is {t2(opt['sr_is'])}, t = {t2(opt['t_is'])}, below the t ≥ 2 bar. "
    "Most of the case rests on out-of-sample years that include the 2020 dislocation.",
])
P("<b>Suggested new thesis line:</b> <i>\"BDC prices overshoot more than comparable listed financials because holders are retail and fair "
  "value (NAV) is published only quarterly. Measuring how far each BDC's discount has moved from the sector's, compared with "
  "its own usual gap, and holding beta-neutral legs, captures that overshoot with no sector bet and low enough turnover to "
  "survive costs.\"</i>")
P("<b>Still to do before presenting:</b>")
bullets([
    "Fix every item in Section 1, especially the ×2 heatmap.",
    "Add Figures 2 (panel c), 5 and 7 to the write-up; they answer the \"isn't this just reversal?\" question before a judge asks it.",
    "Add delisted and merged BDCs (2012–2019) and rerun the main tests.",
    "Show returns by year, so judges can see how much depends on 2020–2023.",
])
P("Code: build_reversal_check.py, build_residual_test.py, build_residual_conditional.py, collect_benchmark_prices.py, "
  "build_followup.py, collect_credit_spread.py, build_neutral.py, build_beta_explainer.py, build_portfolio_beta.py, build_findings_pdf.py (BDC Pitch folder). All EDA uses 2012–2019 only; 2020+ appears only in backtests.", CAPF)

doc = SimpleDocTemplate(str(ROOT / "BDC_Findings.pdf"), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.7 * inch, title="BDC Findings")
doc.build(story)
print("wrote", ROOT / "BDC_Findings.pdf")
