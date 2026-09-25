# CFB Rankings -- Model Specification

**Status:** draft for review. Argue with this before I write the engine.

## The thesis

Every public ranking fails in one of three ways:

- **AP / Coaches** -- opinion, recency bias, preseason anchoring. A team
  ranked #3 in August starts with three weeks of credit it did not earn.
- **CFP committee** -- opinion by conference room, with the added problem
  that the criteria change depending on who is in the room.
- **Most computer polls** -- genuinely quantitative, but the *weights* are
  hand-chosen. "Offense is 40% of the rating" is a subjective claim wearing
  a lab coat.

This model's rule: **no hand-tuned weights anywhere.** Every coefficient is
fit from data by minimizing prediction error. If you disagree with a
ranking, you disagree with a measurable component, and we can go look at it.

## The one rule that matters most

**Predictive and resume are different questions. Never average them.**

| | Predictive | Resume |
|---|---|---|
| Question | Who is best right now? | Who has earned the most? |
| Uses margin | Yes | No -- wins are wins |
| Uses opponent | As context | As the primary input |
| Rewards | Dominance | Beating good teams |
| Blind to | Whether you won | Whether you were lucky |

A one-loss team can be *better* than an undefeated team while being *less
deserving*. Both statements are true at once. Human polls mash these into
one number by feel, and that is the root of most poll incoherence.

We compute both, display them side by side, and show the gap. The gap is
often the most interesting column on the page.

## Data reality (revised 2026-09-22)

Source: `sportsdataverse/cfbfastR-data` on GitHub -- a public static mirror
of CFBD. No API key.

| Dataset | Coverage | Rows | Note |
|---|---|---|---|
| Schedules + scores | 2001-2026 | -- | includes the live season |
| **Play-level data** | **2014-2026** | **1.69M plays** | **includes the live season** |
| Legacy play-by-play | 2002-2021 | -- | richer columns, frozen, optional |

**Correction to an earlier draft.** The first pass concluded play-by-play
stopped at 2021, and the two-tier architecture existed to work around that.
Howie pushed back -- "we should be able to find that somewhere" -- and he
was right.

The `player_stats/` directory is misleadingly named. Those files are **one
row per play**, carrying full `down` / `distance` / `yards_to_goal` /
`period` / `clock` state, and they run **2014 through the live 2026
season**. The frozen `pbp/` directory was a red herring.

Audit of all 13 seasons (`tools/audit_plays.py`):

- 1,690,546 plays across 13 seasons
- Identical 70-column schema every year -- zero drift
- **0.0000% nulls** on every EPA-critical field
- 96.5-100% coverage of scheduled FBS games
- 2026: 57,099 plays, 463 games, **100% of completed games**
- 123-141 plays per game, consistent with real football

Lesson recorded: the first "no data" conclusion came from checking one
directory and stopping. Check every directory before declaring a
constraint.

### What this changes

**The two-tier architecture is dead.** There is no longer a gap to bridge,
because EPA is computable for the current season directly. Every statistic
in the spec below is available live:

- EPA per play, offense and defense -- **live**
- Success rate -- **live**
- Explosiveness, field position, finishing drives -- **live**
- Havoc, sacks, pass breakups, forced fumbles -- **live**
- Garbage-time filtering -- **live** (needs score + clock, both present)

One genuine limitation remains: these files carry **no pre-snap EPA model
and no win probability**, so we compute expected points ourselves from
down / distance / field position / time. That is standard practice and
arguably better -- we control and can explain the model rather than
inheriting someone else's.

2020 remains a 542-game COVID season and is flagged, not silently averaged.

## Architecture: one integrated model

Everything runs on the same 2014-2026 play-level data. No tiers, no
calibration bridge -- that complexity only existed to work around a data
gap that turned out not to exist.

### Layer 1 -- Expected points model

Fit an expected-points surface from down, distance, yards to goal, and
time remaining, trained on 1.69M historical plays. EPA for any play is the
change in expected points it produced.

Built from our own data rather than inherited, so every value is
explainable and auditable.

### Layer 2 -- Per-team component stats

Aggregate plays into opponent-agnostic components: EPA/play, success rate,
explosiveness, havoc, finishing drives, field position -- offense and
defense separately, garbage time excluded.

### Layer 3 -- Opponent adjustment (ridge regression)

The core engine. Solve all ~265 team ratings simultaneously so strength of
schedule falls out of the math instead of being a hand-set weight.

Applied to **both** scoring margin and each component stat, so we get
opponent-adjusted EPA, not just opponent-adjusted points.

- **Ridge penalty (lambda)** shrinks ratings toward zero, which is what
  stops a 3-0 MAC team from topping the table in September. Fit by
  cross-validation.
- **Home field** is a fitted parameter, not an assumed 3 points.
- **Margin curve** -- concave transform, shape fit by cross-validation.
- **Recency decay** -- fit, not assumed.

### Layer 4 -- Dual output

Predictive and resume ratings, side by side, never averaged. Plus the luck
column and the trend column.

## The statistics, and why

### In the model

**Tier 1 -- backbone**
- **EPA per play** (off/def, opponent-adjusted) -- best single team-strength
  signal that exists. Available for all seasons including live.
- **Success rate** -- EPA's stabler sibling; stabilizes in fewer games.
  Together they separate consistency from explosiveness.
- **Scoring margin, curved and opponent-adjusted** -- a genuinely strong
  signal once properly adjusted.

**Tier 2 -- Connelly's Five Factors** (empirically derived against scoring
margin, not chosen by vibe): efficiency, explosiveness, field position,
finishing drives, turnovers.

**Tier 3 -- trenches and disruption**: havoc rate, sack rate, sacks
allowed, pass breakups, forced fumbles. All present in the play files.

**Context and priors**
- **Returning production** -- preseason prior, decays to irrelevance by
  ~week 6. Without a prior, early-season ridge output is noise.
- **Recruiting talent** -- weaker prior, same decay treatment.
- **Garbage-time filter** -- non-negotiable. Unfiltered, you punish teams
  for calling off the dogs and reward stat-padding in blowouts.

### Deliberately excluded

- **Raw total yards** -- a drive-length artifact. 400 yards on 90 plays is
  bad; on 50 plays it is excellent. Per-play or nothing.
- **Total points** -- pace artifact for the same reason.
- **Raw W-L in the predictive rating** -- it is the *output* we predict, not
  an input. It belongs in the resume rating, where it is the whole point.
- **Recruiting rank as a live input** -- that is a prior, not evidence.
  Once games are played, the games are the evidence.
- **Preseason polls** -- pure anchoring bias. The thing we are replacing.

### The one everybody gets wrong: turnovers

Turnover margin correlates strongly with *past* results and weakly with
*future* ones. Fumble recovery is close to a coin flip. Interception rate
is part skill, mostly noise.

So: **turnover luck is estimated and stripped into a separate column.** We
do not bury it in the rating -- we *show* it. Teams that look elite purely
on takeaways are exactly the teams this model should fade, and that is
where it earns its keep against the polls.

## Validation -- how we prove it works

A ranking that cannot be checked is just a louder opinion.

**Benchmarks, hardest first:**

1. **Closing betting line** -- the sharpest public predictor there is. Beating
   it consistently is unlikely; the goal is to come close and know the gap.
   Measured by mean absolute error against actual margin.
2. **AP / Coaches polls** -- the thing we claim to beat. Straight
   head-to-head on predicting later results.
3. **SP+ / SRS** -- established computer ratings, a sanity check. Wild
   divergence is either our edge or our bug, and I want to see which.

**Method: walk-forward, strictly out of sample.** Train on weeks 1..N,
predict week N+1, never let future data leak backward. Repeat across all
12 seasons. Report accuracy by week so we can see how early the model
becomes trustworthy.

**Metrics:** mean absolute error on margin, win/loss accuracy, Brier score
on win probability, and calibration (when we say 70%, does it happen 70%
of the time?).

## Resolved design decisions

Settled with Howie on 2026-09-22. Locked unless evidence says otherwise.

### 1. Margin: diminishing-returns curve, not a hard cap

"Static anything is probably not great." Agreed -- a hard cap at 28 makes
winning by 28 and by 56 literally identical, which is its own kind of wrong.

Instead: a concave transform of margin, so each additional point of victory
adds less than the one before it. Candidates are `log(1 + margin)` and a
tanh-style squash. **The shape parameter is fit by cross-validation**, not
chosen by me -- consistent with the no-hand-tuned-weights rule.

### 2. FCS teams get individual ratings, not a pool

Howie: "a win against Campbell is not the same as a win against Montana."
Correct, and the data lets us do better than pooling.

Verified in the 2024 mirror file:

| Team | Record |
|---|---|
| Montana State | 15-1 |
| Montana | 9-5 |
| Campbell | 3-9 |

The mirror carries **692 FCS-vs-FCS games across 129 FCS teams**, plus ~121
FBS-vs-FCS crossover games per season. That is ample graph connectivity to
place FCS teams on the same rating scale as FBS teams.

So: **one unified ridge regression over ~265 teams.** No pooling, no fixed
FCS constant. Consequences that fall out for free:

- Beating Montana State is a real win; beating Campbell is nearly worthless.
- **Losing** to Montana State is survivable; losing to Campbell is
  catastrophic. Pooling would have flattened both.
- Margin against FCS opponents still carries information, as Howie wanted.

FCS teams are rated but **excluded from the published rankings** -- they are
scaffolding for accurate FBS strength of schedule, not the product.

### 3. Extra games: count them, then show the counterfactual

Howie's concern: "teams shouldn't get punished for losing conference
championships -- that's a game they played beyond others."

The concern is real but it is **not CCG-specific**, and treating it as such
would be a special case begging to multiply. The general problem is
**unequal opportunity**: some teams play games that most teams never get
the chance to play.

| Extra game | Who plays it |
|---|---|
| Conference championship | ~12 of 134 teams |
| CFP first round / quarters / semis / final | up to 12, some play 4 |
| Bowl games | ~80 teams, and opt-outs distort them |

Notre Dame can never lose a CCG. A 6-6 team can never lose a playoff game.
So one framework covers all of it: **games beyond the common schedule**.

#### The key insight: this is a resume problem, not a predictive problem

The two ratings behave completely differently here, which is exactly why
the spec refuses to average them.

**Predictive rating -- count CCGs fully, no adjustment needed.**

A CCG is the single most informative game on the schedule: two strong teams,
neutral field, full effort. Throwing it away would discard the best data we
have. And Howie's worry does not actually materialize here, because
opponent-adjusted ridge regression *already* handles it correctly -- losing
by 3 to the #1 team **raises** your predictive rating. There is no
punishment to undo.

**Resume rating -- this is where the asymmetry is real.**

Resume counts results, not margin. A CCG loss is a loss, and the team that
failed to qualify never had the chance to take that loss. You get punished
for qualifying. That is a genuine flaw.

#### The fix: expectation-relative resume scoring

Rather than counting raw wins and losses, each game contributes **result
versus expectation given the opponent**:

- Beat a team you were expected to beat: small credit.
- Lose to an elite team by 3: **nearly neutral** -- slightly negative, not
  a disaster.
- Lose to Campbell: severely negative.

Under this scheme most of the CCG problem dissolves on its own, because a
narrow loss to the best team in the country was roughly the expected
outcome. No special case required -- the metric was just built correctly.

#### What the UI shows

The residual asymmetry is smaller than it feels, but it is not zero. So,
following the rule Howie already set for recency -- **if an adjustment
moved a team, say so** -- every team that played beyond the common schedule
gets an explicit column:

    Team        Rank   Without extra games   Delta
    Texas        #4          #3              -1  (lost SEC CCG)
    Georgia      #2          #5              +3  (won SEC CCG)
    Notre Dame   #7          #7               0  (no CCG played)

**Symmetric for wins and losses.** Howie asked whether wins deserve a note
too -- yes, and for the same reason. If we footnote the penalty but hide
the bonus, we have quietly built in a thumb on the scale. Both directions
get the same treatment.

The model never silently drops a game. Dropping real results to make a
number feel fairer is the exact subjectivity this project exists to avoid.
We count everything, and we show our work.

#### One empirical question left open

Are extra games *systematically* less predictive -- bowl opt-outs,
motivation gaps, long layoffs? That is testable, not arguable. The backtest
will measure prediction error on CCG / bowl / playoff games separately from
regular-season games. If they are measurably noisier, they get
down-weighted **by the fitted amount**, not by a number I picked.

### 3a. Detecting extra games (implementation note)

Harder than it sounds, and currently imperfect -- flagging this so it does
not get mistaken for solved.

`notes` in the raw CSV covers FCS/D-II/D-III playoffs and the CFP, but
**not** FBS conference championships. The working heuristic is structural:
a conference game played in a week after that conference finished its
regular slate.

Accuracy on 2022-2024 is good -- correctly finds Michigan, Georgia,
Clemson, Oregon, Boise State and the rest. Known false positives:

- **Army-Navy**, a neutral-site rivalry played after the regular season.
- **The two-team Pac-12 remnant** (2024-2025), where Oregon State and
  Washington State playing each other looks structurally like a title game.
- Independents, which have no conference slate to compare against.

Fix before this ships: require the conference to have >= 4 members that
season, and exclude known standing rivalries. Verification target is the
full 2014-2025 CCG list, which is publicly checkable.

### 4. Recency: measure the trend, and report it explicitly

Howie: "there should be some interest given to if a team is performing
significantly different at different times of the season, but that needs to
be made note of if that is utilized."

Sharpest note in the batch. It splits into two features:

- **Fitted recency decay** in the core rating. Recent games weigh more; the
  decay rate is cross-validated, not assumed.
- **A published trend column.** For each team, fit rating over time and
  report the slope with a significance test. Teams whose form changed
  *significantly* get flagged, rising or falling, with magnitude shown.

The rule Howie set: **if recency changed the ranking, the UI has to say so.**
A team ranked #8 that would be #14 without recency weighting must display
that. No silent adjustments.

### 5. Transfer portal / coaching: preseason prior only, if at all

Howie: "seems hard to find and of little importance outside of preseason
rankings." Right -- once real games exist, the games are the evidence.

Deferred. If we build a preseason prior beyond returning production and
recruiting talent, portal and coaching churn go in *there* and decay to
irrelevance by roughly week 6. Never a live-rating input.

## Still open

- Whether bowls behave differently enough to warrant special handling
  (answer comes from backtesting, not opinion).
- Preseason prior composition for a from-scratch August ranking.

## Standardizing the EPA engine's prior (2026-09-23)

Howie's complaint: New Mexico ranked above Ole Miss and near LSU in week
3, and more broadly, "all your weighting systems with tons of decimals
or a +/-X pts adjustment is silly. It should all be pretty
standardized." Both were right, and connected.

**Root cause.** The margin engine shrinks toward last season via a
measured prior (carryover 0.63, unshrink 1.117 -- see the prior section
above). The EPA engine, wired in later, never got one -- it shrank
toward zero only. Three games in, New Mexico's EPA edge over Central
Michigan and an FCS team (+1.88 each) was close to Notre Dame's edge
over Rice, and with nothing to weigh that against, plain ridge took it
at face value.

**The fix, and why it counts as "standardized" rather than a new knob.**
EPA and margin describe the same underlying team quality in different
units. Rather than fitting a second carryover/unshrink pair from
scratch on a noisier signal, `epa_ridge.translate_margin_prior()`
converts the ALREADY-measured margin prior into EPA units using the
model's own fitted points-per-EPA factor -- a unit change, not a new
opinion. `fit_epa_ratings_primed()` applies it. Both engines now answer
"how much of a team's quality survives the offseason" the same way, with
one measured answer, not two.

**This also exposed a second, older bug.** Two analysis tools
(`tools/blend_epa_margin.py`, `tools/compare_epa_fair.py`) had their own
prior logic that reintroduced the exact FCS/FBS double-counting bug
fixed for the margin engine on 2026-09-22 (see the git log) -- it just
never got applied when the EPA engine was extended. Both tools now call
the same `fit_epa_ratings_primed()` production code path. One prior
mechanism, used everywhere it's needed, is the actual meaning of
"standardized" here -- not fewer decimal places for their own sake.

**Re-measuring what the bug had contaminated.** The EPA ridge penalty
(`RELATIVE_LAMBDA`) and the margin/EPA blend weights had both been
chosen using the buggy prior. Re-run under the fix: `RELATIVE_LAMBDA`
moved from 8.0 to 4.0 (13.163 vs 13.524 MAE, walk-forward on
2024-2025), and the blend weights moved from margin +0.611/EPA +0.237 to
margin +0.679/EPA +0.211 (held out on 2025: stacked 12.182 MAE / 74.5%
accuracy vs margin-only 12.320 MAE / 74.7% -- the ensemble still wins on
error, roughly a wash on win/loss accuracy). The ensemble concept holds
up under honest measurement; it was the prior underneath it that didn't.

**What's still open.** New Mexico moved from #11 to #18 (below Ole Miss
at #15 and LSU at #8), not out of the picture entirely. Three games is
genuinely thin evidence, and the EPA engine still has no blowout-curve
analogous to the margin engine's `curve_margin()` tanh transform --
individual game observations enter uncompressed, so two lopsided wins
can still outweigh a small prior when the prior itself is only mildly
skeptical. Properly cross-validating an EPA-scale saturating curve (same
mechanism, own fitted scale, not a hand-picked cap) is the natural next
step if this keeps showing up as seasons accumulate more thin-sample
weeks.

**Display cleanup, same session.** Rating and strength-of-schedule
columns were showing hundredths of a point on a model with a fitted
residual standard deviation over 10 points -- false precision.
Standardized every points-scale figure (CLI and templates) to one
decimal place.

## Game prediction (required feature)

Howie: "we should also have a game prediction feature -- if X played X what
would be the score outcome based on our data. This should not be like
generated, it should be based in the data."

This is not a side feature. **It is the honesty check on the entire
model.** A rating that cannot produce a falsifiable prediction is just an
opinion with decimals attached.

### How it works

Every prediction is arithmetic on fitted parameters. Nothing is generated,
invented, or hand-waved:

    predicted_margin = rating[team_a] - rating[team_b] + home_field_advantage

The ridge regression already produces `rating` and `home_field_advantage`
by minimizing error against thousands of real games. The prediction is
just that model, evaluated forward instead of backward.

Score split comes from fitted pace and efficiency:

    total_points   = f(pace[a], pace[b], off/def efficiency)
    predicted_a    = (total + margin) / 2
    predicted_b    = (total - margin) / 2

Win probability comes from the **empirical** distribution of prediction
errors, not an assumed normal curve. We know the historical spread of
`actual - predicted`, so a predicted margin of +7 maps to whatever win
rate a +7 prediction actually produced historically.

### Every prediction shows its work

    Oklahoma vs Texas (neutral site)

      Predicted:  Oklahoma 27.4 - 24.1 Texas   (OU by 3.3)
      Win prob:   Oklahoma 58%
      Confidence: +/- 13.2 points (1 sigma, from historical error)

      Why:
        Oklahoma rating      +18.2
        Texas rating         +14.9
        Rating edge           +3.3  -> OU
        Home field            +0.0  (neutral site)
        Predicted margin      +3.3

      Biggest factors:
        OU pass offense vs TX pass defense    +2.1
        TX rush offense vs OU rush defense    -1.4
        Turnover luck regression              +0.8

The confidence interval is mandatory. A prediction without an error bar
invites false precision, and college football is genuinely high-variance --
that +/- 13 is real and should be visible.

### Validation

Same standard as the ratings themselves: walk-forward, out of sample,
never letting future data leak backward. Benchmarked against the closing
betting line, which is the sharpest public predictor available. We report
the gap honestly rather than claiming to beat it.

## Non-goals

- Not predicting the CFP committee's behavior. Modeling their picks means
  modeling their bias.
- Not a betting tool.
- No "eye test" adjustment layer. That is the disease, not the cure.

---

## Resume rating: shipped 2026-09-23

Until this date the published table was the **predictive rating only**, and
the resume side of the spec above existed on paper but not in code. Howie
caught it from the output, which is the right way for a gap like this to be
found:

> "Ole Miss just beat LSU and they are several ranks below them. That would
> make sense for like OkSt and Oregon because OkSt also lost to Tulsa."

That is a precise statement of the problem. Two teams, same shape of upset,
and only one of them should show an inversion:

| Team | Predictive | Resume | |
|---|---|---|---|
| Ole Miss 3-0, beat LSU | #15 | **#2** | inversion removed |
| LSU 2-1, lost to Ole Miss | #8 | #46 | |
| Oklahoma State 2-1, beat Oregon, lost to Tulsa | #99 | #64 | inversion kept, correctly |
| Oregon 2-1, lost to Oklahoma State | #28 | #73 | |

The predictive numbers were never wrong. LSU really did look like a better
team than Ole Miss on a neutral field -- they beat Clemson 51-10 and lost by
8 on the road. Margin carries information a single result does not, and the
predictive rating is *supposed* to say that. The bug was publishing only
that rating and letting it answer a question it was never built for.

### Method: strength of record

Implemented in `src/cfbrank/resume.py`.

For each game, ask how likely a **benchmark team** would have been to win
that same game, at that same site, against that same opponent. Sum those to
get expected wins. Resume score is `actual wins - expected wins`.

Win probabilities come from the already-fitted predictive model and its
empirical error distribution, so the resume rating introduces **no new free
parameters**. The benchmark is the 25th-best FBS team by rating -- read from
the data, not hand-set, so it tracks the league automatically.

Properties that fall out of the definition rather than being patched in:

- **Margin is ignored entirely.** A 1-point win and a 50-point win are
  identical. Asserted in `tests/test_resume.py`.
- **Head-to-head is never special-cased.** Beating a team raises your score
  and lowers theirs in the same stroke, so the inversion resolves through
  arithmetic rather than a tiebreak rule bolted on afterwards.
- **Bad losses still hurt.** Oklahoma State stays below Oregon because the
  Tulsa loss costs -0.61, swamping the +0.25 earned for beating Oregon.

### The rule this restores

**Predictive and resume are different questions. Never average them.** The
UI now sorts by either and always shows both, because the gap between them
is the most interesting column on the page. LSU at #8 predictive and #46
resume is not a contradiction -- it is the model saying "good team, has not
earned it yet." One number could never say that.

### Not yet built

The counterfactual "without extra games" column specced above still does not
exist. It matters in December, not September, but it should not be quietly
forgotten just because the resume rating now ships.

---

## Accuracy limits: measured 2026-09-23

Three questions from Howie, all answered with measurements rather than
argument. Reproduce with `tools/accuracy_by_week.py`,
`tools/accuracy_ceiling.py`, `tools/ranking_convergence.py`, and
`tools/blend_predictive_resume.py`.

### 1. Does the ranking get better as the season goes on?

**Yes -- but not in the way you would detect by watching prediction
accuracy, which is nearly flat.**

Pooled walk-forward accuracy by week (2021-2025, 5,803 games):

| | games | MAE | accuracy |
|---|---|---|---|
| weeks <= 8 | 2,713 | 12.79 | 71.9% |
| weeks > 8 | 3,090 | 12.60 | 72.7% |

Eight tenths of a point. That looks like the model stops learning in
September, and it is the wrong conclusion.

Comparing each week's ranking to the **final** ranking tells the real
story (Kendall tau, and how many of the final top 25 are already present):

| week | tau | top 25 matched |
|---|---|---|
| 4 | 0.698 | 17.2 / 25 |
| 8 | 0.809 | 19.8 / 25 |
| 11 | 0.888 | 22.8 / 25 |
| 13 | 0.960 | 23.8 / 25 |

The ranking converges steadily and hard. **Roughly a third of the week-4
top 25 does not belong there.**

The two facts are reconcilable: game outcomes are dominated by variance,
so prediction accuracy is pinned near a ceiling and cannot move much no
matter how good the ratings get. Flat accuracy is a ceiling effect, not
evidence of a static model. Ranking quality is the more sensitive
instrument, and it says the model is still learning a lot.

### 2. Should there be one authoritative blended top 25?

**No.** The spec's "never average them" rule was reasoned from first
principles, so it was worth testing rather than trusting. It survived.

Blending z-scored predictive and resume ratings at every weight from 0 to
1, ranking teams by the blend, then scoring how often the higher-ranked
team wins a later game (1,372 post-week-9 games, 2021-2025):

| weight | hit rate |
|---|---|
| 0.00 (pure resume) | 67.4% |
| 0.50 | 70.0% |
| 0.70 (best) | 70.6% |
| 1.00 (pure predictive) | 70.5% |

The best blend beats pure predictive by **0.15 percentage points -- two
games out of 1,372, or 0.12 standard errors** (SE is 1.23 points; the 95%
CI half-width is 2.41). That is indistinguishable from noise.

So blending buys nothing measurable while costing the thing that makes
the model honest: the ability to say "LSU is the better team but has not
earned it yet." A composite silently picks one answer to a question that
has two, which is exactly the incoherence in human polls.

**Decision: keep two columns. Do not ship a blended number.** If a single
list is ever needed for presentation, sort by resume -- it is the
poll-shaped question -- and keep the predictive column visible beside it.

### 3. How much better can the predictive rating get?

**Not much, and the ceiling is low.**

Refitting on the complete season and scoring games the model has already
seen gives an oracle: not achievable (it uses the future), but it bounds
what perfect knowledge of team strength would buy.

| | MAE | accuracy |
|---|---|---|
| live, out of sample | 12.69 | 72.4% |
| oracle, full hindsight | 10.64 | 80.5% |
| headroom | 2.05 | +8.2% |

A model that already knows how the season ended **still misses by 10.6
points a game and still loses one game in five.** That is irreducible
noise -- turnovers, injuries, weather, a fourth-down spot.

Total available headroom is 2.05 points of MAE, 16% of current error.
Realistically a good chunk of that is unreachable, because the oracle
also benefits from in-sample overfitting on ~1,200 games.

**Implication: further accuracy work has poor expected return.** Effort is
better spent on questions the model cannot currently answer at all --
the counterfactual "without extra games" column, per-team explanations,
uncertainty display -- than on grinding out tenths of a point.

**Correction, same day:** that implication was wrong. See the next
section -- the closing line proves ~0.4 pts/game more is reachable, and
the oracle bound says nothing about how much of the gap is actually
capturable.

---

## Benchmarked against the closing line: 2026-09-23

Howie: "we are not beating ESPN FPI or SP+." SP+/FPI game-level picks
are not reachable from this network (CFBD API and the Prediction Tracker
are proxy-blocked), so the benchmark is the closing line -- which SP+
and FPI themselves are usually measured against. Lines come from
`sportsdataverse/cfbfastR-data` `betting/csv/cfb_line_odds.csv.gz`,
keyed by ESPN game id, so both sides are scored on identical games.

Reproduce: `tools/benchmark_vs_market.py` (production model) and
`tools/improvement_experiment.py` (variants, leave-one-season-out stack
weights so no variant is scored on weights that saw its own games).

### Before (5,070 games, 2021-2025, walk-forward)

Ours 12.41 MAE vs line 11.97: **0.44 pts/game behind** (95% CI +/-0.11),
behind in every season. 0.62 behind in weeks 4-8, 0.29 after. ATS 50.4%.
Stacking ours with the line added nothing out of sample.

### What each upgrade was worth (vs baseline, pts/game, paired SE ~0.03)

| change | effect | shipped |
|---|---|---|
| Roster-aware preseason prior | **-0.045** | yes |
| + success rate as efficiency component | **-0.085** total | yes |
| + fumble-recovery luck removed from margin | -0.002 more | no (noise) |
| + garbage-time margin | +0.006 (worse) | no |
| EPA alongside success rate | -0.002 more | no (redundant) |

**Roster prior** (`cfbrank/returning.py`, `cfbrank/prior_model.py`):
replaces the flat 0.63 carryover with a regression on the last three
seasons' ratings plus returning offensive/QB/defensive production and
incoming-transfer production, from play data joined to next-season
rosters (player ids join at 100%). Weights for season S are fit only on
offseasons before S. Preseason RMSE vs final rating improved in 6 of 7
seasons (2020, the COVID year, is the exception). Biggest fitted effects:
returning offense and incoming portal production.

**Success rate beat EPA** as the per-play component once the better
prior was in. Stack weights: margin 0.731, success rate 0.187. EPA's
weight collapsed to 0.05 when both were offered.

**Garbage-time margin and fumble luck did not help.** Both were in the
spec as "non-negotiable"; measured, they are not. The margin engine's
tanh curve already discounts blowouts, so freezing the margin when the
game is decided removes information rather than noise. Fumble luck is
real (5.4 pts per extra recovery in-game) but already averages out over
a season's worth of games. Both remain available in
`cfbrank/game_features.py` for the luck column; neither feeds the
rating. EPA/success aggregation still drops garbage-time snaps -- tested
both ways, it is a wash (-0.002).

### After

Ours 12.32 vs line 11.97: **0.35 pts/game behind** (was 0.44). Weeks 4-8
gap 0.62 -> 0.53; weeks 9+ 0.29 -> 0.20. Straight-up 73.3% vs the
line's 73.8% (was 72.8%). Improvement holds in all five seasons.

### Still open

- **Offense/defense split** -- built and measured 2026-09-24, see below.
- **Recruiting talent** -- DONE 2026-09-25, see "CFBD roster priors" at
  the end. CFBD turned out to be reachable via proxy.wal-mart.com.
- Early season is still where the gap lives (0.53 vs 0.20).

---

## Offense / defense ratings: measured 2026-09-24

Howie: "add Offense and Defense ratings to see if that helps predictive
power." Built, tested on the same 5,070 games against the same closing
line. **Short answer: great for explanation and for predicting totals;
no help for spreads.** Reproduce: `tools/offdef_experiment.py`.

### Model (`cfbrank/offdef.py`)

Each game becomes two observations, one per offense:

    points(side) = mu + off[offense] - def[defense] + venue*HFA/2 + FCS term

Ridge-solved for ~530 parameters at once, same machinery as the margin
engine. Both columns read bigger-is-better. The same engine runs on
per-side success rate (`is_points=False`, with a fitted conversion to
points). The earlier note that "margin-only data cannot separate them"
was wrong: points FOR and AGAINST can. A margin is what throws that away.

**Separate priors** (`cfbrank/offdef_prior.py`): last three seasons of
each side plus returning offense/QB/defense and portal production, one
regression per side, fit only on earlier offseasons. It learns sensible
things: returning QB production mainly feeds offense, returning defense
mainly feeds defense.

### Why a split could help at all

In a linear model the predicted margin depends only on off + def, so a
split adds nothing unless the sides are treated differently. All three
routes tested (pts/game vs production, negative = better):

| route | test | result |
|---|---|---|
| separate priors per side | split prior vs the net roster prior halved onto each side (control) | control was **better** by 0.002-0.010 |
| points for/against as separate evidence | off/def points model stacked on production | -0.019 at best, CI +/-0.022 |
| off/def success rate | stacked on production | +0.000 (fully redundant) |

Best spread variant: -0.019, inside its own 95% CI, and the control
matched it. So the little it adds comes from being a third margin-style
model, not from offense/defense information. **Not shipped into the
rating.** That matches the earlier ensemble finding: the success-rate
component already uses both sides' per-play numbers.

### Totals: the thing a net rating cannot do

| predictor of combined points | MAE |
|---|---|
| league-average total (what the app used) | 13.71 |
| off/def points model, lambda 5 | **12.94** |
| closing over/under | 12.63 |

Lambda sweep 0.5 / 1 / 2 / 5 / 12 gave 13.80 / 13.50 / 13.19 / 12.94 /
12.95; 5 shipped (`offdef_prior.TOTALS_LAMBDA`). O/U side-picking is
50.9%, so no edge over the market, but the gap to it shrank from 1.08
to 0.31.

### Shipped

- Off / Def columns (points above/below an average FBS team vs an
  average opponent, plus rank), sortable, on the rankings page.
- Predicted score uses the matchup-specific total; the MARGIN still
  comes from the production ensemble, unchanged. Asserted in
  `tests/test_offdef.py`.
- The predictive rank is untouched.

---

## Postseason: loaded, trained on, and graded (2026-09-24)

**Bug found:** postseason games were missing for 2014-2022 (the mirror's
per-season files stop at the regular season before 2023). Where they did
load (2023-25), upstream labels them all week 1, so walk-forward code
sorted the national title game before the opener.

**Fix** (`cfbrank/games.py`):

- Supplements, tried in order when a season has no postseason:
  CFBD `data/raw/<yr>/games.json.gz` (via `fetch_cfbd.py`), then the
  mirror's `cfb_games_info.csv` (2007-2020 bowls, divisions back-filled).
- Weeks renumbered by date: games inside a regular week keep it (FCS
  playoffs), later ones get one week per 7 days after the last regular
  week. Rounds stay in order; nothing trains on a later game.
- Every consumer gets this through `load_season`: ratings, priors,
  resume, backtests, benchmark.

**Still missing:** 2021-2022 bowls. Not in any mirror file; ESPN and
CFBD are unreachable here. Running `fetch_cfbd.py --years 2021 2022`
off-network fills them with no code change.

**Benchmark** (now 5,215 games, 146 postseason with a line):

| slice | n | ours | line | gap | winner |
|---|---|---|---|---|---|
| weeks 4-8 | 2,336 | 12.30 | 11.78 | +0.52 | 72.7% |
| week 9+ | 2,733 | 12.33 | 12.13 | +0.21 | 73.4% |
| postseason | 146 | 13.07 | 11.98 | +1.08 +/-0.80 | 61.0% (line 61.6%) |

Postseason bias check (`tools/check_postseason_bias.py`): slope 1.06,
so margins are not too extreme; shrinking them loses 0.118 held out.
The line wins on bowls because it knows opt-outs, coaching changes and
motivation, which game results cannot see. Not fixed by tuning; 146 games
is too few to fit a bowl-specific correction honestly.

---

## CFBD roster priors: measured 2026-09-25

CFBD is reachable after all: via `proxy.wal-mart.com:8080` (the PAC
file's route), not `sysproxy`. `set HTTPS_PROXY=http://proxy.wal-mart.com:8080`
then `tools/fetch_cfbd.py --only priors`. Free tier 1,000 calls/month.

**Candidates** (`cfbrank/cfbd_priors.py`), screened by
`tools/screen_priors.py`: held-out RMSE of each team's final rating,
leave-one-season-out, 1,297 team-seasons 2016-2025.

| added to current prior | RMSE 2021+ change |
|---|---|
| 247 team talent composite | -0.097 |
| recruiting classes (4-yr mean) | -0.094 |
| coaching change (+ x r1) | -0.031 |
| **talent + coaching change** | **-0.120** (best) |
| talent + recruiting + coach | -0.105 |
| portal net 247 rating | -0.002 (noise) |
| CFBD returning PPA % | -0.001 (noise) |

Recruiting and talent overlap almost completely; talent alone is simpler
and at least as good. Portal ratings and CFBD returning PPA add nothing
over `returning.py`, which already measures actual production.

**Coaching-change leakage trap.** "This season's main coach differs"
flags coaches fired mid-season, i.e. it knows the season went badly.
`new_coach` flags only a coach from last season who coached ZERO games
this season -- knowable before week 1. Tested in
`tests/test_cfbd_priors.py`.

**Shipped** in `prior_model.FEATURES`: `talent` (per 100 pts, centred),
`new_coach`, `r1_new_coach`. 2026 weights: talent +0.51 per 100; a new
coach -1.34 pts and 13% less carryover of last season's edge.

**Walk-forward vs closing line**, same 5,215 games
(`tools/compare_benchmarks.py`):

| slice | before | after | change | +/- |
|---|---|---|---|---|
| all | +0.370 | +0.342 | **-0.029** | 0.019 |
| weeks 4-8 | +0.517 | +0.471 | **-0.046** | 0.035 |
| weeks 9+ | +0.207 | +0.194 | -0.013 | 0.020 |
| postseason | +1.084 | +1.029 | -0.055 | 0.096 |

Four of five seasons improved (2022 flat). With 2021-22 bowls now loaded
from CFBD the benchmark is 5,299 games: +0.35 pooled, postseason +0.89
over 230 games, CFP -0.11 over 31 (level with the line, wide CI).

**Still zero independent information vs the market**
(`tools/check_market_blend.py`): a leave-one-season-out blend of ours +
line beats the line alone by +0.004 +/- 0.014 -- nothing.

