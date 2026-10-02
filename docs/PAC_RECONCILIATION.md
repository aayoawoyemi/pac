# PAC reconciliation: your documents vs this session's runs (2026-09-29)

Nothing in your original documents was edited. This file lines up what they define with what the runs measured.

## 1. What your documents define (quoted)

- **Thesis** (HANDOFF_TIPOFF_WNBA_UWRTS.md, card 6): *"basketball is a zero-sum sport. Each team only gets a set
  number of possessions in a game, and every single possession can only end with one player shooting... rTS only
  answers one half of this: how efficiently that scarce resource was disseminated, without talking about the volume."*
- **The Kobe / Brunson clause** (THREAD_UWRTS_2026-09-01.md, 16/): *"raw shooting volume still provides value that
  would otherwise be lost if replaced by their teammates."*
- **Possession displacement** (UWRTS_PAPER.md, K.1): *"every possession has to end... a player who absorbs 30% of
  them at league-average efficiency prevents 30% of them from going to a replacement-level guard."*
- **Creation vs displacement** (UWRTS_PAPER.md, K): adding playmaking *"absorbs 31%"* of the usage effect; about 69%
  survives as possession displacement. Gravity/spacing *"not separable here."*
- **A real replacement baseline** was the open item (UWRTS_PAPER.md, D.5).
- **PAC** (COEF_PAC_MASTER.md, I-2): `PAC = PTS - TSA x (2 lgTS - s L)`, `L = TSA / POSS` (season share), s measured.
- **Scope** (COEF_PAC_MASTER.md, II-9): creation, gravity and foul drawing are value *"PAC deliberately does not price."*

## 2. Unchanged by the runs

- The PAC formula, `L = TSA / POSS`, TSA, the break-even algebra, per-game / season / per-100 forms.
- The meaning: every possession has a price, which is what the team would have gotten from it in someone else's hands.
- The published through-origin value: the game-level design re-estimates it at 0.257 [0.223, 0.291] (published 0.245).

## 3. What the runs measured (refinements, not changes of concept)

| question | result | file |
|---|---|---|
| is the budget zero-sum? | when a player sits, teammates re-take 99% of his shots | (inline check, see PAC_VALIDATED_NUMBERS.md) |
| total cost of an absence by share | gap ~0 below ~13% share; 0.092 at 30% share | `_pac_gamelevel_results.md` |
| is the method sound? | recovers planted prices; invents no effect or threshold | `_pac_how_sure.md` |
| better than TS Add? | 30%+ share absence: team loses 3.5 pts/g; TS Add says 0.5, PAC 2.8 | `_pac_vs_tsadd.md` |
| team-specific prices? | not detectable (split-half r = 0.03) | `_pac_team_price.md` |

## 4. Scarcity vs playmaking: the part that matters for your spec

The absence cost contains the absorbed shots (your possession scarcity) **and** the absent player's playmaking.
Your spec says PAC does not price creation, so the relevant price is the scarcity part. Measured by the absent
player's playmaking at the same share (`_pac_scarcity_forms.md`):

| absent player | gap at 28%+ share (bins) | gap at 30% share (free line) |
|---|---|---|
| pure scorers (lowest third AST%): scarcity with little creation | 0.065 [0.020, 0.110] | 0.072 [0.043, 0.101] |
| playmakers (highest third AST%) | 0.114 | 0.106 [0.088, 0.123] |
| **published PAC, 0.25 L** | **0.076** | **0.075** |

- At high usage, **scarcity is about 60-70% of a playmaker's absence cost and playmaking about 30-40%**. Your UW-rTS
  paper found 69% / 31% with a completely different method (RAPM regressions).
- At high usage, **your published price (0.25 L) is close to the scarcity-only price.**
- Where the published price is off: in the 15-25% share range, pure scorers show no scarcity cost
  (bins: -0.020 at 16-20%, +0.022 at 20-24%), while 0.25 L charges 0.045-0.054.
- Below ~16% share, pure scorers' bins go negative (-0.091). That is like-for-like replacement of efficient
  low-usage finishers (the mechanical coefficient is 0.61 there), not negative scarcity.

**Retraction.** An earlier message this session gave a "new value" of slope 0.258 / 2.2 TS points at 30% share.
That came from forcing one shared threshold on every group. With the shape left free (bins, free intercept), the
scarcity-only gap at 30% share is about 0.07 (3.5 TS points).

## 5. Decisions that are yours

1. **Keep the published price (0.25 L).** At high usage it matches the scarcity-only measurement. It over-credits
   15-25% share players, which is a stated limitation.
2. **Scarcity-only schedule.** ~0 below ~20% share, ~0.07 at 30%, from pure scorers' absences. Truest to your spec,
   noisier (fewer high-share pure scorers).
3. **Total absence schedule** (0.535 above 13%). Largest validation base, but it includes playmaking, which your spec
   excludes.

Whatever you choose, the sentence in your own words holds: every possession has a price, and for players who
carry the offense it is below league average, because the teammates who would have to absorb those possessions
get worse at it.
