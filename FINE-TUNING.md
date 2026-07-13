### Fine-tuning

## Non-Hardcoded

I fine-tuned Qwen 3-4B models on data/sft and data_non_hardcoded_enhanced.
Both of these train the model to emit a reliable crossword generation and fill algorithm, without needing to come up with a word list. To enforce this, I trained the models to write a generate_crossword() function that takes in a word_list parameter.

# T1
T1 was run on data/sft, which includes data up until generation 3 of the LLM Evolutionary Search.

T1 was evaluated on the **bare** `eval.jsonl` deployment prompt (the EVAL 2-style prompt, so 4 sizes), re-scored on the purified palette (`WORD_LIST_FULLY_PURIFIED`), n=25/size:

| size    | valid% | fullyOK% | within% | dictOK% | fill (n)      |
| ------- | ------ | -------- | ------- | ------- | ------------- |
| 7×7     | 24     | 24       | 24      | 24      | 0.84 (7)      |
| 9×9     | 8      | 8        | 8       | 8       | 0.83 (2)      |
| 11×11   | 0      | 0        | 0       | 0       | — (0)         |
| 15×15   | 0      | 0        | 0       | 0       | 0.83 (1)      |
| **all** | **8**  | **8**    | **8**   | **8**   | **0.84 (10)** |

`fill (n)` = mean white-square fraction of the grids that returned (n = how many of the 25 produced a grid); ~0.84 (16% black) is a real crossword. The generator hash-seeds its RNG per process, so 7×7 wobbles ~24–28% valid across re-scores. **Not directly comparable to T2**, which uses the EVAL 3 size-specific prompt (T1 = bare prompt). Result files: `.../eval/rescore_purified_exact.json` (valid/within); fill from a purified re-score.

### T1 example crosswords
(`#` = black square; scored on the purified palette)

**Valid** — real, interlocking vocabulary at proper ~16%-black density:

`prog_027_s9` (9×9) — SUBSTANCE · ALLOTMENT · BACH · AQUA · EDGE
```text
B A C H # S H O O
A Q U A # H A N D
S U B S T A N C E
H A S # E D G E #
# # # A N Y # # #
# C E D E # P A S
A L L O T M E N T
P U M P # A N N A
T E S T # S T A R
```

`prog_000_s7` (7×7) — ELEMENT · ADAPTED · WRY · RUE
```text
A P T # W R Y
W O E # R U E
E L E M E N T
# # N O S # #
A D A P T E D
L O G # L E I
A G E # E L M
```

`prog_042_s9` (9×9) — EDITORIAL · FERN · ACID · SPIN
```text
# D O E # # S P A
F E R N # S P I N
A C I D # P E N T
R I G # L A C E #
E D I T O R I A L
# U N I T # A P E
B O A R # A L P S
R U L E # I L L S
A S S # # D Y E #
```

**Invalid** — T1 usually emits **no grid at all** (92/100 empty or crash); when it does fill a large grid, the crossings are gibberish:

`prog_082_s15` (15×15) — 52 non-word runs (PSOE, FND, DAMOCRNHN …):
```text
# T P T S # # J A R M C E P #
C R E A D # D A M O C R N H N
H A R K D # D B S R U P V A O
O N S R S B G # C R E M I R W
P S O E # U I S H O N E R M #
# F N D # O I K E D # # O A P
L O I K # R H O # # # F N C T
N R F S # # H E M # # P M O T
B M I E # # # G O B # U E L S
D A C # # S A P P I # P N O #
# T A R A N G E R M # D T G L
T I T U A S E # E E I Z A I A
E O I T A N T L Y # D W L C M
E N O U N I A E Y # A V L A L
# S N T T I N A # # G E Y L #
```

`prog_001_s7`, `prog_002_s7` (7×7) — **syntax error, no grid produced** (truncated/malformed code).

# T2
T2 was run on data/sft_non_hardcoded_enhanced, which includes data up until generation 5 of the LLM Evolutionary Search. T2 was trained with a prompt that was identical to Claude's EVAL 3 for optimal comparison. T2 was the best model (see scores in EVAL.md).

### T2 example crosswords
(`#` = black square; scored on the purified palette)

**Valid** — dense, all-real-word crosswords:

`prog_037_s9` (9×9) — MADDENING · SHARPNESS · TRIBESMEN · AWAIT
```text
# A R K # # E V E
A W A I T # P A R
M A D D E N I N G
P S I # M A D # #
S H A R P N E S S
# # T O T # R H O
T R I B E S M E N
W O O # D E A L S
O W N # # E L F #
```

`prog_000_s7` (7×7) — DELIGHT · LAYERED · NAP
```text
L I D # A D O
I R E # N A P
D E L I G H T
# # A R E # #
L A Y E R E D
A L E # E M U
W A D # D U G
```

`prog_043_s9` (9×9) — MIGRATE · WAVE · AVOW · EXIT
```text
# W A N # S N A P
H A L O # W A V E
E X I T # A V O W
P E G # E P I C #
# S N A G # G A P
# # M I G R A T E
L I E D # I B I S
A C N E # S L O T
V E T S # K E N #
```

**Invalid** — dense grids with **gibberish crossings** (the same "mode A" failure the base model shows in EVAL 2 — force a dense grid, invent words to make it cross):

`prog_052_s11` (11×11) — 21 non-word runs (SDUYBNOFELG, FBLISALBONS …):
```text
O U T S I D G # H D U
F O R L T D A # Y E S
F B L I S A L B O N S
# B U D # # A E P # #
F I R # # A X T E N S
T I E # F L Y # W I K
R E V K U P # # O B I
# # W E M # # A S B #
S D U Y B N O F E L G
P U C # L E H A T M A
A G O # E O M R E S S
```

`prog_035_s9` (9×9) — 15 non-word runs (note the mangled "PORSFOLIO" ≈ PORTFOLIO):
```text
C O L # # # A S H
G O U T P # P H I
T H D H O T E E F
# # P E N T I L F
# H O F T E L F #
P O N T I F F # #
P O R S F O L I O
G O U # F S I D O
S I C # # # A S H
```

`prog_003_s7` (7×7) — 11 non-word runs (TIVNLW, AOSS, SRMPLEA …):
```text
# # A O S S #
U O M I M P #
S R M P L E A
E N A S I O N
P A R A B L E
# T I V N L W
# E A N G # #
```

## Hardcoded

I also fine-tuned a Qwen 3-4B on the data/sft_hardcoded_words dataset, since I wanted to make the model more robust by requiring an outside word list. To build this dataset, I ran every program in data/sft 10 times and hardcoded all the possible words that the LLM used in its generated crossword.

After fine-tuning this model, I applied Reinforcement Learning with Verifiable Rewards (RLVR), verifying the model primarily based on its validity score. Unfortunately, the programs were so large (due to hardcoded words) that I was unable to complete an evaluation for this fully fine-tuned model due to deadline constraints.