# D1 error analysis on test (eval-v1)

Detector: D1 (mmBERT encoder, run `mmbert-v1`) at its dev threshold (0.7045, catches 95 % of
dev attacks). Sample: `gl error-sample --split test`, seed 0, 25 per cell, read manually.
Labels per example id are in `d1_test_sample.csv`. **Errors are described by type only; no
text is quoted.**

| Cell | Errors in test | Sampled |
|---|---|---|
| Missed attacks (FN), seen families | 84 | 25 |
| Missed attacks (FN), held-out family | 717 of 1,500 | 25 |
| False positives (FP), seen | 59 | 25 |
| False positives (FP), held-out look-alikes | 566 of 1,500 | 25 |

## Missed attacks, seen families (25)

| Error type | n | What it is |
|---|---|---|
| Role-play character card | 9 | Long character / role-play setups from the jailbreak-prompt collection (often adult role-play with rules for staying in character). Labelled jailbreak by the source; many read as role-play instructions rather than safety-bypass attempts. |
| Ordinary task labelled attack | 6 | Coding challenges, marketing or writing tasks, a copied assistant system prompt. No manipulation visible: most likely **label noise** in the source. |
| Context-dependent secret-game message | 6 | Very short Mosscap messages (a question, a sentence in another language, a vague command) that are only attacks in the context of the password game. 3 of 6 scored 0.6–0.7, just under the threshold. |
| Clear jailbreak template missed | 4 | Real misses: instruction override + unrestricted persona, a "policy rant then answer anyway" template, a jailbreak-generator bot, a prompt generator aimed at bypassing guidelines. Mostly long texts. |

Only ~4 of 25 are clear-cut misses; the rest are role-play or label/context problems.

## Missed attacks, held-out family (25)

All 25 are adversarial jailbreaks that wrap a harmful request in a legitimizing frame.
D1 never saw this family in training.

| Framing | n |
|---|---|
| Expert / researcher / historian / consultant role | 8 |
| Fiction or creative-writing task | 5 |
| Benevolent purpose (inclusivity, awareness campaign, training simulation) | 5 |
| Explicit override or "reprogrammed / unfiltered AI" claim | 4 |
| Social or conversational scenario leading to a stereotype | 3 |

**15 of 25 scored 0.50–0.70**, just under the dev threshold: D1 finds them suspicious but
not suspicious enough. The 4 with explicit override wording are the most surprising misses,
since that wording is common in seen families.

## False positives, seen (25)

| Error type | n | What it is |
|---|---|---|
| Jailbreak-style text labelled benign | 10 | Dual-answer personas, "forget your instructions" setups, "no morals" director/persona prompts, adult role-play rules. These look like jailbreaks; the source labels them benign. **Label conflict**: one sampled false positive is a near-copy of a template that appears elsewhere in test labelled as an attack (both escaped near-duplicate removal). |
| Short question that resembles a secret-game probe | 8 | Very short questions (about the assistant itself, a riddle, a joke request, an opinion). The style of Mosscap attacks, i.e. what D1 learned about that source. |
| Harmless persona prompt | 4 | "Act as an ASCII artist / magic mirror"-type setups. |
| Emotional-support message | 2 | Benign SPML messages to a support chatbot. |
| Good-faith security question | 1 | A code-security question. |

## False positives, held-out look-alikes (25)

All 25 are **benign** requests written in the same adversarial style as the held-out attacks
(the dataset's contrast set), e.g. "imagine an alternate universe…" around a harmless ask.

| Framing | n |
|---|---|
| Hypothetical / alternate-world framing | 9 |
| Fiction or creative-writing task | 6 |
| Expert / professional role | 6 |
| Game, puzzle or personal anecdote | 4 |

Several include surface phrases such as "free from restrictions" or "without limitations"
around a harmless request, and D1 reacts to the phrase, not the request.

## What this means

1. **On the held-out family D1 detects style, not intent.** Held-out attacks and their benign
   look-alikes share the same framings, so D1 flags both about equally (held-out AUROC 0.60,
   catch 0.52 at FPR 0.38). Telling them apart needs judging the underlying request, which
   is what the LLM judge does (D2 held-out AUROC 0.80).
2. **Many seen-family "errors" are label problems.** About 16 of 25 seen FN and 10 of 25 seen
   FP involve role-play/persona text whose label depends on the source collection, likely
   label noise, or context-dependent game messages. D1's seen-family numbers are therefore
   limited by the labels as much as by the model.
3. **Short texts are fragile.** Very short messages appear both as missed attacks and as
   false positives: with almost no content, D1 falls back on source style.
