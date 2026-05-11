# Resume Matching Engine
### TF-IDF + Cosine Similarity · Redrob AI Campus Hackathon · Individual

![Python](https://img.shields.io/badge/Python-3.x-blue) ![No External Libs](https://img.shields.io/badge/dependencies-none-green) ![Standard Library](https://img.shields.io/badge/stdlib-math%20%7C%20re-purple)

---

## Overview

A resume matching engine built under strict hackathon constraints — no numpy, no pandas, no sklearn. Implements a full IR pipeline from raw noisy resume strings to ranked candidate output using only `math` and `re` from the Python standard library.

---

## Pipeline

| Stage | Description |
|---|---|
| Skill Normalization | Comma-split → lowercase → regex whitespace collapse → alias map lookup → discard unknowns |
| Deduplication | Canonical skills deduplicated per resume using a seen-set. Order preserved. |
| Vocabulary Construction | Built from resume skills only. Alphabetically sorted. 48 terms. |
| TF-IDF Vectors | Exact spec formulas. TF = 1/N post-dedup. IDF = ln(10/df). No smoothing. |
| Binary JD Vectors | JD skills normalized over the same vocabulary. Out-of-vocab terms excluded per spec. |
| Ranking | Cosine similarity. Top 3 per JD. Ties broken alphabetically by candidate name. |

---

## Formulas

```
TF(skill, resume)  =  1 / N           where N = total unique skills after dedup
IDF(skill)         =  ln( 10 / df )   where df = number of resumes containing the skill
TF-IDF             =  TF × IDF
Cosine(A, B)       =  (A · B) / (|A| × |B|)
```

---

## Output

```
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

---

## Constraints

- No external libraries (numpy, pandas, scikit-learn)
- No ML embeddings or transformers
- Standard library only: `math`, `re`
- SKILL_ALIASES used exactly as provided — no modifications

---

## Key Design Decisions

**Vocabulary from resumes only** — JD skills absent from all resumes (e.g. `pytorch`, `redis`) contribute zero to dot products by construction. No vocab inflation.

**Exact TF formula** — After deduplication each skill appears exactly once per resume, so count = 1 and TF = 1/N. This is not an approximation — it is mathematically exact.

**Natural logarithm, no smoothing** — `math.log()` in Python is ln. IDF = ln(10/df) strictly per spec. Adding +1 smoothing (sklearn default) would change every score.

**Whitespace normalization** — `re.sub(r'\s+', ' ', token.strip()).lower()` collapses multi-space tokens before alias lookup, handling dirty real-world input correctly.

**Tie-breaking** — `scores.sort(key=lambda x: (-x[1], x[0]))` — descending score, then lexicographic name. Deterministic.

---

## File Structure

```
TanishqSethi_Hackathon.ipynb   — full implementation notebook
README.md                      — this file
```

---

## Author

Tanishq Sethi | MAIT Delhi | 14214802723
```
