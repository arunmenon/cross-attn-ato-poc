# Day-2 data-overlap diagnostic

Read-only diagnostic of `data/train_llm_narrated/{train,eval}.jsonl`. Quantifies the three overlap layers between the train and eval splits and compares each `journey_family`'s observed skeleton count to a theoretical upper bound on the bucket-combination space. Source: `scripts/diagnose_data_overlap.py`.

## Headline numbers

- Eval rows: **5,000**
- Eval rows with a train **text-hash** match: **533** (10.7%) — exact narrative duplicates
- Eval rows with a train **structured_events-hash** match: **534** (10.7%) — identical structured payload (a strict superset of text overlap up to one row)
- Eval rows with a train **bucket-event skeleton** match: **4,661** (93.2%) — same event sequence + same bucket tokens, different identifiers/timestamps (reproduces Codex review-018's 4,661/5,000 number)
- Total distinct bucket-event skeletons across train ∪ eval: **2,454**
- Skeletons appearing in BOTH splits: **637** (26.0% of distinct skeletons)
- Skeletons train-only: **1,493**; eval-only: **324**
- Skeletons with mixed labels (legit ↔ fraud): **0**
- H(label | skeleton): **0.0000 bits**

## Per-family stats

| journey_family | n_train | n_eval | skel_train | skel_eval | skel_uniq_all | skel_overlap_eval % | theor_bucket_space | saturation | text_overlap | events_overlap | label_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 8074 | 2019 | 643 | 323 | 701 | 96.8 | 34,828,517,376 | 0.069 | 267 | 267 | legit=10093 |
| cred_stuff | 1184 | 296 | 83 | 69 | 84 | 99.7 | 282,429,536,481 | 0.057 | 0 | 0 | fraud=1480 |
| hn_account_recovery | 1954 | 488 | 28 | 21 | 31 | 99.4 | 241,864,704 | 0.013 | 80 | 80 | legit=2442 |
| hn_large_purchase | 1985 | 496 | 12 | 12 | 12 | 100.0 | 1,679,616 | 0.005 | 175 | 175 | legit=2481 |
| hn_travel | 1991 | 498 | 306 | 153 | 359 | 88.8 | 34,828,517,376 | 0.144 | 0 | 0 | legit=2489 |
| malware_rat | 1184 | 296 | 26 | 24 | 26 | 100.0 | 241,864,704 | 0.018 | 11 | 12 | fraud=1480 |
| mule_chain | 1228 | 307 | 566 | 162 | 702 | 55.4 | 133,524,176,512,370,948,405,598,766,497,792 | 0.457 | 0 | 0 | fraud=1535 |
| phish_takeover | 1233 | 308 | 421 | 162 | 493 | 75.0 | 5,015,306,502,144 | 0.320 | 0 | 0 | fraud=1541 |
| sim_swap | 1167 | 292 | 45 | 35 | 46 | 99.7 | 5,015,306,502,144 | 0.032 | 0 | 0 | fraud=1459 |

Column legend:

- `skel_train`, `skel_eval`, `skel_uniq_all` — count of distinct bucket-event skeletons observed in the train split, eval split, and their union for the family.
- `skel_overlap_eval %` — fraction of eval rows in this family whose skeleton also appears in the family's train rows.
- `theor_bucket_space` — rough upper bound on distinct skeletons the family's template *could* emit, computed from the cardinalities in `data/gen/feature_bucketer.py` and the family's most common event-length (see `BUCKET_CARDINALITY` in this script). Bucket families that vary per event (`amount_bucket`, `txn_velocity`, `recipient_age`, `merchant_risk`, `session_dwell`) are raised to the most-common event-length power; bucket families that are set once per journey (`auth_strength`, `ip_risk`, `geo_distance`, `device_age`) contribute a single factor. This is intentionally coarse — it is an order-of-magnitude estimate, not a counting argument.
- `saturation` — `skel_uniq_all / min(theor_bucket_space, n_train + n_eval)`. Values close to 1 mean the family has saturated its bucket-space at the sample size used; values close to 0 mean the bucket-space is much larger than the sample, so revisiting the same skeleton in train and eval is unlikely.
- `text_overlap`, `events_overlap` — counts of eval rows whose narrative-text hash or `structured_events`-hash appears in train.
- `label_counts` — distribution of `legit` vs `fraud` labels across train+eval for the family. Every family in the current generator has a single label.

## Why train/eval skeleton overlap is near 100%

The dataset's structured stream uses **bucketed** features by design (see `PLAN.md` §3 — bucketed-feature tokens preserve fraud signal in privacy-safe form). Each bucket family has 2-4 values: `amount_bucket` has four, `geo_distance` has three, `ip_risk` has three, `merchant_risk` has two, and so on. The journey templates in `data/gen/journey_templates.py` then pick from these values along narrow, family-specific paths — `gen_clean` always emits `ip_risk=low`, `geo_distance=local`, `auth_strength=mfa_strong`, `device_age=known`; `gen_phish_takeover` always emits `ip_risk=high`, `geo_distance=international`, `auth_strength=password_only`, `device_age=rare`; and the per-event variation reduces almost entirely to `amount_bucket` and `txn_velocity`. Multiplied out, each family's effective bucket-combination space is small enough (hundreds to a few thousand per family) that 20,000 train + 5,000 eval rows easily revisit every cell — which is what the skeleton-uniqueness column in the table above shows.

Because narration in `data/gen/build_dataset.py` happens BEFORE the train/eval split — and `data/gen/narrative_generator.py` caches narratives by `(structured_events_hash, model, temp)` (see `_journey_cache_key` at `narrative_generator.py:252`) — a structured-event payload that occurs twice in the same generation run reuses the cached narrative. When the post-narration split then assigns one copy to train and another to eval, the result is a row pair with identical text AND identical structured payload across splits. That is the mechanism behind the 533 exact-text duplicates Codex reported. The events-only-hash class (1 extra row, dropping 534 in total) is a row whose structured payload matches train but whose narrative happens to differ in whitespace or model output variance.

The skeleton-level overlap (4,661 of 5,000 eval rows) is one layer deeper than the narrative-hash leak. Even if narration were perfectly cached-by-split, and even if every structured-event-hash were unique across train and eval, the bucket-event skeleton would still match because the bucket-combination space per family is smaller than the per-family sample size. This is a property of the synthetic distribution, not a code bug — and it is what makes the structured stream label-deterministic in the observed support (H(label | skeleton) = 0). The event-only classifier's perfect-looking performance on the original eval is partly a consequence: it memorizes a compact categorical mapping over a feature stream that is deterministic in the support its eval inhabits.

## Recommendations for future regenerations

Future regenerations must enforce **pre-narration structured-events-hash stratification**: assign each unique `structured_events_hash` to exactly one split (train OR eval, never both) BEFORE the narration step caches anything, and balance the assignment per `journey_family` so that the requested `eval_frac` is honored at the family level. Concretely: group generated rows by `(journey_family, structured_events_hash)`, then walk the groups in a deterministic order and assign whole groups to the split that is currently under its target eval-fraction. This makes structured-events-hash disjoint between splits and removes the narrative-cache reuse vector in one step.

Add a **post-narration text-hash dedup gate** as a defensive invariant: after narration, hash every row's text and assert no duplicates exist within or across splits. This catches anything the pre-narration gate misses (e.g., LLM output collision on distinct structured payloads).

Neither gate removes the **skeleton-level** overlap, because that is a property of the bucket-combination space, not of the cache or the split. Removing skeleton overlap would require either (a) enlarging the per-family bucket-combination space (more bucket families per event type, finer bucket granularity, more variable per-family templates) or (b) holding out a separate skeleton-disjoint eval set sampled from regions of bucket-space the train set does not cover. (a) is a generator redesign; (b) is a future Day-N deliverable. For the current POC we document the skeleton overlap as a synthetic-data finding and constrain the Day-3 claim accordingly (see `docs/day-2-results.md`).

## Reproducibility

```bash
python3 scripts/diagnose_data_overlap.py \
    --data-dir data/train_llm_narrated \
    --write-md docs/day-2-data-diagnostic.md
```

This script is read-only and idempotent — re-running it does not modify the dataset.
