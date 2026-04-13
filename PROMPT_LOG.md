# Prompt Iteration Log — Query Fan-Out Engine

## Draft 1 — Minimal Prompt

```
Generate 10-15 sub-queries for the query "{target_query}" across these types:
comparative, feature_specific, use_case, trust_signals, how_to, definitional.
Return JSON.
```

### Problems
- **No structure enforcement:** The model returned prose with JSON embedded in markdown fences, or added explanatory text before/after the JSON.
- **Inconsistent types:** The model used variations like `"comparison"` instead of `"comparative"`, `"howto"` instead of `"how_to"`.
- **Missing types:** Some responses had 0 queries for `trust_signals` or `definitional` — the model over-indexed on `comparative` and `how_to`.
- **Extra fields:** The model added fields like `"rationale"`, `"relevance_score"`, and `"category"` that weren't requested.

## Draft 2 — Added Type Descriptions and JSON Instruction

```
You are a search query decomposition engine. Generate 10-15 sub-queries for: "{target_query}"

Types:
- comparative: Query vs alternatives
- feature_specific: Specific capability
- use_case: Real-world application
- trust_signals: Reviews, credibility
- how_to: Procedural/instructional
- definitional: Conceptual/"what is"

Return ONLY valid JSON with key "sub_queries", each item having "type" and "query".
```

### Improvements
- Type names became consistent with the descriptions as anchors.
- JSON was returned more reliably.

### Remaining Issues
- Still occasionally wrapped in markdown code fences.
- Still sometimes returned fewer than 2 per type.
- No example output = the model was guessing at the exact schema shape.

## Draft 3 — Added Example, Minimum Per Type, Explicit Constraints

Added a full concrete example output for a different query ("best project management software") showing exactly 12 sub-queries across all 6 types. Added explicit rules:

- "Return ONLY a valid JSON object. No markdown, no commentary, no extra text."
- "At least 2 sub-queries per type"
- "Do NOT add any extra fields beyond 'type' and 'query'"

### Improvements
- Example output anchored the model to the exact schema.
- Minimum-per-type constraint was respected.
- Extra fields stopped appearing.

### Remaining Issue
- Markdown code fences still appeared ~20% of the time. Rather than fighting this in the prompt (diminishing returns), I added `_extract_json()` to strip fences programmatically.

## Final Prompt

The final prompt (in `fanout_engine.py`) combines:

1. **Role assignment:** "You are a search-query decomposition engine" — frames the task precisely.
2. **Type definitions with descriptions:** Each of the 6 types defined with a one-line description, preventing type name confusion.
3. **Explicit rules block:** 6 bullet points covering format (JSON only), schema (exactly two fields), constraints (10-15 total, 2+ per type), and prohibitions (no extra fields).
4. **Concrete example:** Full JSON output for a different query, showing the exact structure expected. This is the single most impactful addition — it reduced format errors from ~30% to ~5%.
5. **Programmatic defense:** `_extract_json()` handles residual markdown fences. `_validate_sub_queries()` filters invalid types and strips extra fields. Combined with 3 retries, the effective failure rate is near zero.

### Key Insight

The example in the prompt is worth more than any number of descriptive instructions. LLMs are pattern-completion engines — showing the pattern once is more effective than describing it five ways.
