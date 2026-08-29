# Future independent-confirmation candidate list v1

Status: outcome-free candidate inventory only. It is not a frozen analysis
sample, an authorization to parse archives, or an authorization to access
labels, conditional residuals, or fitted results.

Created: 2026-08-26

## Purpose

This compact inventory is the starting point for a future independent
confirmation of the fixed 60-second exponentially decayed residual-history
state. It does not alter the existing `lag-1` confirmation or reclassify any
D1/E1 result.

## Selection

`candidate_archives.csv` lists 50 raw match archives (40 ESL Pro League Season
21 archives and the first 10 IEM Katowice 2025 archives in ordinal filename
order). The rule uses archive names and filesystem metadata only, not gameplay
outcomes, labels, residuals, or prior estimates. The 50-archive target is a
compact planning size: the existing 100-match D1-B/C analyses met the fixed
D4 precision gates, whereas the 30-match cluster minimum would be expected to
be underpowered if standard errors scale approximately with the inverse square
root of match count. This is a planning rationale, not an achieved precision
claim.

The selected archive metadata index has SHA-256
`c1f5e78f40ed151598eb78c6b008694997dc125bd36daa074c7f5d143bf82a9c`.
It hashes the ordered triples `source_archive|bytes|last_write_utc`; it is not
a hash of archive contents.

## Preliminary provenance screen

At inventory creation, the selected archive basenames had no exact-name overlap
with files recursively enumerated in the D1, E1, development, external
confirmation-pilot, and external-pilot raw archive locations. This is only a
preliminary screen: exact `match_id`/map de-duplication and prior-access audit
remain required before a sample can be frozen.

## Required next gates

1. Verify every listed archive exists and reproduces the recorded metadata.
2. Establish research-use rights and map each archive to its exact `match_id`
   and `map_id` without reading labels.
3. Prove zero overlap with every D1-A/B/C and E1-v2 match/map in the existing
   scope locks, and record all prior data/result-access paths.
4. Build an outcome-withheld candidate feature cache and verify the fixed
   `p_hat` feature contract and 60-second state prerequisites.
5. Before any label access, obtain explicit authorization and create a new
   frozen confirmation protocol, input lock, execution script, checkpointed
   result root, and report path. That protocol must inherit the already fixed
   60-second state, D1-A reference scaling, current-row eligibility,
   continuous-time adjustment, two-way CRV1 inference, D3/D4 gates, and the
   single two-sided `H0: beta = 0` test.

Any failed gate excludes the archive from a future frozen sample; it must not
be replaced after labels are accessed.
