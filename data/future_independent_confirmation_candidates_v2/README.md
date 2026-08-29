# Future independent-confirmation candidate list v2

Status: outcome-free candidate inventory only. It is not a frozen analysis
sample, an authorization to parse archives, or an authorization to access
labels, conditional residuals, or fitted results.

Created: 2026-08-26

## Scope

This v2 expands v1 to every `.rar` archive currently in
`D:\CS2_raw_matches`: 78 archives totaling 64,636,114,767 bytes (60.197 GiB).
The v2 candidate set is the union of the 50 archives in
`../future_independent_confirmation_candidates_v1/candidate_archives.csv` and
the 28 additional archives in `additional_archives.csv`. The union must be
ordered lexically by `source_archive` before reproducing the metadata-index
hash.

Archive names and filesystem metadata were the only selection inputs. Archive
contents, gameplay outcomes, labels, residuals, and fitted results were not
read. This expansion does not freeze a sample, establish independent
eligibility, or modify the existing `lag-1` confirmation or any D1/E1 result.

The full ordered metadata index has SHA-256
`7d41a17abe9373677e38124d2a6bbdec120b3206c023d1fe7ce94e272500662a`.
It hashes newline-joined `source_archive|bytes|last_write_utc` triples and is
not a content hash.

## Required before freezing

The v1 requirements remain unchanged: research-use rights, outcome-free exact
`match_id`/`map_id` extraction, zero-overlap and prior-access audits, and the
fixed `p_hat` feature-contract/state-precondition audit all must pass. Before
any label access, the user must explicitly authorize a new frozen confirmation
protocol that inherits the 60-second state specification, D1-A reference
scaling, current-row eligibility, continuous-time adjustment, two-way CRV1,
D3/D4 gates, and the single two-sided `H0: beta = 0` test.
