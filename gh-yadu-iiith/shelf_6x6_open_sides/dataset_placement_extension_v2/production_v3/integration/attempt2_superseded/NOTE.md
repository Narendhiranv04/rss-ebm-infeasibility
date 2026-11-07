# Integration attempt 2 (complete, superseded by attempt 3)

Every check passed except one the orchestrator could not read: the killed smoke B run's ledger ended in a torn line (deliberately
appended without a newline), so the resumed run's 'start' record was glued to that fragment and became unparseable; the orchestrator
then read the killed run's start record and reported the damaged unit as not quarantined. The collector itself was correct: the
damaged unit failed validation (npz_sha_mismatch), was quarantined with REASON.json (smoke_B/quarantine/) and recomputed to a
byte-identical NPZ (sha 524c1679...).

Fix (production v3): collect3d.ledger_open() terminates a torn last line before appending (test
test_ledger_torn_last_line_is_terminated_before_append); the orchestrator requires both start records. Integration repeated from scratch.
