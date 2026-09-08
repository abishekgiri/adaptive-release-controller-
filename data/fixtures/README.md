# Frozen audit fixtures

These files make the corrected numerical replay reproducible without an ignored `data/raw/` directory.

* `github_actions_real.csv`: byte-for-byte copy of the 600-row public-main headline file at commit `a437abf7ecd51e1e9da50834c72b56628e33c697`. SHA256 `fa2b3d5fc04bdfa195afc59779d68ec4cf74173168205c4784eec86a3225e582`. These are workflow-run proxy observations from two repositories, not 600 distinct commits or deployment outcomes. The export lacks workflow/run IDs, event type, branch and collection manifest; provenance cannot be reconstructed completely.
* `travistorrent_smoke.csv`: recovered frozen synthetic fixture from the existing local worktree's ignored raw data, not a sample claimed to come from TravisTorrent. SHA256 `a1a0dcdc752150861bdb4b467b0e1d62f5a525e65fb7e9a94b0d50a74e39ed6a`. There are 1,150 rows and two artificial projects. The exact original generator and its seed are unavailable; the unrelated legacy fixture generator does not recreate this file.

The corrected loader treats decisions as occurring at CI start. Current-run tests executed and duration are replaced by the most recently completed run's values; outcome history uses completion time and excludes unresolved labels. Author experience counts distinct prior SHAs. Artificial delay experiments also delay the availability of those history fields. Empty history is encoded as zero, an explicit cold-start convention rather than an estimated zero risk.

A future publication still needs collection provenance, suitable data-use/license documentation, workflow-aware sampling, and independent projects. Freezing the CSV repairs execution reproducibility, not external validity or provenance.
