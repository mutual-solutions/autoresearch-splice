# Open Questions

## Agent Coordination v2 - 2026-04-15
- [ ] chmod 555 reversibility: should we provide a restore script (`chmod 755`) for when data needs updating? — Matters because updating GT data requires temporarily lifting permissions
- [ ] find -type f returns 0 at top level but subdirs show 91 files — may be a macOS indexing quirk, preflight should use explicit per-subdir counting (already planned), but worth verifying on a clean clone
- [ ] Phase 2 trigger threshold: how many concurrent conflict incidents justify promoting lock infrastructure? — Avoids premature or delayed promotion
