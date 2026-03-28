---
name: Working folders are personal only
description: Never create or modify files in Working/ folders — they are the user's personal scratch space for experiments
type: feedback
---

Do not place generated content in Working/ folders. They exist in every section directory and are gitignored for the user's own experimentation.

**Why:** The user explicitly reserves Working/ folders for their own code and experiments. Generated lab content should go in dedicated named folders (e.g., `linear-algebra/`).

**How to apply:** When creating learning materials or notebooks for a section, create a new named subfolder (not Working/) under the relevant section directory.
