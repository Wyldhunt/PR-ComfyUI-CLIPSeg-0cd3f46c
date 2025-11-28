# Agent Notes

- Project context: ComfyUI and PostgreSQL integration for an Agentic RAG Companion AI using n8n, Ollama, Postgres, and ComfyUI; focus on easy cross-referencing of stored data from ComfyUI nodes without unnecessary schema changes.
- Document workflow or data-handling conventions that help with cross-table access (naming patterns, key fields, etc.).
- Keep database schema changes minimal unless required; prefer documenting mappings or helper utilities first.
- Whenever you make a change that might be important to remember later, update this AGENTS.md with the relevant notes.
- Be mindful of code style notes from system instructions (e.g., avoid wrapping imports in try/except) and keep new notes here for future tasks.
- Mask combination updates: ensure masks are aligned to the input image shape, sum only provided masks, and clamp/normalize combined masks into [0,1] before visualizing or returning results.
