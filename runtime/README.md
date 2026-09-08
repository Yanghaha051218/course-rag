# Local runtime data

This directory is reserved for local application state. Planned locations are:

- `uploads/` for ingested source files;
- `qdrant/` for local vector-store data; and
- SQLite database files directly under `runtime/`.

Runtime contents are ignored by Git. The empty placeholder files keep the
directory structure visible without committing user data.
