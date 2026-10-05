# Pulser docs archive

HTML of the released Pulser docs, as published on Read the Docs, one folder per
version (`v1.8.0/`, `v1.7.0/`…). The Pasqal Doc Portal clones this branch to
show the older versions of the Pulser docs without downloading them from Read
the Docs on every build.

`versions.json` lists the archived versions, in the format used by mike:

```json
[{ "version": "v1.8.0", "title": "v1.8.0", "aliases": [] }]
```

Only add a version to `versions.json` once its folder is complete, and don't
archive the current stable release: the portal already shows it as latest.
