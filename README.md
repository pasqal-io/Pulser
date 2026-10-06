# Pulser docs archive

HTML of the released Pulser docs, as published on Read the Docs, one folder per
version (`v1.8.0/`, `v1.7.0/`…). The Pasqal Doc Portal clones this branch to
show the older versions of the Pulser docs without downloading them from Read
the Docs on every build.

`versions.json` lists the archived versions, in the format used by mike:

```json
[{ "version": "v1.8.0", "title": "v1.8.0", "aliases": [] }]
```

Only add a version to `versions.json` once its folder is complete. The current
stable release is archived too, with the `latest` alias: the portal shows it as
latest and doesn't build it twice.

New releases are archived by the `Archive docs` workflow
(`.github/workflows/archive-docs.yml` on `develop`), which runs on each
published release.
