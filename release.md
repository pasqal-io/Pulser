# Release Procedure

## Versioning

Pulser version follows the [Semantic Versioning 2.0.0 specifcation](https://semver.org/spec/v2.0.0.html), which means its versions are numbered as MAJOR.MINOR.PATCH.

According to this specification, three types of releases are envisioned:

- A major release, where breaking changes may be introduced. Here, the MAJOR is bumped and the MINOR and PATCH are reset (`{x}.{y}.{z} -> {x+1}.0.0`)
- A minor release, where only the backwards compatible changes are added. Here, the MINOR is bumped and the PATCH is reset (`{x}.{y}.{z} -> {x}.{y+1}.0`).
- A hotfix, where the PATCH is bumped (`{x}.{y}.{z} -> {x}.{y}.{z+1}`)

Only releases are tracked and tagged in the `master` branch, while development is done in the `develop` branch. To signal this, the version in the `develop` branch should always be one MINOR ahead of `master` and follow the `MAJOR.{MINOR+1}dev{PATCH}` format (e.g. if the latest release tagged in `master` was `1.4.3`, then the version in `develop` should be `1.5dev3`). Through this format, we mark which release is under development and how many patches have occured since its development started (which tells us how many times we brought in changes done directly in `master` through an hotfix).

The version number is centralized in the `VERSION.txt` file and is shared between all the Pulser packages.

## Preparing a scheduled release

A scheduled release is the result of a series of features that were added to the `develop` branch over time. The release process starts out with the creation of a release branch, which should be branched out from `develop` to contain all the desired features and be named `release/v{x}.{y}.{z}`, where `x, y, z` are the MAJOR, MINOR and PATCH of the version to be released (though usually a scheduled release will have PATCH=0).

In the release branch, no other features can be added. Changes to the documentation and bug fixes are allowed, but should only be done when the development in the `develop` branch needs to continue while the release is being prepared; otherwise, do all the changes in `develop` before checking out the release branch. Note that the release branch will ultimately be *merged* to the `master` branch *without being squashed*, so keep the ammount of commits small and document them well to preserve the quality of the history.

Crucially, the release branch must feature a commit changing the development version in `VERSION.txt` to the desired version of the release.

- For a major release: `{x}.{y}dev{z} -> {x+1}.0.0`.
- For a minor release: `{x}.{y}dev{z} -> {x}.{y}.0`.

Finally, open a PR from the `release/v{x}.{y}.{z}` branch to `master`, have someone review and accept the changes introduced in the release branch (all the changes done in `develop` will be there as well, but those have already been reviewed) and merge the branch to `master` **without squashing the commits**. To keep the `master` branch's history clean and informative, replace Github's default merge commit message with `Release v{x}.{y}.{z}`. Optionally, you can also include a summary of the most important changes introduced in the release.


## Preparing a hotfix

Unlike with a scheduled release, a hotfix serves only to fix bugs found in the latest release. The hotfix branch must be branched out from `master` and feature only the changes required to fix any bugs.

Along with the bug fixes, the hotfix branch must also have a commit updating the version with an increment of the PATCH, ie `{x}.{y}.{z} -> {x}.{y}.{z+1}`.

When ready, open a PR to merge the hotfix branch to `master` and, once that is reviewed and accepted, **squash and merge the commits** (note the difference with respect to the scheduled release procedure).


## Writing the release notes

In the [Pulser Releases](https://github.com/pasqal-io/Pulser/releases), draft a new release where you **tag the HEAD of `master`** with **`v{new-version}`** (eg for version 1.2.3, the tag will be `v1.2.3`).

The release notes should include:

- A summary of the main changes introduced (for scheduled releases)
- A list of the bug fixes (if any)
- The full list of changes since the last release. When on the `master` branch, you can get the list of changes since the last tag (which should be the last release) by running:
    ```bash
    previous_version=$(git describe --tags --abbrev=0)
    git log $previous_version..HEAD "--pretty=%h %s"
    ```
    If you've tagged the latest version already, just manually replace `previous_version` with the previously released version number.
- A thank you to all the contributors. Reusing the `previous_version` variable defined before, you can get this list by running:
    ```bash
    git log $previous_version..HEAD --pretty="%an" | sort | uniq
    ```
    Note that this will list only the authors of the PRs to `develop`. If you know of other contributors that do not appear listed, make sure to add them.


## Deploying the release

The publication of the release notes will trigger a Github Actions workflow that automatically builds all the packages, publishes them to PyPI and runs some tests to check the publication succeed.

Make sure this workflow ran without errors - if not, assess why it failed and, if it was a third-party problem (e.g. a network connection issue), try to rerun the workflow.
However, in the unlikely scenario that the deployment failed, it is more likely that there is something that needs to be fixed, in which case you should make an hotfix right away.

The "stable" version of the documentation on Read the Docs will be automatically updated to the new released version. To get the latest documentation on [Pasqal's docs portal](https://docs.pasqal.com/pulser), notify the contact person responsible for the docs portal, so that they will manually rebuild and deploy. Please check afterwards if there are any rendering issues and report if any. 

## Merging the changes back to `develop`

Finally, you must open a PR from `master` to `develop` to merge the changes that occured in `master`. In this PR, you must also bump the version you just released, `{x}.{y}.{z}`, to the new development version, `{x}.{y+1}dev{z}` (e.g. `1.8.3 -> 1.9dev3`).

Once the PR is accepted, merge it **without squashing** (again, replacing the merge commit message with something more informative) and that's it, you're done!
