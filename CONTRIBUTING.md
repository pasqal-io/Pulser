# How to Contribute

First of all, thank you for wanting to contribute to Pulser! We have some guidelines you should follow in order to make the contribution process as smooth as possible.

## Reporting a bug or suggesting a feature

The steps to take will depend on what you want to do, but generally you'll want to start by raising an issue. If you have found a bug or have a feature you would like see added to **Pulser**, you're welcome to raise an issue on [Pulser's GitHub issue tracker](https://github.com/pasqal-io/Pulser/issues). Some steps to take here:

1. Do a quick search for keywords over the existing issues to ensure yours has not been added yet.
2. If you can't find your issue already listed, create a new one. Please try to be as clear and detailed as possible in your description.

- If you just want to give a suggestion or report a bug, that's already excellent and we thank you for it! Your issue will be listed and, hopefully, someone will take care of it at some point.
- However, you may also want to be the one solving your issue, which would be even better! In these cases, you would proceed by preparing a [Pull Request](#making-a-pull-request).

## Making a Pull Request

We're thrilled that you want to contribute to Pulser! For general contributions, we use a combination of two Git workflows: the [Forking workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) and the [Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). If you don't know what any of this means, don't worry, you should still be able to make your contribution just by following the instructions detailed below. Nonetheless, in a nutshell, this workflow will have you making a fork from the main Pulser repository and working off a branch from `develop` (**not** `master`). Thus, you'll start your branch from `develop` and end with a pull request that merges your branch back to `develop`. The only exception to this rule is when making a `hotfix`, but in these cases the Pulser development team will take care of it for you.

Here are the steps you should follow to make your contribution:

0. Fork the Pulser repository and add the main Pulser repository as the `upstream`. You only have to do this once and you do so by clicking the "Fork" button at the upper right corner of the [repo page](https://github.com/pasqal-io/Pulser). This will create a new GitHub repo at `https://github.com/USERNAME/Pulser`, where `USERNAME` is your GitHub ID. Then, `cd` into the folder where you would like to place your new fork and clone it by doing:

    ```bash
    git clone https://github.com/USERNAME/Pulser.git
    ```

    **Note**: `USERNAME` should be replaced by your own GitHub ID.

   Then, you'll want to go into the directory of your brand new Pulser fork and add the main Pulser repository as the `upstream` by running:

   ```bash
   git remote add upstream https://github.com/pasqal-io/Pulser.git
   ```

1. Have the related issue assigned to you. We suggest that you work only on issues that have been assigned to you; by doing this, you make sure to be the only one working on this and we prevent everyone from doing duplicate work. If a related issue does not exist yet, consult the [section above](#reporting-a-bug-or-suggesting-a-feature) to see how to proceed.

2. You'll want to create a new branch where you will do your changes. The starting point will be `upstream/develop`, which is where you'll ultimately merge your changes. Inside your fork's root folder, run:

    ```bash
    git fetch upstream
    git checkout -b branch-name-here upstream/develop
    ```

    This will create and checkout the new branch, where you will do your changes.

    **Note**: `branch-name-here` should be replaced by the name you'll give your branch. Try to be descriptive, pick a name that identifies your new feature.

3. Do your work and commit the changes to this new branch. Try to make the first line of your commit messages short but informative; in case you want to go into more detail, you have the option to do so in the next lines.

4. At this point, your branch might have drifted out of sync with Pulser's `develop` branch (the `upstream`). By running

    ```shell
    git pull upstream develop
    ```

   you will fetch the latest changes in `upstream/develop` and merge them with your working branch, at which point you'll have to solve any merge conflicts that may    arise. This will keep your working branch in sync with `upstream/develop`.

5. Finally, you push your code to your local branch:

    ```bash
    git push origin branch-name-here
    ```

6. Once you're happy with your changes, go over to [Pulser's repo page](https://github.com/pasqal-io/Pulser) and start a new Pull Request from `USERNAME:branch-name-here` to `pasqal-io:develop`. Before you do this, make sure your code is obeying the [continuous integration requirements](#continuous-integration-requirements).

7. At this point, you've successfully started the review process. The code reviewers might ask you to perform some changes, which you should push to your local branch in the same way you've done before. You'll see they'll automatically show up in your open PR every time you do this.

## Continuous Integration Requirements

We enforce some continuous integration standards in order to maintain the quality of Pulser's code. Make sure you follow them, otherwise your pull requests will be blocked until you fix them. To check if your changes pass all CI tests before you make the PR, you'll need additional packages, which you can install by running

```shell
pip install -r requirements.txt
```

- **Tests**: We use [`pytest`](https://docs.pytest.org/en/latest/) to run unit tests on our code. If your changes break existing tests, you'll have to update these tests accordingly. Additionally, we aim for 100% coverage over our code. Try to cover all the new lines of code with simple tests, which should be placed in the `Pulser/pulser/tests` folder. To run all tests and check coverage, run:

    ```bash
    pytest --cov pulser
    ```

    All lines that are not meant to be tested must be tagged with `# pragma: no cover`. Use it sparingly,
    every decision to leave a line uncovered must be well justified.

- **Style**: We use [`flake8`](https://flake8.pycqa.org/en/latest/) and the `flake8-docstrings` extension to enforce PEP8 style guidelines. To lint your code with `flake8`, simply run:

    ```bash
    flake8 .
    ```

    To help you keep your code compliant with PEP8 guidelines effortlessly, we suggest you look into installing a linter for your text editor of choice.

- **Format**: We use the [`black`](https://black.readthedocs.io/en/stable/index.html) auto-formatter to enforce a consistent style throughout the entire code base, including the Jupyter notebooks (so make sure to install `black[jupyter]`). It will also ensure your code is compliant with the formatting enforced by `flake8` for you. To automatically format your code with black, just run:

    ```bash
    black .
    ```

    Note that some IDE's and text editors support plug-ins which auto-format your code with `black` upon saving, so you don't have to worry about code format at all.

- **Import sorting**: We use [`isort`](https://pycqa.github.io/isort/) to automatically sort all library imports. You can do the same by running:

    ```bash
    isort .
    ```

- **Type hints**: We use [`mypy`](http://mypy-lang.org/) to type check the code. Your code should have type
annotations and pass the type checks from running:

    ```bash
    mypy
    ```

    In case `mypy` produces a false positive, you can ignore the respective line by adding the `# type: ignore` annotation.

    **Note**: Type hints for `numpy` have only been added in version 1.20. Make sure you have `numpy >= 1.20`
    installed before running the type checks.

### Use `pre-commit` to automate CI checks

[`pre-commit`](https://pre-commit.com/) is a tool to easily setup and manage [git hooks](https://git-scm.com/docs/githooks).

Run

```bash
pre-commit install --hook-type pre-push
```

to install `black`, `isort`, `flake8` and `mypy` hooks in your local repository (at `.git/hooks/` by defaults)
and run them automatically before any push to a remote git repository.
If an issue is found by these tools, the git hook will abort the push. `black` and `isort` hooks may reformat guilty files.

Disable the hooks with

```bash
pre-commit uninstall --hook-type pre-push
```
