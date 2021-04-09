# How to Contribute

First of all, thank you for wanting to contribute to Pulser! We have some guidelines you should follow in order to make the contribution process as smooth as possible.

## Reporting a bug or suggesting a feature

The steps to take will depend on what you want to do, but generally you'll want to start by raising an issue. If you have found a bug or have a feature you would like see added to **Pulser**, you're welcome to raise an issue on [Pulser's GitHub issue tracker](https://github.com/pasqal-io/Pulser/issues). Some steps to take here:

1. Do a quick search for keywords over the existing issues to ensure yours has not been added yet.
2. If you can't find your issue already listed, create a new one. Please try to be as clear and detailed as possible in your description.
- If you just want to give a suggestion or report a bug, that's already excellent and we thank you for it! Your issue will be listed and, hopefully, someone will take care of it at some point.
- However, you may also want to be the one solving your issue, which would be even better! In these cases, you would proceed by preparing a [Pull Request](#making-a-pull-request).


## Making a Pull Request

We're thrilled that you want to contribute to Pulser. Here are the steps you should follow to make your contribution:

0. Fork the Pulser repository. You only have to do this once and you do so by clicking the "Fork" button at the upper right corner of the [repo page](https://github.com/pasqal-io/Pulser). This will create a new GitHub repo at `https://github.com/USERNAME/Pulser`, where `USERNAME` is your GitHub ID. Then, `cd` into the folder where you would like to place your new fork and clone it by doing:
    ```bash
    git clone https://github.com/USERNAME/Pulser.git
    ```
    **Note**: `USERNAME` should be replaced by your own GitHub ID.

1. Have the related issue assigned to you. We suggest that you work only on issues that have been assigned to you; by doing this, you make sure to be the only one working on this and we prevent everyone from doing duplicate work. If a related issue does not exist yet, consult the [section above](#reporting-a-bug-or-suggesting-a-feature) to see how to proceed.

2. You'll want to create a new branch where you will do your changes. **Do not push changes to your fork's `master` branch**, it should be used only to keep your fork in sync with Pulser's `master` branch (more on how to do this later). Go to the location where you cloned your fork and do:
    ```bash
    cd Pulser
    git checkout -b branch-name-here
    ```
    This will create and checkout the new branch where you will do your changes.

3. Do your work and commit changes to this new branch.

4. At this point, your fork's `master` branch might have drifted out of sync with Pulser's `master` branch (the `upstream`). The following lines will sync your local repo's `master` with `upstream/master` and then merge the local `master` with your working branch, at which point you'll have to solve any merge conflicts that may arise.
    ```shell
    # Track the upstream repo (you only have to do this one time):
    git remote add upstream https://github.com/pasqal-io/Pulser.git

    # Update your local master.
    git fetch upstream
    git checkout master
    git merge upstream/master
    # Merge local master into your branch.
    git checkout branch-name-here
    git merge master
    ```
5. Finally, you push your code to your new branch:
    ```bash
    git push origin branch-name-here
    ```

6. Once you're happy with your changes, go over to [Pulser's repo page](https://github.com/pasqal-io/Pulser) and start a new Pull Request from `USERNAME:branch-name-here` to `pasqal-io:master`. Before you do this, make sure your code is obeying the [continuous integration requirements](#continuous-integration-requirements).

7. At this point, you've successfully started the review process. The code reviewers might ask you to perform some changes, which you should push to your local branch in the same way you've done before. You'll see they'll automatically show up in your open PR every time you do this.

## Continuous Integration Requirements

We enforce some continuous integration standards in order to maintain the quality of Pulser's code. Make sure you follow them, otherwise your pull requests will be blocked until you fix them. To check if your changes pass all CI tests before you make the PR, you'll need additional packages, which you can install by running

```shell
cd Pulser
pip install -r requirements.txt
```

- **Tests**: We use [pytest](https://docs.pytest.org/en/latest/) to run unit tests on our code. If your changes break existing tests, you'll have to update these tests accordingly. Additionally, we aim for 100% coverage over our code. Try to cover all the new lines of code with simple tests, which should be placed in the `Pulser/pulser/tests` folder. To run all tests and check coverage, run:
    ```bash
    pytest --cov pulser
    ```
All lines that are not meant to be tested must be tagged with `# pragma: no cover`. Use it sparingly,
every decision to leave a line uncovered must be well justified.

- **Style**: We use [flake8](https://flake8.pycqa.org/en/latest/) to enforce PEP8 style guidelines. To lint your code with `flake8`, simply run:
    ```bash
    flake8 .
    ```
