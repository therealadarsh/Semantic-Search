# .github 🤫

Hi! Glad to see you made it here! You're probably wondering: 

> _What's in this directory? What are the files for?_

Let me help you with that:

_The `.github` directory is where to store all the GitHub related files._

> So, what do we have here already?

We have a few things related to [Github Actions](https://docs.github.com/en/actions) (which is maybe the main reason you're here).

## Overview of contents

The `workflows/` directory contains 3 GitHub Actions workflows:

1. GitHub Actions Demo in the `github-actions-demo.yml` file,
2. Close PRs for Incorrect Submissions in the `pr-close-workflow.yml` file, and
3. Auto Assign PR Challenge Submission Reviewer in the `pr-reviewers-workflow.yml` file

The 1st one is meant to show you an example of a simple GitHub Actions Workflow if you're looking to automate any part of your challenge.

The 2nd and 3rd are meant to help you with your submision, so I'd recommend not touching them. Also don't touch the `config-reviewers.yml` file as that is the config for the Close PRs for Incorrect Submissions workflow.

❗If you want to use GitHub Actions (including the Pull Request workflows already made for you) read the following sections.❗

## Enable GitHub Actions

GitHub Actions aren't enabled by default when you fork a repository. You need to manually enable them. Luckily it's a simple process.

### Steps

1. Go the the `Actions` tab in your forked repository.
![github-actions-tab](https://user-images.githubusercontent.com/19933411/165649783-608542cd-ee30-441e-af6f-023baf44bd02.png)
2. Click on the big green button in the middle of the screen that says `I understand my workflows, go ahead and enable them`.
![github-enable-actions](https://user-images.githubusercontent.com/19933411/165649881-8ea2b71f-1c5b-4c25-939a-27bd81059ee5.png)
3. *Voila!* You can now use your own GitHub Actions to automate your submission and get some bonus points! 🚀

**❗Note:❗** Just enabling GitHub actions won't enable the PR workflows. To enable them follow the steps in the section below.

### Enable Pull Request GitHub Actions Workflows

**❗Note:❗** It's not mandatory you do this, however 1 workflow will help you a bit with your submission.

1. Go to the `Settings` tab of your forked repository.
 
![github-settings-tab](https://user-images.githubusercontent.com/19933411/165649997-e0016b93-75a2-43e0-8c23-6c1371afd911.png)

2. On the column on the left, under `Code and automation`, click on the `Actions` dropdown 
 
![github-settings-general](https://user-images.githubusercontent.com/19933411/165650128-b34bae56-16a4-4496-9430-67d83aa01c6b.png)

3. Then click on `General`.

![github-settings-actions-general](https://user-images.githubusercontent.com/19933411/165650193-692d47f0-839e-4c15-8a00-b3a92fe9553d.png)

4. Now on the main body of the page scroll down to `Workflow permissions` and check the box that says `Read and write permissions` and click the Save button below.![github-settings-actions-workflow-permissions](https://user-images.githubusercontent.com/19933411/165650238-876b7be3-0601-4a66-9b31-70e42cd9067e.png)

5. *Voila!* The PR workflows are now enabled on your repository
