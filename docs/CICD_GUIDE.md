# How this project is built, tested and shipped

This guide explains how the CI/CD setup in this repo works and why each piece is there.
The ideas carry over to any project; only the commands are Python-specific.

- [1. What the old setup did, and what was wrong with it](#1-what-the-old-setup-did-and-what-was-wrong-with-it)
- [2. The mental model: one path from laptop to production](#2-the-mental-model-one-path-from-laptop-to-production)
- [3. File by file](#3-file-by-file)
- [4. One-time GitHub settings](#4-one-time-github-settings)
- [5. Checklist for your next project](#5-checklist-for-your-next-project)
- [6. Where to go next](#6-where-to-go-next)

---

## 1. What the old setup did, and what was wrong with it

The original repo came from an ML challenge template. It had the right parts (a Dockerfile,
GitHub Actions and a deploy), but they weren't connected into a pipeline and several of them
were broken. Each problem below is a common mistake, so it's worth knowing what to look for.

### Workflows

| Problem | Why it matters |
| --- | --- |
| `main.yml` deployed to production on every push to `develop` **and to a feature branch** (`feature/fine_training`). | Unfinished work could go live. Deploy only from a protected branch or a tag. |
| Nothing ran before the deploy: no lint, no tests, not even a test build. | The first place a bug showed up was production. |
| `docker-image.yml` ran only on `feature/dockerfile`. | Once that branch was merged, the image was never built in CI again. |
| `github-actions-demo.yml` only printed "Hello World". | Leftover noise that still used runner minutes on every push to `main`. |
| `pr-reviewers-workflow.yml` was **active** in this fork and added the template's author as a reviewer on every PR. | Template automation that doesn't fit your repo should be removed, not left running. |
| Both PR workflows used `pull_request_target`. | That trigger runs with write access and secrets, even for PRs from forks. Avoid it unless you understand the [risks](https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/). |
| No `permissions:` block. | Every job got the default token permissions, which may include write access. |
| `actions/checkout@v3`. | Its Node runtime has been retired by GitHub. Without Dependabot, actions quietly go stale. |
| The deploy ran `heroku container:push` twice. | Built and uploaded the image twice for no benefit. |
| Heroku. | Heroku dropped its free tier in November 2022, so the README's demo link is dead. |

### Dockerfile

| Problem | Why it matters |
| --- | --- |
| `ENV PORT = 8000` | Old `ENV key value` syntax, so `PORT` becomes the string `"= 8000"` and `int(os.environ["PORT"])` crashes. The container only ran on Heroku because Heroku overrode `PORT`; `docker run` locally failed. (Reproduced while writing this guide.) |
| `python:3.7` | Python 3.7 reached end of life in June 2023 and gets no security fixes. |
| `faiss-gpu` plus a CUDA 11.3 PyTorch wheel | Several GB of GPU libraries in an image that runs on CPU-only hosts. |
| Downloaded the dataset and re-embedded it **during `docker build`** | Slow, needs the internet, and can change between builds (not reproducible). |
| `models/` was never copied into the image | The fine-tuned model was silently ignored; the image downloaded the base model at runtime. |
| Ran as `root`, no `.dockerignore`, no `HEALTHCHECK`, `reload=True` in production | Weaker security, bigger build context, and no way for the platform to tell whether the app was healthy. |

### Code and repo

- **The search index and the query embedder used different models.** `embed_data.py` always
  used the base model, while `search_query.py` used the fine-tuned model whenever
  `models/semantic-model` existed. Vectors from two different models aren't comparable, so
  local results were quietly wrong. This is why there is now one `settings.py` that every script reads.
- **The committed data was never used.** The code loaded `data/train[:12000]`, but git stored the
  directory as `data/train[_12000]`. So `fetch_load_data.py` downloaded the data again on every build.
- The model and dataset were loaded **on every request** (several seconds per search).
- 64 MB of `datasets` cache files and an 18 MB generated index were committed to git.
- No tests, no linter, and unpinned or inconsistent dependency versions.

---

## 2. The mental model: one path from laptop to production

```
 laptop                 pull request             main                 tag v1.2.0
───────────            ──────────────           ──────               ───────────
 make lint              CI: lint                 CI runs again        Release:
 make test      ──►     CI: test        ──►      (same checks)  ──►    CI again
 make smoke             CI: docker build                               build + push image to GHCR
 pre-commit             + smoke test                                   attest provenance
                        (required to merge)                            deploy (needs approval)
                                                                       check /health
```

Five principles behind it:

1. **Every change goes through a PR, and the PR must be green to merge.** Branch protection
   enforces this. `main` is always releasable.
2. **The same commands run locally and in CI.** The `Makefile` and `tests/smoke_test.sh` are
   what CI calls. If it passes on your laptop, it passes in CI.
3. **Build once, deploy that exact artifact.** The release workflow builds one image, tags it
   with the version, and deploys it. Production never builds from source.
4. **Fail fast and cheap first.** Lint (seconds) → unit tests (a minute) → Docker build and
   smoke test (minutes). The expensive job runs only if the cheap ones pass.
5. **Least privilege everywhere.** Read-only workflow tokens by default, write access granted
   per job, secrets scoped to an environment, and a container that runs as a non-root user.

---

## 3. File by file

### `.github/workflows/ci.yml`: the quality gate

- **Triggers**: `pull_request` (every PR), `push` to `main`, `workflow_dispatch` (a manual
  "Run workflow" button) and `workflow_call` (so the release workflow can reuse it).
- **`permissions: contents: read`**: the token can't push code, edit PRs or publish packages.
- **`concurrency` with `cancel-in-progress`**: if you push twice quickly, the first run is cancelled.
- **`timeout-minutes`**: a stuck job fails after N minutes instead of the default 6 hours.
- **`lint`**: `ruff check` (bugs and style) and `ruff format --check` (formatting).
  `--output-format=github` shows errors as inline annotations on the PR diff.
- **`test`**: installs CPU-only PyTorch (much faster to download), builds the index, then runs
  `pytest`. `cache: pip` reuses downloads between runs.
- **`docker`**: `needs: [lint, test]` makes it wait for the cheap jobs. It builds the image with
  Buildx, using the GitHub Actions cache (`type=gha`) so unchanged layers aren't rebuilt.
  Then it **runs the container and sends real requests** (`tests/smoke_test.sh`). A
  Dockerfile that builds isn't proof that the app starts; the old `PORT` bug would have been
  caught here.

### `.github/workflows/release.yml`: build once, ship that

- **Trigger**: pushing a tag like `v1.2.0`. Releases are deliberate, versioned and easy to
  roll back: re-deploy the previous tag.
- **`uses: ./.github/workflows/ci.yml`** runs the full CI pipeline again on the tagged commit.
  This is a *reusable workflow*, so the checks are defined in one place.
- **GHCR (GitHub Container Registry)**: free for public repos, and it authenticates with the
  built-in `GITHUB_TOKEN`, so there are no long-lived registry passwords to manage.
- **`docker/metadata-action`** turns `v1.2.3` into the tags `1.2.3`, `1.2` and `latest`, and adds
  standard OCI labels (source repo, commit SHA, build time).
- **`attest-build-provenance`** signs a record of which workflow and commit built this image.
  Anyone can check it with `gh attestation verify`. This is a supply-chain security practice.
- **`deploy`**:
  - `environment: production` keeps secrets in a GitHub *Environment*, which can require a
    manual approval and restrict which branches or tags may deploy.
  - It calls a **deploy hook**, which most container hosts provide, then polls `/health` until
    the new version answers. A deploy isn't done until you've checked that it works.
  - `if: vars.DEPLOY_ENABLED == 'true'` keeps the job off until you've chosen a host.

### `.github/dependabot.yml`: staying current automatically

Dependabot opens weekly PRs for GitHub Actions versions, the Docker base image and pip
packages. CI runs on each one, so a green Dependabot PR is usually safe to merge. Related
packages are grouped (the ML stack updates together) to cut down on PR noise. This is how
you avoid ending up with Python 3.7 and `checkout@v3` years later.

### `Dockerfile`: multi-stage, reproducible, small

```
deps     python:3.12-slim + venv + CPU torch + requirements
  └─ index    copy model + dataset → run embed_data.py → training_index.faiss
runtime  fresh python:3.12-slim + venv (from deps) + index (from index) + code, non-root user
```

- **Multi-stage**: the build steps happen in throwaway stages. The final image gets only the
  results through `COPY --from=...`.
- **Order layers from least to most frequently changed.** Dependencies come first and code
  comes last, so editing `app.py` doesn't trigger a reinstall of PyTorch.
- **`--mount=type=cache`** keeps pip's download cache between builds without storing it in the image.
- **`HF_HUB_OFFLINE=1`**: the build and runtime must never download a model. Everything the
  image needs is in the build context, which makes builds reproducible.
- **The index is built inside the image with the same model the API uses**, so the
  mismatch described in section 1 can't happen again.
- **`USER app`**: if the app is compromised, the attacker isn't root in the container.
- **`HEALTHCHECK`** hits `/health`. Docker, and many hosts, use it to restart unhealthy containers.
- **`CMD ["sh", "-c", "exec uvicorn ... --port ${PORT}"]`**: `sh -c` lets the host's `$PORT`
  expand, and `exec` makes uvicorn PID 1 so it gets `SIGTERM` and shuts down gracefully.
- **`ARG TORCH_INDEX_URL`**: build arguments make the image configurable without editing the Dockerfile.

### `.dockerignore`

Without it, `docker build` sends the whole folder, including `.git`, to the builder, and a
`COPY . .` would copy secrets like `.env` into the image. Treat it like `.gitignore` for images.

### `Makefile`, `pyproject.toml`, `.pre-commit-config.yaml`

- `make lint`, `make test`, `make smoke`: one short command for each CI step.
- `pyproject.toml` holds the ruff and pytest settings in one standard place.
- `pre-commit` runs ruff on `git commit`, so formatting problems never reach a PR.

### `tests/`

- `test_api.py` is a **unit test**: it uses a fake search engine, runs in milliseconds and tests
  the HTTP contract (status codes, validation, response shape).
- `test_search.py` is an **integration test**: it uses the real model and index and checks that
  results make sense ("stock market" should return Business articles).
- `smoke_test.sh` is an **end-to-end test** of the real Docker image.

This is the usual testing pyramid: many fast tests and a few slow, realistic ones.

### Requirements files

- `requirements.txt`: runtime dependencies only, with **exact pins** (`==`), so every build installs the same versions.
- `requirements-dev.txt`: the runtime file plus test and lint tools.
- `requirements-train.txt`: extra packages needed only to fine-tune the model. Keeping them
  out of the runtime image keeps it smaller.

---

## 4. One-time GitHub settings

The workflow files are only half of the setup. These settings live in the GitHub UI:

1. **Protect `main`** (Settings → Rules → Rulesets → New branch ruleset, target `main`):
   - Require a pull request before merging.
   - Require status checks to pass: `lint`, `test`, `docker`.
   - Block force pushes.
2. **Protect release tags** (Rulesets → New tag ruleset, pattern `v*`) so only maintainers can
   create releases.
3. **Actions permissions** (Settings → Actions → General → Workflow permissions): set the
   default to **Read repository contents**. The workflows request write access where they need it.
4. **Production environment** (Settings → Environments → New environment `production`):
   - Add yourself as a required reviewer if you want to approve each deploy.
   - Deployment branches and tags: allow only tags matching `v*`.
   - Environment secret `DEPLOY_HOOK_URL`; environment or repo variable `PRODUCTION_URL`.
5. **Repository variable** `DEPLOY_ENABLED=true` (Settings → Secrets and variables → Actions → Variables).
6. **Security** (Settings → Code security): turn on Dependabot alerts, Dependabot security
   updates, secret scanning and push protection.

### Choosing where to deploy

The image is a standard container that listens on `$PORT` and has a `/health` endpoint, so any
container host works. Pick one that can pull from `ghcr.io/<you>/<repo>:latest` and gives you
a deploy hook or API: Render, Railway, Koyeb, Fly.io, Google Cloud Run or AWS App Runner. For
hosts that use a CLI instead of a hook (Fly, Cloud Run), replace the `curl` step with that
CLI's deploy command. Prefer **OIDC** (`id-token: write` plus the cloud's GitHub
integration) over long-lived API keys where the host supports it.

The running container uses about 600 MB of RAM because it holds the model and index in
memory, so choose a plan with at least 1 GB. Many free tiers offer only 512 MB.

---

## 5. Checklist for your next project

**Repository**
- [ ] `main` is protected; changes go through PRs; required checks are set.
- [ ] No generated files, large binaries or secrets in git (use `.gitignore`, Git LFS or a model hub).
- [ ] README covers how to run, test and deploy the project.

**CI**
- [ ] Runs on `pull_request` and on pushes to `main`.
- [ ] `permissions:` is set to the minimum; `concurrency` and `timeout-minutes` are set.
- [ ] Lint → test → build → smoke test, cheapest first, with `needs:` between jobs.
- [ ] Dependency and Docker layer caching is on.
- [ ] The same commands can be run locally (Makefile or scripts).

**Container**
- [ ] Supported base image version, `-slim` variant, multi-stage build.
- [ ] Dependencies installed before code is copied (layer caching).
- [ ] `.dockerignore`, non-root `USER`, `HEALTHCHECK`, listens on `$PORT`, PID 1 handles signals.
- [ ] No network access needed at build time beyond package installs; reproducible builds.

**Release and deploy**
- [ ] Versioned with tags (`vMAJOR.MINOR.PATCH`); the image is built once and deployed by tag or digest.
- [ ] Published to a registry with the token GitHub provides, not a personal password.
- [ ] Deploy runs in a GitHub Environment with scoped secrets and, optionally, required approval.
- [ ] Post-deploy health check; rollback means re-deploying the previous tag.

**Maintenance**
- [ ] Dependabot (or Renovate) for actions, base images and packages.
- [ ] Dependencies pinned; security alerts and secret scanning on.

---

## 6. Where to go next

Ordered roughly by value for effort:

1. **Lockfile with hashes.** `uv lock` (or `pip-compile --generate-hashes`) pins every
   *transitive* dependency, not just the top-level ones. `uv` is also much faster than pip in CI.
2. **Scan the image for vulnerabilities**, with `aquasecurity/trivy-action` or Docker Scout,
   in the `docker` job.
3. **Pin actions to commit SHAs** (`uses: actions/checkout@<sha> # v7`) instead of tags. A tag
   can be moved to point at malicious code; a SHA can't. Dependabot updates SHAs too.
4. **Move the model out of git.** At 88 MB it bloats every clone. Push it to the Hugging Face Hub
   (or use Git LFS) and download a pinned revision in the Docker `index` stage. Note that
   deleting files doesn't shrink git history; you'd need `git filter-repo` for that.
5. **Release automation.** `release-please` can generate the changelog, version bump and tag
   from your commit messages (Conventional Commits).
6. **Preview environments.** Deploy every PR to a temporary URL so changes can be reviewed live.
7. **Observability.** Structured JSON logs, request latency metrics, and an uptime check on `/health`.
