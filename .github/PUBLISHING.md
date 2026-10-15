# Publishing publiplots to PyPI

This guide covers how to publish the `publiplots` package to PyPI using either automated GitHub Actions or manual publishing with `uv`.

## Prerequisites

- Maintainer access to this repository
- PyPI account (create at https://pypi.org/account/register/)
- TestPyPI account for testing (optional, create at https://test.pypi.org/account/register/)

## Automated Publishing with GitHub Actions (Recommended)

The repository includes a GitHub Actions workflow that automatically publishes to PyPI when you create a GitHub release.

### Setup (One-time)

1. **Configure Trusted Publishing on PyPI** (recommended, no API tokens needed):

   a. Go to https://pypi.org/manage/account/publishing/

   b. Scroll to "Add a new pending publisher"

   c. Fill in:
      - **PyPI Project Name**: `publiplots`
      - **Owner**: `jorgebotas`
      - **Repository name**: `publiplots`
      - **Workflow name**: `publish.yml`
      - **Environment name**: `pypi`

   d. Click "Add"

   e. (Optional) Repeat for TestPyPI at https://test.pypi.org/manage/account/publishing/
      - Use environment name: `testpypi`

2. **Alternative: Using API Tokens** (if not using trusted publishing):

   a. Generate an API token on PyPI:
      - Go to https://pypi.org/manage/account/token/
      - Create a new API token
      - Scope it to the `publiplots` project (after first upload)

   b. Add the token as a GitHub secret:
      - Go to repository Settings → Secrets and variables → Actions
      - Create a new secret named `PYPI_API_TOKEN`
      - Paste your PyPI API token

   c. Modify `.github/workflows/publish.yml` to use the token:
      ```yaml
      - name: Publish distribution to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
      ```

### Publishing a Release

1. **Update the version** in `pyproject.toml`:
   ```toml
   version = "0.2.0"  # Update this
   ```

2. **Commit and push** the version change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.2.0"
   git push
   ```

3. **Create a GitHub Release**:

   a. Go to https://github.com/jorgebotas/publiplots/releases/new

   b. Click "Choose a tag" and create a new tag (e.g., `v0.2.0`)

   c. Set the release title (e.g., "v0.2.0")

   d. Add release notes describing changes

   e. Click "Publish release"

4. **Monitor the workflow**:
   - Go to Actions tab to watch the build and publish process
   - The package will be automatically published to PyPI

### Testing with TestPyPI

Before publishing to PyPI, you can test with TestPyPI:

1. Go to Actions tab
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. This will trigger the TestPyPI upload

## Manual Publishing with uv

For local testing or if you prefer manual control:

### Setup (One-time)

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create PyPI API token** (if not using trusted publishing):
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Save it securely (you'll need it for publishing)

### Publishing Steps

1. **Update the version** in `pyproject.toml`:
   ```toml
   version = "0.2.0"  # Update this
   ```

2. **Clean previous builds**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

3. **Build the package**:
   ```bash
   uv build
   ```

   This creates distribution files in the `dist/` directory:
   - `publiplots-0.2.0.tar.gz` (source distribution)
   - `publiplots-0.2.0-py3-none-any.whl` (wheel)

4. **Inspect the build** (optional):
   ```bash
   ls -lh dist/
   tar -tzf dist/publiplots-*.tar.gz | head -20
   ```

5. **Test installation locally**:
   ```bash
   uv pip install dist/publiplots-*.whl
   python -c "import publiplots; print(publiplots.__version__)"
   ```

6. **Publish to TestPyPI** (recommended first time):
   ```bash
   uv publish --publish-url https://test.pypi.org/legacy/
   ```

   Enter your TestPyPI credentials when prompted.

   Test the installation:
   ```bash
   uv pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     publiplots
   ```

7. **Publish to PyPI**:
   ```bash
   uv publish
   ```

   Enter your PyPI credentials or API token when prompted.

8. **Verify on PyPI**:
   - Check https://pypi.org/project/publiplots/
   - Test installation:
     ```bash
     uv pip install publiplots
     ```

## Version Management

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.2.0): New functionality, backwards-compatible
- **PATCH** version (0.1.1): Backwards-compatible bug fixes

## Pre-release Checklist

Before publishing a new version:

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with changes
- [ ] Run tests: `pytest`
- [ ] Check code formatting: `black . && ruff check .`
- [ ] Build locally and test installation
- [ ] Test on TestPyPI first
- [ ] Create and tag a git commit
- [ ] Create GitHub release (for automated publishing)

## Troubleshooting

### "File already exists" error
You cannot re-upload the same version. Increment the version number and rebuild.

### Missing dependencies
If users report missing dependencies, verify `dependencies` in `pyproject.toml`.

### Import errors after installation
Make sure the package structure in `pyproject.toml` is correct:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/publiplots"]
```

### Testing specific Python versions
Use `uv` to test with different Python versions:
```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
pytest
```

## Resources

- [PyPI Help](https://pypi.org/help/)
- [Python Packaging Guide](https://packaging.python.org/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [GitHub Actions Publishing](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
