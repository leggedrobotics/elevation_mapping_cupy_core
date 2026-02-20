# Publishing Guide

This guide explains how to publish `elevation_mapping_cupy_core` to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account on [PyPI](https://pypi.org) if you don't have one
2. **Trusted Publishing Setup** (Recommended): Configure trusted publishing in PyPI for automatic publishing via GitHub Actions

## Method 1: Automated Publishing via GitHub Actions (Recommended)

This is the easiest and most secure method using GitHub Actions with trusted publishing.

### Steps:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.0"  # Update to your new version
   ```

2. **Commit and push**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.0"
   git push origin main
   ```

3. **Create a git tag**:
   ```bash
   git tag v0.1.0  # Must match version in pyproject.toml (without 'v' prefix)
   git push origin v0.1.0
   ```

4. **Create a GitHub Release**:
   - Go to your repository on GitHub
   - Click "Releases" → "Create a new release"
   - Select the tag you just created (e.g., `v0.1.0`)
   - Add release notes
   - Click "Publish release"

5. **The workflow will automatically**:
   - Build the package (wheel and source distribution)
   - Verify the package
   - Publish to PyPI using trusted publishing

### Setting up Trusted Publishing:

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Navigate to "API tokens" → "Add API token"
3. Select "Trusted publishers" → "Add"
4. Fill in:
   - **PyPI project name**: `elevation_mapping_cupy_core`
   - **Owner**: Your GitHub organization/username
   - **Repository name**: `elevation_mapping_cupy_core`
   - **Workflow filename**: `publish.yml`
5. Click "Add trusted publisher"

## Method 2: Manual Publishing with uv

If you prefer to publish manually using `uv` (recommended for speed):

### Steps:

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or: pip install uv
   ```

2. **Update version** in `pyproject.toml`

3. **Build the package**:
   ```bash
   uv build
   ```

4. **Publish to PyPI**:
   ```bash
   # Using trusted publishing (recommended)
   uv publish --trusted-publishing always
   
   # Or using API token
   uv publish --token pypi-<your-token-here>
   ```

### Using API Token:

1. Create an API token on PyPI: Account Settings → API tokens
2. Use it as:
   ```bash
   uv publish --token pypi-<your-token-here>
   ```

### Publishing to TestPyPI:

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-<testpypi-token>
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., `1.2.3`)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Testing Before Publishing

Always test the package locally before publishing:

### Using uv (recommended):

```bash
# Build the package
uv build

# Install from the built wheel
uv pip install dist/elevation_mapping_cupy_core-*.whl

# Test it
uv run python -c "from elevation_mapping_cupy import ElevationMap; print('Import successful!')"
```

### Using traditional tools:

```bash
# Build the package
python -m build

# Install from the built wheel
pip install dist/elevation_mapping_cupy_core-*.whl

# Test it
python -c "from elevation_mapping_cupy import ElevationMap; print('Import successful!')"
```

## Troubleshooting

### "Package already exists" error
- The version number must be unique
- Increment the version in `pyproject.toml`

### "Invalid version" error
- Ensure version follows semantic versioning (e.g., `0.1.0`, not `0.1`)

### Trusted publishing not working
- Verify the repository name matches exactly
- Check that the workflow file is named `publish.yml`
- Ensure the tag format matches (e.g., `v0.1.0` for version `0.1.0`)
