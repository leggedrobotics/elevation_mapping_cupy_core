# elevation_mapping_cupy_core

Core CuPy-based elevation mapping library extracted from elevation_mapping_cupy.

## Installation

```bash
pip install elevation_mapping_cupy_core
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/leggedrobotics/elevation_mapping_cupy_core.git
cd elevation_mapping_cupy_core

# Install in editable mode with test dependencies
pip install -e ".[test]"
```

### Running Tests

```bash
pytest elevation_mapping_cupy/tests/
```

### Building

Using `uv` (recommended):
```bash
uv build
```

Or using traditional tools:
```bash
python -m build
```

## CI/CD

This project uses GitHub Actions for continuous integration:

- **CI Workflow** (`ci.yml`): Runs tests on Python 3.10, 3.11, and 3.12
- **GPU Test Workflow** (`test-gpu.yml`): Tests with CUDA support (when available)
- **Publish Workflow** (`publish.yml`): Automatically publishes to PyPI on releases

### Publishing to PyPI

1. Update version in `pyproject.toml`
2. Create a git tag: `git tag v0.1.0`
3. Push tag: `git push origin v0.1.0`
4. Create a GitHub release with the same tag
5. The workflow will automatically publish to PyPI

Alternatively, use trusted publishing with GitHub Actions (recommended):
- Configure PyPI API token in GitHub repository settings
- The workflow uses `pypa/gh-action-pypi-publish` for secure publishing

## License

MIT
