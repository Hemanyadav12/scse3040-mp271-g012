from pathlib import Path


def test_ci_uses_python_version_supported_by_requirements():
    workflow = Path(".github/workflows/ci.yml").read_text()

    assert 'python-version: "3.12"' in workflow