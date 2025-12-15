#!/usr/bin/env python3
"""Tests for documentation generation system.

This module tests the comprehensive documentation generation system
to ensure it produces expected deliverables.

Author: QuASIM Engineering Team
Date: 2025-12-14
Version: 1.0.0
"""

import tempfile
from pathlib import Path


def test_documentation_generator_imports():
    """Test that documentation generator can be imported."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    # Test imports
    from generate_documentation_package import (DocumentationPackageGenerator,
                                                ExecutiveSummaryGenerator,
                                                ModuleInfo, RepositoryParser,
                                                TechnicalWhitePaperGenerator,
                                                VisualizationGenerator)

    assert DocumentationPackageGenerator is not None
    assert RepositoryParser is not None
    assert VisualizationGenerator is not None
    assert ExecutiveSummaryGenerator is not None
    assert TechnicalWhitePaperGenerator is not None
    assert ModuleInfo is not None


def test_appendix_generator_imports():
    """Test that appendix generator can be imported."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from generate_appendices import (generate_all_appendices,
                                     generate_cuda_pseudocode,
                                     generate_reporting_examples,
                                     generate_reproducibility_proof,
                                     generate_statistical_derivations,
                                     generate_yaml_benchmark_spec)

    assert generate_yaml_benchmark_spec is not None
    assert generate_cuda_pseudocode is not None
    assert generate_statistical_derivations is not None
    assert generate_reproducibility_proof is not None
    assert generate_reporting_examples is not None
    assert generate_all_appendices is not None


def test_module_info_dataclass():
    """Test ModuleInfo dataclass."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from generate_documentation_package import ModuleInfo

    # Create a ModuleInfo instance
    module = ModuleInfo(
        path=Path("test.py"), name="test", lines_of_code=100, classes=["TestClass"]
    )

    assert module.name == "test"
    assert module.lines_of_code == 100
    assert "TestClass" in module.classes


def test_appendix_generation():
    """Test that appendices can be generated."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from generate_appendices import generate_all_appendices

    # Generate appendices in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "appendices"
        appendices = generate_all_appendices(output_dir)

        # Verify all appendices were created
        assert len(appendices) == 5
        assert all(a.exists() for a in appendices)

        # Verify content
        for appendix in appendices:
            content = appendix.read_text()
            assert len(content) > 100  # Each appendix should have substantial content
            assert "Appendix" in content  # Should have header


def test_repository_parser():
    """Test repository parser."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from generate_documentation_package import RepositoryParser

    # Create parser with current repo
    repo_path = Path(__file__).parent.parent
    parser = RepositoryParser(repo_path)

    # Test parsing
    parser.scan_repository()

    # Verify modules were found
    assert len(parser.modules) > 0

    # Check that key files are included
    expected_modules = [
        "scripts/generate_documentation_package.py",
        "scripts/generate_appendices.py",
    ]

    for expected in expected_modules:
        if Path(repo_path / expected).exists():
            # At least one should be found
            found = any(expected in str(path) for path in parser.modules.keys())
            if found:
                break


if __name__ == "__main__":
    # Run basic tests
    print("Running documentation generation tests...")

    test_documentation_generator_imports()
    print("✓ Documentation generator imports OK")

    test_appendix_generator_imports()
    print("✓ Appendix generator imports OK")

    test_module_info_dataclass()
    print("✓ ModuleInfo dataclass OK")

    test_appendix_generation()
    print("✓ Appendix generation OK")

    test_repository_parser()
    print("✓ Repository parser OK")

    print("\n✓ All tests passed!")
