import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor.domains import DomainTagger


def test_domain_tagger_matches_multiple_domains() -> None:
    tagger = DomainTagger()
    domains = tagger.tag("The quantum algorithm optimizes satellite payload constraints.")
    assert "physics" in domains
    assert "computer_science" in domains
    assert isinstance(domains, list)


def test_domain_tagger_defaults_to_general() -> None:
    tagger = DomainTagger()
    assert tagger.tag("This sentence has no obvious domain.") == ["general_science"]
