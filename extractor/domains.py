"""Domain tagging heuristics for empirical statements."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

_DEFAULT_DOMAIN_KEYWORDS: Mapping[str, Sequence[str]] = {
    "physics": ("quantum", "photon", "thermodynamic", "relativity", "gravity", "optics"),
    "mathematics": ("algebra", "calculus", "theorem", "proof", "matrix", "derivative"),
    "computer_science": ("algorithm", "computational", "runtime", "complexity", "neural", "database"),
    "engineering": ("prototype", "circuit", "load-bearing", "tolerance", "mechanical", "sensor"),
    "biology": ("cell", "protein", "genome", "enzyme", "metabolic", "organism"),
    "biometrics": ("iris", "fingerprint", "facial recognition", "gait", "retina"),
    "chemistry": ("molecule", "reaction", "catalyst", "solvent", "polymer"),
    "space": ("orbital", "launch", "payload", "satellite", "spacecraft"),
}


@dataclass
class DomainTagger:
    """Assigns coarse domains to text based on keyword heuristics."""

    keyword_map: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: dict(_DEFAULT_DOMAIN_KEYWORDS)
    )

    def tag(self, text: str) -> list[str]:
        lowered = text.lower()
        domains: list[str] = []
        for domain, keywords in self.keyword_map.items():
            if _contains_any(lowered, keywords):
                domains.append(domain)
        LOGGER.debug("Tagged text '%s' with domains %s", text, domains)
        return domains or ["general_science"]


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(re.search(r"\b" + re.escape(keyword) + r"\b", text) for keyword in keywords)

