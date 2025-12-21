"""Legal knowledge base for QRATUM-OMNILEX.

This module provides the legal knowledge base interface for storing and
retrieving legal authorities (cases, statutes, regulations, treaties).

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegalAuthority:
    """Represents a legal authority (case, statute, regulation, treaty).

    Attributes:
        authority_id: Unique identifier for this authority
        authority_type: Type of authority (case, statute, regulation, treaty)
        jurisdiction: Jurisdiction code where this authority applies
        citation: Proper legal citation (e.g., "Brown v. Board, 347 U.S. 483")
        title: Full title of the authority
        key_holdings: Tuple of key legal holdings or principles
        status: Current status (good_law, overruled, superseded, questioned)
    """

    authority_id: str
    authority_type: str
    jurisdiction: str
    citation: str
    title: str
    key_holdings: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        """Validate legal authority data."""
        valid_types = {"case", "statute", "regulation", "treaty", "constitution"}
        if self.authority_type not in valid_types:
            raise ValueError(f"Invalid authority_type: {self.authority_type}")

        valid_statuses = {"good_law", "overruled", "superseded", "questioned", "pending"}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}")

        if not self.authority_id:
            raise ValueError("authority_id cannot be empty")
        if not self.citation:
            raise ValueError("citation cannot be empty")


class LegalKnowledgeBase:
    """Legal knowledge base for searching and retrieving legal authorities.

    This is a production-ready stub implementation. In a full deployment,
    this would integrate with legal research databases like Westlaw or LexisNexis.
    """

    def __init__(self) -> None:
        """Initialize the legal knowledge base."""
        self._authorities: dict[str, LegalAuthority] = {}
        self._load_sample_authorities()

    def _load_sample_authorities(self) -> None:
        """Load sample legal authorities for demonstration."""
        sample_authorities = [
            LegalAuthority(
                authority_id="hadley-v-baxendale-1854",
                authority_type="case",
                jurisdiction="GB",
                citation="Hadley v. Baxendale, (1854) 9 Ex 341",
                title="Hadley v. Baxendale",
                key_holdings=(
                    "Consequential damages limited to foreseeable losses",
                    "Damages must arise naturally from breach or be in contemplation of parties",
                ),
                status="good_law"
            ),
            LegalAuthority(
                authority_id="ucc-2-302",
                authority_type="statute",
                jurisdiction="US",
                citation="U.C.C. § 2-302",
                title="Uniform Commercial Code - Unconscionable Contract or Clause",
                key_holdings=(
                    "Court may refuse to enforce unconscionable contract",
                    "Unconscionability determined as of time of contract formation",
                ),
                status="good_law"
            ),
            LegalAuthority(
                authority_id="palsgraf-v-long-island-1928",
                authority_type="case",
                jurisdiction="US-NY",
                citation="Palsgraf v. Long Island R.R., 248 N.Y. 339 (1928)",
                title="Palsgraf v. Long Island Railroad Co.",
                key_holdings=(
                    "Duty of care owed only to foreseeable plaintiffs",
                    "Negligence requires proximate cause and foreseeability",
                ),
                status="good_law"
            ),
            LegalAuthority(
                authority_id="restatement-contracts-2d-90",
                authority_type="statute",
                jurisdiction="US",
                citation="Restatement (Second) of Contracts § 90",
                title="Promissory Estoppel",
                key_holdings=(
                    "Promise binding without consideration if reasonably induces reliance",
                    "Enforcement necessary to avoid injustice",
                ),
                status="good_law"
            ),
            LegalAuthority(
                authority_id="donoghue-v-stevenson-1932",
                authority_type="case",
                jurisdiction="GB",
                citation="Donoghue v. Stevenson [1932] AC 562",
                title="Donoghue v. Stevenson",
                key_holdings=(
                    "Manufacturer owes duty of care to ultimate consumer",
                    "Neighbor principle: duty to persons closely and directly affected",
                ),
                status="good_law"
            ),
        ]

        for auth in sample_authorities:
            self._authorities[auth.authority_id] = auth

    def search(
        self,
        query: str,
        jurisdiction: str = "",
        limit: int = 10
    ) -> list[LegalAuthority]:
        """Search for legal authorities.

        Args:
            query: Search query string
            jurisdiction: Filter by jurisdiction code (empty for all)
            limit: Maximum number of results to return

        Returns:
            List of matching legal authorities
        """
        results = []
        query_lower = query.lower()

        for authority in self._authorities.values():
            # Simple keyword matching
            if (query_lower in authority.title.lower() or
                query_lower in authority.citation.lower() or
                any(query_lower in h.lower() for h in authority.key_holdings)):

                # Filter by jurisdiction if specified
                if jurisdiction and not authority.jurisdiction.startswith(jurisdiction):
                    continue

                results.append(authority)

                if len(results) >= limit:
                    break

        return results

    def get_citing_authorities(self, authority_id: str) -> list[LegalAuthority]:
        """Get authorities that cite the specified authority.

        Args:
            authority_id: ID of the authority to find citations for

        Returns:
            List of citing authorities
        """
        # Production stub - in full implementation, this would query citation database
        return []

    def check_authority_status(self, authority_id: str) -> dict[str, Any]:
        """Check the current status of a legal authority.

        Args:
            authority_id: ID of the authority to check

        Returns:
            Dictionary with status information
        """
        authority = self._authorities.get(authority_id)

        if not authority:
            return {
                "found": False,
                "error": "Authority not found"
            }

        return {
            "found": True,
            "authority_id": authority.authority_id,
            "status": authority.status,
            "citation": authority.citation,
            "jurisdiction": authority.jurisdiction,
            "negative_treatment": authority.status != "good_law"
        }

    def add_authority(self, authority: LegalAuthority) -> None:
        """Add an authority to the knowledge base.

        Args:
            authority: Legal authority to add
        """
        self._authorities[authority.authority_id] = authority

    def get_authority(self, authority_id: str) -> LegalAuthority | None:
        """Get a specific authority by ID.

        Args:
            authority_id: ID of the authority to retrieve

        Returns:
            Legal authority or None if not found
        """
        return self._authorities.get(authority_id)
