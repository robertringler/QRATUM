"""Safety and ethics guard for QuNimbus integration."""


class QNimbusGuard:
    """Policy guard for QuNimbus queries.

    This guard implements safety checks to prevent dangerous or
    unethical query patterns.
    """

    def __init__(self):
        """Initialize guard with banned patterns."""

        self.banned_patterns = [
            "bio-weapons",
            "bioweapons",
            "biological weapons",
            "personal re-identification",
            "mass manipulation",
            "surveillance",
            "targeted attacks",
            "exploit",
            "vulnerability exploitation",
        ]

    def allow_query(self, query: str) -> bool:
        """Check if query is allowed.

        Parameters
        ----------
        query : str
            Query string to validate

        Returns
        -------
        bool
            True if query is allowed, False otherwise

        Examples
        --------
        >>> guard = QNimbusGuard()
        >>> guard.allow_query("climate simulation")
        True
        >>> guard.allow_query("bio-weapons design")
        False
        """

        query_lower = query.lower()
        return all(pattern not in query_lower for pattern in self.banned_patterns)

    def get_rejection_reason(self, query: str) -> str:
        """Get reason for query rejection.

        Parameters
        ----------
        query : str
            Query string

        Returns
        -------
        str
            Rejection reason or empty string if allowed

        Examples
        --------
        >>> guard = QNimbusGuard()
        >>> guard.get_rejection_reason("bio-weapons")
        'Query contains banned pattern: bio-weapons'
        """

        query_lower = query.lower()

        for pattern in self.banned_patterns:
            if pattern in query_lower:
                return f"Query contains banned pattern: {pattern}"

        return ""
