"""Adversarial legal simulation for QRATUM-OMNILEX.

This module simulates adversarial legal debate between opposing positions,
modeling courtroom argument dynamics with deterministic execution.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass

from omnilex.knowledge import LegalKnowledgeBase


@dataclass
class LegalArgument:
    """Represents a single legal argument.

    Attributes:
        premise: The premise of the argument
        reasoning: The reasoning supporting the premise
        authority: Legal authority cited
        strength: Strength score (0.0 to 1.0)
    """

    premise: str
    reasoning: str
    authority: str
    strength: float

    def __post_init__(self) -> None:
        """Validate legal argument."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Strength must be between 0.0 and 1.0")


@dataclass
class DebatePosition:
    """Represents one side's position in legal debate.

    Attributes:
        position: Name of the position (e.g., "Plaintiff", "Defendant")
        arguments: List of arguments supporting this position
        strength_score: Overall strength score for the position
    """

    position: str
    arguments: list[LegalArgument]
    strength_score: float

    def __post_init__(self) -> None:
        """Validate debate position."""
        if not 0.0 <= self.strength_score <= 1.0:
            raise ValueError("Strength score must be between 0.0 and 1.0")


class AdversarialSimulator:
    """Simulates adversarial legal debate between opposing sides.

    This simulator models the dialectical process of legal argument,
    generating pro and con positions with supporting arguments.
    """

    def __init__(self, knowledge_base: LegalKnowledgeBase | None = None) -> None:
        """Initialize the adversarial simulator.

        Args:
            knowledge_base: Legal knowledge base to use
        """
        self.knowledge_base = knowledge_base or LegalKnowledgeBase()

    def simulate_adversarial_debate(
        self,
        issue: str,
        facts: str,
        jurisdiction: str,
        rounds: int = 3
    ) -> dict:
        """Simulate adversarial legal debate.

        Args:
            issue: Legal issue to debate
            facts: Factual scenario
            jurisdiction: Jurisdiction code
            rounds: Number of debate rounds

        Returns:
            Dictionary with debate results including both positions
        """
        # Generate plaintiff/prosecution arguments
        plaintiff_position = self._generate_position(
            side="Plaintiff",
            issue=issue,
            facts=facts,
            jurisdiction=jurisdiction,
            favor=True
        )

        # Generate defendant/defense arguments
        defendant_position = self._generate_position(
            side="Defendant",
            issue=issue,
            facts=facts,
            jurisdiction=jurisdiction,
            favor=False
        )

        # Simulate rounds of argument
        debate_rounds = []
        for round_num in range(rounds):
            debate_rounds.append({
                "round": round_num + 1,
                "plaintiff_argument": plaintiff_position.arguments[min(round_num, len(plaintiff_position.arguments) - 1)].premise if plaintiff_position.arguments else "No additional arguments",
                "defendant_argument": defendant_position.arguments[min(round_num, len(defendant_position.arguments) - 1)].premise if defendant_position.arguments else "No additional arguments",
            })

        # Predict likely outcome
        outcome_prediction = self._predict_outcome(
            plaintiff_position,
            defendant_position
        )

        return {
            "issue": issue,
            "jurisdiction": jurisdiction,
            "plaintiff_position": {
                "position": plaintiff_position.position,
                "arguments": [
                    {
                        "premise": arg.premise,
                        "reasoning": arg.reasoning,
                        "authority": arg.authority,
                        "strength": arg.strength
                    }
                    for arg in plaintiff_position.arguments
                ],
                "strength_score": plaintiff_position.strength_score
            },
            "defendant_position": {
                "position": defendant_position.position,
                "arguments": [
                    {
                        "premise": arg.premise,
                        "reasoning": arg.reasoning,
                        "authority": arg.authority,
                        "strength": arg.strength
                    }
                    for arg in defendant_position.arguments
                ],
                "strength_score": defendant_position.strength_score
            },
            "debate_rounds": debate_rounds,
            "outcome_prediction": outcome_prediction
        }

    def _generate_position(
        self,
        side: str,
        issue: str,
        facts: str,
        jurisdiction: str,
        favor: bool
    ) -> DebatePosition:
        """Generate arguments for one side of the debate.

        Args:
            side: Name of the side (e.g., "Plaintiff", "Defendant")
            issue: Legal issue
            facts: Factual scenario
            jurisdiction: Jurisdiction code
            favor: Whether this side favors the claim (True) or opposes it (False)

        Returns:
            Debate position with arguments
        """
        arguments = []

        # Search for relevant authorities
        keywords = self._extract_keywords(issue)
        authorities = []

        for keyword in keywords[:3]:  # Limit search
            found = self.knowledge_base.search(keyword, jurisdiction, limit=2)
            authorities.extend(found)

        # Generate arguments based on authorities
        if favor:
            # Arguments in favor
            if authorities:
                for auth in authorities[:3]:
                    if auth.key_holdings:
                        arguments.append(LegalArgument(
                            premise=f"{side} argues that {auth.key_holdings[0].lower()}",
                            reasoning=f"Under {auth.citation}, the law supports this position",
                            authority=auth.citation,
                            strength=0.75
                        ))

            # Add policy argument
            arguments.append(LegalArgument(
                premise=f"{side} argues public policy supports this interpretation",
                reasoning="This interpretation promotes fairness and predictability in legal relations",
                authority="General policy considerations",
                strength=0.60
            ))
        else:
            # Arguments opposing
            if authorities:
                for auth in authorities[:3]:
                    if len(auth.key_holdings) > 1:
                        arguments.append(LegalArgument(
                            premise=f"{side} distinguishes {auth.citation}",
                            reasoning=f"The facts here differ materially from {auth.title}",
                            authority=auth.citation,
                            strength=0.70
                        ))

            # Add counterargument
            arguments.append(LegalArgument(
                premise=f"{side} argues the facts do not satisfy the required elements",
                reasoning="Critical factual elements are missing or disputed",
                authority="Factual analysis",
                strength=0.65
            ))

        # Ensure at least one argument
        if not arguments:
            arguments.append(LegalArgument(
                premise=f"{side} presents its case based on the facts and law",
                reasoning="The facts and applicable law support this position",
                authority="General legal principles",
                strength=0.50
            ))

        # Calculate overall strength
        if arguments:
            strength_score = sum(arg.strength for arg in arguments) / len(arguments)
        else:
            strength_score = 0.50

        return DebatePosition(
            position=side,
            arguments=arguments,
            strength_score=strength_score
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract legal keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        legal_keywords = [
            "breach", "contract", "damages", "duty", "negligence",
            "consideration", "foreseeability", "reasonable"
        ]

        text_lower = text.lower()
        return [kw for kw in legal_keywords if kw in text_lower]

    def _predict_outcome(
        self,
        plaintiff: DebatePosition,
        defendant: DebatePosition
    ) -> dict:
        """Predict likely outcome based on argument strengths.

        Args:
            plaintiff: Plaintiff's position
            defendant: Defendant's position

        Returns:
            Dictionary with outcome prediction
        """
        # Calculate relative strengths
        total_strength = plaintiff.strength_score + defendant.strength_score

        if total_strength > 0:
            plaintiff_win_prob = plaintiff.strength_score / total_strength
            defendant_win_prob = defendant.strength_score / total_strength
        else:
            plaintiff_win_prob = 0.50
            defendant_win_prob = 0.50

        # Determine likely winner
        if plaintiff_win_prob > defendant_win_prob + 0.1:
            likely_winner = "Plaintiff"
            confidence = "Moderate"
        elif defendant_win_prob > plaintiff_win_prob + 0.1:
            likely_winner = "Defendant"
            confidence = "Moderate"
        else:
            likely_winner = "Uncertain"
            confidence = "Low"

        return {
            "likely_winner": likely_winner,
            "plaintiff_win_probability": round(plaintiff_win_prob, 2),
            "defendant_win_probability": round(defendant_win_prob, 2),
            "confidence": confidence,
            "reasoning": (
                f"Based on argument strength analysis, {likely_winner} "
                f"has the stronger position, though the outcome depends on "
                f"how the trier of fact evaluates the evidence and arguments."
            )
        }
