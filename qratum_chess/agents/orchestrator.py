"""Multi-agent orchestrator for chess engine decision making.

Coordinates multiple specialized agents to produce high-quality move decisions:
1. Board Manager: Maintains position state and history
2. Evaluation Agent: Provides position assessments
3. Move Proposer: Generates candidate moves
4. Rule Validator: Ensures move legality
5. Meta-Strategy Director: Coordinates overall strategy

Communication protocol uses structured messages with:
- Message type (request/response/event)
- Sender/receiver agent IDs
- Payload with typed data
- Timestamp and sequence number
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"


@dataclass
class AgentMessage:
    """Message exchanged between agents.

    Attributes:
        msg_type: Type of message.
        sender: ID of sending agent.
        receiver: ID of receiving agent (or "broadcast").
        action: Requested action or event type.
        payload: Message data.
        msg_id: Unique message ID.
        timestamp: Message timestamp.
        correlation_id: ID linking request/response pairs.
    """

    msg_type: MessageType
    sender: str
    receiver: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    correlation_id: str | None = None


@dataclass
class AgentState:
    """State maintained by an agent."""

    agent_id: str
    is_active: bool = True
    messages_sent: int = 0
    messages_received: int = 0
    last_activity: float = field(default_factory=time.time)
    memory: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Base class for all chess agents."""

    def __init__(self, agent_id: str):
        """Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent.
        """
        self.agent_id = agent_id
        self.state = AgentState(agent_id=agent_id)
        self.message_handlers: dict[str, Any] = {}

    def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process an incoming message.

        Args:
            message: Incoming message to process.

        Returns:
            Response message, if any.
        """
        self.state.messages_received += 1
        self.state.last_activity = time.time()

        handler = self.message_handlers.get(message.action)
        if handler:
            return handler(message)

        return None

    def send_message(
        self,
        msg_type: MessageType,
        receiver: str,
        action: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> AgentMessage:
        """Create a message to send.

        Args:
            msg_type: Type of message.
            receiver: Target agent ID.
            action: Action to request/report.
            payload: Message data.
            correlation_id: ID for request/response linking.

        Returns:
            The created message.
        """
        self.state.messages_sent += 1
        return AgentMessage(
            msg_type=msg_type,
            sender=self.agent_id,
            receiver=receiver,
            action=action,
            payload=payload,
            correlation_id=correlation_id,
        )


class BoardManagerAgent(BaseAgent):
    """Agent responsible for board state management."""

    def __init__(self):
        super().__init__("board_manager")
        from qratum_chess.core.position import Position

        self.position: Position | None = None
        self.move_history: list[Any] = []
        self.position_history: list[str] = []

        self.message_handlers = {
            "get_position": self._handle_get_position,
            "set_position": self._handle_set_position,
            "make_move": self._handle_make_move,
            "undo_move": self._handle_undo_move,
            "get_legal_moves": self._handle_get_legal_moves,
            "get_fen": self._handle_get_fen,
        }

    def _handle_get_position(self, msg: AgentMessage) -> AgentMessage:
        """Return current position."""
        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "position_data",
            {"position": self.position, "fen": self.position.to_fen() if self.position else None},
            correlation_id=msg.msg_id,
        )

    def _handle_set_position(self, msg: AgentMessage) -> AgentMessage:
        """Set position from FEN."""
        from qratum_chess.core.position import Position

        fen = msg.payload.get("fen")
        if fen:
            self.position = Position.from_fen(fen)
        else:
            self.position = Position.starting()

        self.move_history = []
        self.position_history = [self.position.to_fen()]

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "position_set",
            {"success": True, "fen": self.position.to_fen()},
            correlation_id=msg.msg_id,
        )

    def _handle_make_move(self, msg: AgentMessage) -> AgentMessage:
        """Execute a move on the board."""
        from qratum_chess.core.position import Move

        if not self.position:
            return self.send_message(
                MessageType.RESPONSE,
                msg.sender,
                "move_result",
                {"success": False, "error": "No position set"},
                correlation_id=msg.msg_id,
            )

        move_data = msg.payload.get("move")
        if isinstance(move_data, str):
            move = Move.from_uci(move_data)
        else:
            move = move_data

        # Validate move
        legal_moves = self.position.generate_legal_moves()
        if move not in legal_moves:
            return self.send_message(
                MessageType.RESPONSE,
                msg.sender,
                "move_result",
                {"success": False, "error": "Illegal move"},
                correlation_id=msg.msg_id,
            )

        # Make move
        self.move_history.append(move)
        self.position = self.position.make_move(move)
        self.position_history.append(self.position.to_fen())

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "move_result",
            {
                "success": True,
                "fen": self.position.to_fen(),
                "is_check": self.position.is_in_check(),
                "is_checkmate": self.position.is_checkmate(),
                "is_stalemate": self.position.is_stalemate(),
                "is_draw": self.position.is_draw(),
            },
            correlation_id=msg.msg_id,
        )

    def _handle_undo_move(self, msg: AgentMessage) -> AgentMessage:
        """Undo the last move."""
        from qratum_chess.core.position import Position

        if len(self.position_history) < 2:
            return self.send_message(
                MessageType.RESPONSE,
                msg.sender,
                "undo_result",
                {"success": False, "error": "No moves to undo"},
                correlation_id=msg.msg_id,
            )

        self.position_history.pop()
        self.move_history.pop()
        self.position = Position.from_fen(self.position_history[-1])

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "undo_result",
            {"success": True, "fen": self.position.to_fen()},
            correlation_id=msg.msg_id,
        )

    def _handle_get_legal_moves(self, msg: AgentMessage) -> AgentMessage:
        """Get all legal moves."""
        if not self.position:
            return self.send_message(
                MessageType.RESPONSE,
                msg.sender,
                "legal_moves",
                {"moves": []},
                correlation_id=msg.msg_id,
            )

        legal_moves = self.position.generate_legal_moves()
        move_strings = [m.to_uci() for m in legal_moves]

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "legal_moves",
            {"moves": move_strings, "count": len(legal_moves)},
            correlation_id=msg.msg_id,
        )

    def _handle_get_fen(self, msg: AgentMessage) -> AgentMessage:
        """Get current FEN string."""
        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "fen_data",
            {"fen": self.position.to_fen() if self.position else None},
            correlation_id=msg.msg_id,
        )


class EvaluationAgent(BaseAgent):
    """Agent responsible for position evaluation."""

    def __init__(self):
        super().__init__("evaluator")
        self.evaluator = None
        self.trimodal_core = None

        self.message_handlers = {
            "evaluate_position": self._handle_evaluate,
            "evaluate_moves": self._handle_evaluate_moves,
            "get_features": self._handle_get_features,
        }

    def _handle_evaluate(self, msg: AgentMessage) -> AgentMessage:
        """Evaluate a position."""
        from qratum_chess.core.position import Position
        from qratum_chess.neural.trimodal import TriModalCore

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        if not position:
            return self.send_message(
                MessageType.RESPONSE,
                msg.sender,
                "evaluation",
                {"error": "No position provided"},
                correlation_id=msg.msg_id,
            )

        # Use Tri-Modal Core if available
        if self.trimodal_core is None:
            self.trimodal_core = TriModalCore()

        legal_moves = position.generate_legal_moves()
        best_move, value, diagnostics = self.trimodal_core.evaluate(position, legal_moves)

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "evaluation",
            {
                "value": value,
                "best_move": best_move.to_uci() if best_move else None,
                "diagnostics": diagnostics,
            },
            correlation_id=msg.msg_id,
        )

    def _handle_evaluate_moves(self, msg: AgentMessage) -> AgentMessage:
        """Evaluate specific moves."""
        from qratum_chess.core.position import Move, Position
        from qratum_chess.neural.trimodal import TriModalCore

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        moves = msg.payload.get("moves", [])
        if isinstance(moves[0] if moves else "", str):
            moves = [Move.from_uci(m) for m in moves]

        if self.trimodal_core is None:
            self.trimodal_core = TriModalCore()

        _, value, diagnostics = self.trimodal_core.evaluate(position, moves)

        # Get individual move scores
        move_scores = {}
        for move in moves:
            new_pos = position.make_move(move)
            _, move_value, _ = self.trimodal_core.evaluate(new_pos, new_pos.generate_legal_moves())
            move_scores[move.to_uci()] = -move_value

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "move_evaluations",
            {"move_scores": move_scores},
            correlation_id=msg.msg_id,
        )

    def _handle_get_features(self, msg: AgentMessage) -> AgentMessage:
        """Extract features from position."""
        from qratum_chess.core.position import Position
        from qratum_chess.neural.encoding import PositionEncoder

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        encoder = PositionEncoder()
        tensor = encoder.encode(position)

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "features",
            {
                "tensor_shape": tensor.shape,
                "num_channels": encoder.num_channels,
            },
            correlation_id=msg.msg_id,
        )


class MoveProposalAgent(BaseAgent):
    """Agent responsible for generating move candidates."""

    def __init__(self):
        super().__init__("move_proposer")
        self.searcher = None

        self.message_handlers = {
            "propose_moves": self._handle_propose,
            "search_position": self._handle_search,
        }

    def _handle_propose(self, msg: AgentMessage) -> AgentMessage:
        """Propose candidate moves for a position."""
        from qratum_chess.core.position import Position

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        num_candidates = msg.payload.get("num_candidates", 5)

        legal_moves = position.generate_legal_moves()

        # Score moves using simple heuristics
        scored_moves = []
        for move in legal_moves:
            score = self._quick_score(position, move)
            scored_moves.append((move, score))

        scored_moves.sort(key=lambda x: x[1], reverse=True)
        candidates = scored_moves[:num_candidates]

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "move_candidates",
            {"candidates": [{"move": m.to_uci(), "score": s} for m, s in candidates]},
            correlation_id=msg.msg_id,
        )

    def _handle_search(self, msg: AgentMessage) -> AgentMessage:
        """Run full search on position."""
        from qratum_chess.core.position import Position
        from qratum_chess.search.aas import AsymmetricAdaptiveSearch

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        depth = msg.payload.get("depth", 10)
        time_limit = msg.payload.get("time_limit_ms")

        if self.searcher is None:
            self.searcher = AsymmetricAdaptiveSearch()

        best_move, value, stats = self.searcher.search(
            position, depth=depth, time_limit_ms=time_limit
        )

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "search_result",
            {
                "best_move": best_move.to_uci() if best_move else None,
                "value": value,
                "stats": {
                    "phase": stats.phase.value,
                    "nodes": stats.nodes_searched,
                    "depth": stats.depth_reached,
                    "time_ms": stats.time_ms,
                },
            },
            correlation_id=msg.msg_id,
        )

    def _quick_score(self, position: Position, move: Move) -> float:
        """Quick heuristic score for move ordering."""
        score = 0.0

        # Capture bonus
        captured = position.board.piece_at(move.to_sq)
        if captured:
            _, cap_type = captured
            capture_values = {0: 1, 1: 3, 2: 3, 3: 5, 4: 9, 5: 0}
            score += capture_values.get(int(cap_type), 0)

        # Center bonus
        to_rank, to_file = move.to_sq // 8, move.to_sq % 8
        if 2 <= to_rank <= 5 and 2 <= to_file <= 5:
            score += 0.5

        # Promotion bonus
        if move.promotion:
            score += 5

        return score


class RuleValidatorAgent(BaseAgent):
    """Agent responsible for rule enforcement and validation."""

    def __init__(self):
        super().__init__("validator")

        self.message_handlers = {
            "validate_move": self._handle_validate_move,
            "check_game_end": self._handle_check_game_end,
            "validate_position": self._handle_validate_position,
        }

    def _handle_validate_move(self, msg: AgentMessage) -> AgentMessage:
        """Validate if a move is legal."""
        from qratum_chess.core.position import Move, Position

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        move = msg.payload.get("move")
        if isinstance(move, str):
            move = Move.from_uci(move)

        legal_moves = position.generate_legal_moves()
        is_legal = move in legal_moves

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "validation_result",
            {
                "is_legal": is_legal,
                "move": move.to_uci(),
            },
            correlation_id=msg.msg_id,
        )

    def _handle_check_game_end(self, msg: AgentMessage) -> AgentMessage:
        """Check if game has ended."""
        from qratum_chess.core.position import Position

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        is_checkmate = position.is_checkmate()
        is_stalemate = position.is_stalemate()
        is_draw = position.is_draw()

        game_over = is_checkmate or is_stalemate or is_draw

        result = None
        if is_checkmate:
            result = "0-1" if position.side_to_move.value == 0 else "1-0"
        elif is_stalemate or is_draw:
            result = "1/2-1/2"

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "game_end_status",
            {
                "game_over": game_over,
                "is_checkmate": is_checkmate,
                "is_stalemate": is_stalemate,
                "is_draw": is_draw,
                "result": result,
            },
            correlation_id=msg.msg_id,
        )

    def _handle_validate_position(self, msg: AgentMessage) -> AgentMessage:
        """Validate a position is legal."""
        from qratum_chess.core import Color, PieceType
        from qratum_chess.core.position import Position

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        errors = []

        # Check for exactly one king per side
        for color in Color:
            king_count = bin(int(position.board.pieces[color, PieceType.KING])).count("1")
            if king_count != 1:
                errors.append(f"{color.name} has {king_count} kings (must be 1)")

        # Check opponent not in check
        opponent = Color(1 - position.side_to_move.value)
        if position.is_in_check(opponent):
            errors.append("Opponent is in check (invalid position)")

        # Check for too many pawns
        for color in Color:
            pawn_count = bin(int(position.board.pieces[color, PieceType.PAWN])).count("1")
            if pawn_count > 8:
                errors.append(f"{color.name} has {pawn_count} pawns (max 8)")

        is_valid = len(errors) == 0

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "position_validation",
            {
                "is_valid": is_valid,
                "errors": errors,
            },
            correlation_id=msg.msg_id,
        )


class MetaStrategyDirector(BaseAgent):
    """Agent responsible for high-level strategy coordination."""

    def __init__(self):
        super().__init__("director")
        self.current_strategy = "balanced"
        self.strategy_history: list[str] = []

        self.message_handlers = {
            "select_strategy": self._handle_select_strategy,
            "get_strategy": self._handle_get_strategy,
            "analyze_position": self._handle_analyze_position,
        }

    def _handle_select_strategy(self, msg: AgentMessage) -> AgentMessage:
        """Select strategy based on position."""
        from qratum_chess.core.position import Position

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        # Analyze position to select strategy
        strategy = self._analyze_and_select(position)

        self.current_strategy = strategy
        self.strategy_history.append(strategy)

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "strategy_selected",
            {
                "strategy": strategy,
                "reasoning": self._get_strategy_reasoning(strategy),
            },
            correlation_id=msg.msg_id,
        )

    def _handle_get_strategy(self, msg: AgentMessage) -> AgentMessage:
        """Get current strategy."""
        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "current_strategy",
            {
                "strategy": self.current_strategy,
                "history": self.strategy_history[-10:],
            },
            correlation_id=msg.msg_id,
        )

    def _handle_analyze_position(self, msg: AgentMessage) -> AgentMessage:
        """Provide strategic analysis of position."""
        from qratum_chess.core.position import Position

        position = msg.payload.get("position")
        if isinstance(position, str):
            position = Position.from_fen(position)

        analysis = self._full_analysis(position)

        return self.send_message(
            MessageType.RESPONSE,
            msg.sender,
            "position_analysis",
            {"analysis": analysis},
            correlation_id=msg.msg_id,
        )

    def _analyze_and_select(self, position: Position) -> str:
        """Analyze position and select appropriate strategy."""
        from qratum_chess.core import Color, PieceType

        # Count material
        material = {Color.WHITE: 0, Color.BLACK: 0}
        piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0,
        }

        for color in Color:
            for pt in PieceType:
                count = bin(int(position.board.pieces[color, pt])).count("1")
                material[color] += count * piece_values[pt]

        our_color = position.side_to_move
        their_color = Color(1 - our_color.value)

        material_diff = material[our_color] - material[their_color]

        # Select strategy based on material and position
        if material_diff > 3:
            return "simplify"  # Trade pieces when ahead
        elif material_diff < -3:
            return "complicate"  # Create chaos when behind
        elif position.is_in_check():
            return "defensive"
        elif position.fullmove_number < 10:
            return "development"
        else:
            return "balanced"

    def _get_strategy_reasoning(self, strategy: str) -> str:
        """Get explanation for strategy selection."""
        reasons = {
            "simplify": "Material advantage - simplify to realize advantage",
            "complicate": "Material disadvantage - create complications",
            "defensive": "Under pressure - prioritize king safety",
            "development": "Opening phase - focus on piece development",
            "balanced": "Neutral position - maintain flexibility",
        }
        return reasons.get(strategy, "Standard play")

    def _full_analysis(self, position: Position) -> dict[str, Any]:
        """Generate full strategic analysis."""
        from qratum_chess.core import Color, PieceType

        analysis = {
            "phase": "middlegame",
            "material_balance": 0,
            "pawn_structure": "normal",
            "king_safety": {"white": "safe", "black": "safe"},
            "piece_activity": {"white": 0, "black": 0},
            "recommendations": [],
        }

        # Material balance
        piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0,
        }

        for color in Color:
            sign = 1 if color == Color.WHITE else -1
            for pt in PieceType:
                count = bin(int(position.board.pieces[color, pt])).count("1")
                analysis["material_balance"] += sign * count * piece_values[pt]

        # Game phase
        total_material = sum(
            bin(int(position.board.pieces[c, pt])).count("1") * piece_values[pt]
            for c in Color
            for pt in PieceType
        )

        if position.fullmove_number < 10:
            analysis["phase"] = "opening"
        elif total_material < 20:
            analysis["phase"] = "endgame"

        # Recommendations based on analysis
        if analysis["material_balance"] > 0:
            analysis["recommendations"].append("Consider simplifying the position")
        elif analysis["material_balance"] < 0:
            analysis["recommendations"].append("Look for tactical complications")

        if analysis["phase"] == "opening":
            analysis["recommendations"].append("Focus on piece development")
        elif analysis["phase"] == "endgame":
            analysis["recommendations"].append("Activate your king")

        return analysis


class AgentOrchestrator:
    """Orchestrates communication between chess agents.

    Manages the multi-agent pipeline for chess engine decision making.
    Routes messages between agents and coordinates the overall workflow.
    """

    def __init__(self):
        """Initialize the orchestrator."""
        # Create all agents
        self.board_manager = BoardManagerAgent()
        self.evaluator = EvaluationAgent()
        self.move_proposer = MoveProposalAgent()
        self.validator = RuleValidatorAgent()
        self.director = MetaStrategyDirector()

        # Agent registry
        self.agents: dict[str, BaseAgent] = {
            "board_manager": self.board_manager,
            "evaluator": self.evaluator,
            "move_proposer": self.move_proposer,
            "validator": self.validator,
            "director": self.director,
        }

        # Message log
        self.message_log: list[AgentMessage] = []

    def route_message(self, message: AgentMessage) -> AgentMessage | None:
        """Route a message to the appropriate agent.

        Args:
            message: Message to route.

        Returns:
            Response message, if any.
        """
        self.message_log.append(message)

        if message.receiver == "broadcast":
            # Send to all agents
            responses = []
            for agent in self.agents.values():
                response = agent.handle_message(message)
                if response:
                    responses.append(response)
                    self.message_log.append(response)
            return responses[0] if responses else None

        agent = self.agents.get(message.receiver)
        if agent:
            response = agent.handle_message(message)
            if response:
                self.message_log.append(response)
            return response

        return None

    def get_best_move(
        self, fen: str | None = None, depth: int = 10, time_limit_ms: float | None = None
    ) -> dict[str, Any]:
        """Get best move using full agent pipeline.

        Args:
            fen: Position in FEN format (uses current position if None).
            depth: Search depth.
            time_limit_ms: Time limit in milliseconds.

        Returns:
            Dictionary with best move and analysis.
        """
        # Set position if provided
        if fen:
            self.route_message(
                AgentMessage(
                    msg_type=MessageType.REQUEST,
                    sender="orchestrator",
                    receiver="board_manager",
                    action="set_position",
                    payload={"fen": fen},
                )
            )

        # Get current position
        pos_response = self.route_message(
            AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="orchestrator",
                receiver="board_manager",
                action="get_position",
                payload={},
            )
        )

        position = pos_response.payload.get("position") if pos_response else None
        if not position:
            return {"error": "No position set"}

        # Get strategy recommendation
        strategy_response = self.route_message(
            AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="orchestrator",
                receiver="director",
                action="select_strategy",
                payload={"position": position},
            )
        )

        strategy = strategy_response.payload if strategy_response else {}

        # Search for best move
        search_response = self.route_message(
            AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="orchestrator",
                receiver="move_proposer",
                action="search_position",
                payload={
                    "position": position,
                    "depth": depth,
                    "time_limit_ms": time_limit_ms,
                },
            )
        )

        search_result = search_response.payload if search_response else {}

        # Validate best move
        if search_result.get("best_move"):
            validation_response = self.route_message(
                AgentMessage(
                    msg_type=MessageType.REQUEST,
                    sender="orchestrator",
                    receiver="validator",
                    action="validate_move",
                    payload={
                        "position": position,
                        "move": search_result["best_move"],
                    },
                )
            )
            is_valid = validation_response.payload.get("is_legal") if validation_response else False
        else:
            is_valid = False

        # Get evaluation
        eval_response = self.route_message(
            AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="orchestrator",
                receiver="evaluator",
                action="evaluate_position",
                payload={"position": position},
            )
        )

        evaluation = eval_response.payload if eval_response else {}

        return {
            "best_move": search_result.get("best_move"),
            "value": search_result.get("value", 0),
            "is_valid": is_valid,
            "strategy": strategy.get("strategy"),
            "strategy_reasoning": strategy.get("reasoning"),
            "search_stats": search_result.get("stats", {}),
            "evaluation_diagnostics": evaluation.get("diagnostics", {}),
        }

    def new_game(self, fen: str | None = None) -> dict[str, Any]:
        """Start a new game.

        Args:
            fen: Starting position (uses standard if None).

        Returns:
            Game initialization status.
        """
        response = self.route_message(
            AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="orchestrator",
                receiver="board_manager",
                action="set_position",
                payload={"fen": fen} if fen else {},
            )
        )

        return response.payload if response else {"error": "Failed to start game"}

    def make_move(self, move: str) -> dict[str, Any]:
        """Make a move in the current game.

        Args:
            move: Move in UCI format.

        Returns:
            Move result.
        """
        response = self.route_message(
            AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="orchestrator",
                receiver="board_manager",
                action="make_move",
                payload={"move": move},
            )
        )

        return response.payload if response else {"error": "Failed to make move"}

    def get_analysis(self) -> dict[str, Any]:
        """Get full analysis of current position.

        Returns:
            Position analysis.
        """
        # Get position
        pos_response = self.route_message(
            AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="orchestrator",
                receiver="board_manager",
                action="get_position",
                payload={},
            )
        )

        position = pos_response.payload.get("position") if pos_response else None
        if not position:
            return {"error": "No position set"}

        # Get strategic analysis
        analysis_response = self.route_message(
            AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="orchestrator",
                receiver="director",
                action="analyze_position",
                payload={"position": position},
            )
        )

        return analysis_response.payload if analysis_response else {"error": "Analysis failed"}
