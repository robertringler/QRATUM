"""UCI (Universal Chess Interface) protocol implementation.

Provides a complete UCI interface for QRATUM-Chess engine,
enabling integration with chess GUIs like Arena, ChessBase, and Cute Chess.

Supported commands:
- uci: Initialize UCI mode
- isready: Check if engine is ready
- ucinewgame: Start new game
- position: Set position (startpos or fen)
- go: Start search
- stop: Stop search
- quit: Exit engine
- setoption: Configure engine options
"""

from __future__ import annotations

import sys
import threading
from typing import TextIO


class UCIEngine:
    """UCI protocol handler for QRATUM-Chess.

    Implements the full UCI protocol specification for communication
    with chess GUIs.
    """

    ENGINE_NAME = "QRATUM-Chess"
    ENGINE_AUTHOR = "QRATUM Team"

    # Engine options
    DEFAULT_OPTIONS = {
        "Hash": {"type": "spin", "default": 256, "min": 1, "max": 4096},
        "Threads": {"type": "spin", "default": 1, "min": 1, "max": 64},
        "MultiPV": {"type": "spin", "default": 1, "min": 1, "max": 10},
        "Ponder": {"type": "check", "default": False},
        "UCI_Chess960": {"type": "check", "default": False},
        "TriModalCore": {"type": "check", "default": True},
        "SearchMode": {
            "type": "combo",
            "default": "adaptive",
            "options": ["adaptive", "alphabeta", "mcts"],
        },
    }

    def __init__(self, input_stream: TextIO | None = None, output_stream: TextIO | None = None):
        """Initialize the UCI engine.

        Args:
            input_stream: Input stream (defaults to stdin).
            output_stream: Output stream (defaults to stdout).
        """
        self.input = input_stream or sys.stdin
        self.output = output_stream or sys.stdout

        # Engine state
        self.position = None
        self.searcher = None
        self.search_thread: threading.Thread | None = None
        self.stop_flag = threading.Event()

        # Options
        self.options = {k: v["default"] for k, v in self.DEFAULT_OPTIONS.items()}

        # Initialize engine components
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize engine components."""
        from qratum_chess.core.position import Position
        from qratum_chess.search.aas import AsymmetricAdaptiveSearch

        self.position = Position.starting()
        self.searcher = AsymmetricAdaptiveSearch()

    def send(self, message: str) -> None:
        """Send a message to the GUI.

        Args:
            message: Message to send.
        """
        print(message, file=self.output, flush=True)

    def run(self) -> None:
        """Run the UCI protocol loop."""
        while True:
            try:
                line = self.input.readline()
                if not line:
                    break

                command = line.strip()
                if not command:
                    continue

                if command == "quit":
                    self._cmd_quit()
                    break

                self._process_command(command)

            except KeyboardInterrupt:
                break

    def _process_command(self, command: str) -> None:
        """Process a UCI command.

        Args:
            command: Command string to process.
        """
        parts = command.split()
        if not parts:
            return

        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        handlers = {
            "uci": self._cmd_uci,
            "isready": self._cmd_isready,
            "ucinewgame": self._cmd_ucinewgame,
            "position": lambda: self._cmd_position(args),
            "go": lambda: self._cmd_go(args),
            "stop": self._cmd_stop,
            "setoption": lambda: self._cmd_setoption(args),
            "d": self._cmd_debug,  # Debug: display board
            "eval": self._cmd_eval,  # Debug: show evaluation
        }

        handler = handlers.get(cmd)
        if handler:
            handler()

    def _cmd_uci(self) -> None:
        """Handle 'uci' command."""
        self.send(f"id name {self.ENGINE_NAME}")
        self.send(f"id author {self.ENGINE_AUTHOR}")

        # Send options
        for name, spec in self.DEFAULT_OPTIONS.items():
            opt_str = f"option name {name} type {spec['type']}"
            if spec["type"] == "spin":
                opt_str += f" default {spec['default']} min {spec['min']} max {spec['max']}"
            elif spec["type"] == "check":
                opt_str += f" default {'true' if spec['default'] else 'false'}"
            elif spec["type"] == "combo":
                opt_str += f" default {spec['default']}"
                for opt in spec["options"]:
                    opt_str += f" var {opt}"
            self.send(opt_str)

        self.send("uciok")

    def _cmd_isready(self) -> None:
        """Handle 'isready' command."""
        self.send("readyok")

    def _cmd_ucinewgame(self) -> None:
        """Handle 'ucinewgame' command."""
        from qratum_chess.core.position import Position

        self.position = Position.starting()
        if self.searcher:
            self.searcher.alphabeta.clear_tables()

    def _cmd_position(self, args: list[str]) -> None:
        """Handle 'position' command.

        Args:
            args: Command arguments.
        """
        from qratum_chess.core.position import Move, Position

        if not args:
            return

        # Parse position
        if args[0] == "startpos":
            self.position = Position.starting()
            move_idx = 1
        elif args[0] == "fen":
            # Find 'moves' keyword or end of FEN
            fen_end = len(args)
            for i, arg in enumerate(args[1:], 1):
                if arg == "moves":
                    fen_end = i
                    break

            fen = " ".join(args[1:fen_end])
            self.position = Position.from_fen(fen)
            move_idx = fen_end
        else:
            return

        # Apply moves if present
        if move_idx < len(args) and args[move_idx] == "moves":
            for move_str in args[move_idx + 1 :]:
                move = Move.from_uci(move_str)
                self.position = self.position.make_move(move)

    def _cmd_go(self, args: list[str]) -> None:
        """Handle 'go' command.

        Args:
            args: Command arguments.
        """
        # Parse search parameters
        params = self._parse_go_params(args)

        # Start search in background thread
        self.stop_flag.clear()
        self.search_thread = threading.Thread(target=self._search_thread, args=(params,))
        self.search_thread.start()

    def _parse_go_params(self, args: list[str]) -> dict:
        """Parse 'go' command parameters.

        Args:
            args: Command arguments.

        Returns:
            Dictionary of search parameters.
        """
        params = {
            "depth": None,
            "nodes": None,
            "movetime": None,
            "wtime": None,
            "btime": None,
            "winc": None,
            "binc": None,
            "movestogo": None,
            "infinite": False,
            "ponder": False,
        }

        i = 0
        while i < len(args):
            if args[i] == "depth" and i + 1 < len(args):
                params["depth"] = int(args[i + 1])
                i += 2
            elif args[i] == "nodes" and i + 1 < len(args):
                params["nodes"] = int(args[i + 1])
                i += 2
            elif args[i] == "movetime" and i + 1 < len(args):
                params["movetime"] = int(args[i + 1])
                i += 2
            elif args[i] == "wtime" and i + 1 < len(args):
                params["wtime"] = int(args[i + 1])
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                params["btime"] = int(args[i + 1])
                i += 2
            elif args[i] == "winc" and i + 1 < len(args):
                params["winc"] = int(args[i + 1])
                i += 2
            elif args[i] == "binc" and i + 1 < len(args):
                params["binc"] = int(args[i + 1])
                i += 2
            elif args[i] == "movestogo" and i + 1 < len(args):
                params["movestogo"] = int(args[i + 1])
                i += 2
            elif args[i] == "infinite":
                params["infinite"] = True
                i += 1
            elif args[i] == "ponder":
                params["ponder"] = True
                i += 1
            else:
                i += 1

        return params

    def _search_thread(self, params: dict) -> None:
        """Execute search in background thread.

        Args:
            params: Search parameters.
        """
        from qratum_chess.core import Color

        # Determine time limit
        time_limit_ms = None

        if params["movetime"]:
            time_limit_ms = params["movetime"]
        elif params["wtime"] or params["btime"]:
            # Simple time management
            our_time = (
                params["wtime"] if self.position.side_to_move == Color.WHITE else params["btime"]
            )
            if our_time:
                moves_left = params["movestogo"] or 40
                time_limit_ms = our_time / moves_left

        depth = params["depth"] or 20

        # Run search
        try:
            best_move, value, stats = self.searcher.search(
                self.position, depth=depth, time_limit_ms=time_limit_ms
            )

            if not self.stop_flag.is_set() and best_move:
                # Send info about search
                self.send(
                    f"info depth {stats.depth_reached} "
                    f"nodes {stats.nodes_searched} "
                    f"time {int(stats.time_ms)} "
                    f"score cp {int(value * 100)}"
                )

                # Send best move
                self.send(f"bestmove {best_move.to_uci()}")

        except Exception:
            # On error, try to return a legal move
            legal_moves = self.position.generate_legal_moves()
            if legal_moves:
                self.send(f"bestmove {legal_moves[0].to_uci()}")

    def _cmd_stop(self) -> None:
        """Handle 'stop' command."""
        self.stop_flag.set()
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=1.0)

    def _cmd_setoption(self, args: list[str]) -> None:
        """Handle 'setoption' command.

        Args:
            args: Command arguments.
        """
        if len(args) < 4:
            return

        # Parse: name <name> value <value>
        if args[0] != "name":
            return

        # Find 'value' keyword
        value_idx = -1
        for i, arg in enumerate(args):
            if arg == "value":
                value_idx = i
                break

        if value_idx < 0:
            return

        name = " ".join(args[1:value_idx])
        value = " ".join(args[value_idx + 1 :])

        # Set option
        if name in self.options:
            spec = self.DEFAULT_OPTIONS[name]
            if spec["type"] == "spin":
                self.options[name] = int(value)
            elif spec["type"] == "check":
                self.options[name] = value.lower() == "true"
            else:
                self.options[name] = value

    def _cmd_quit(self) -> None:
        """Handle 'quit' command."""
        self.stop_flag.set()
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=1.0)

    def _cmd_debug(self) -> None:
        """Handle 'd' (debug) command - display current board."""
        if self.position:
            self.send(str(self.position))

    def _cmd_eval(self) -> None:
        """Handle 'eval' command - show position evaluation."""
        from qratum_chess.neural.trimodal import TriModalCore

        if not self.position:
            return

        core = TriModalCore()
        legal_moves = self.position.generate_legal_moves()

        if legal_moves:
            best_move, value, diagnostics = core.evaluate(self.position, legal_moves)
            self.send(f"Evaluation: {value:.4f}")
            self.send(f"Best move: {best_move.to_uci() if best_move else 'none'}")
            self.send(f"Tactical confidence: {diagnostics.get('tactical_confidence', 0):.2f}")
            self.send(f"Strategic confidence: {diagnostics.get('strategic_confidence', 0):.2f}")


def main():
    """Entry point for UCI engine."""
    engine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
