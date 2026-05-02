from pathlib import Path

from chatgpt_scraper.loader import load_raw_conversations
from chatgpt_scraper.normalize import (build_ledger, normalize_conversation,
                                       summarize_conversation)

FIXTURE_DIR = Path(__file__).parent / "data" / "export_dir"


def test_normalize_conversation_roles_and_content():
    conversations = load_raw_conversations(FIXTURE_DIR)
    turns = normalize_conversation(conversations[0])
    assert [turn.role for turn in turns] == ["system", "user", "assistant"]
    assert "System priming" in turns[0].content
    assert "Hello assistant" in turns[1].content


def test_build_ledger_and_summary_counts():
    conversations = load_raw_conversations(FIXTURE_DIR)
    turns, summaries = build_ledger(conversations)
    assert len(turns) == 5
    assert len(summaries) == 2
    conv_id = conversations[1].id
    summary = summarize_conversation(
        conv_id, [t for t in turns if t.conversation_id == conv_id], "List based conversation"
    )
    assert summary.num_messages == 2
    assert "gpt-3.5" in summary.models_used
