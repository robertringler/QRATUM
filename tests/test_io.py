import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor.io import ConversationLoader, LedgerWriter, SummaryWriter
from extractor.models import ClassificationLabel, SentenceRecord


def test_conversation_loader_reads_messages(tmp_path: Path) -> None:
    payload = {
        "conversations": [
            {
                "conversation_id": "conv-test",
                "messages": [
                    {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Measured data"]},
                    }
                ],
            }
        ]
    }
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loader = ConversationLoader(path)
    messages = loader.load()
    assert len(messages) == 1
    assert messages[0].content == "Measured data"


def make_record(source_path: Path) -> SentenceRecord:
    return SentenceRecord(
        statement_id="123",
        conversation_id="conv-test",
        message_index=0,
        sentence_index=0,
        text="Measured data",
        classification=ClassificationLabel.EMPIRICAL_CONFIRMED,
        domains=["physics"],
        source_path=source_path,
    )


def test_ledger_writer_outputs_jsonl(tmp_path: Path) -> None:
    records = [make_record(tmp_path / "conversations.json")]
    writer = LedgerWriter(tmp_path)
    path = writer.write(records)
    content = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    data = json.loads(content[0])
    assert data["artifact_path"] == str(path)
    assert data["classification"] == "EMPIRICAL_CONFIRMED"


def test_summary_writer_outputs_csv(tmp_path: Path) -> None:
    records = [make_record(tmp_path / "conversations.json")]
    writer = SummaryWriter(tmp_path)
    path = writer.write(records)
    csv_lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0] == "classification,count,top_domains"
    assert any("EMPIRICAL_CONFIRMED,1" in line for line in csv_lines[1:])
