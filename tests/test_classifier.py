import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor.classifier import SentenceClassifier, SentenceSegmenter
from extractor.models import ClassificationLabel, MessageRecord


def make_message(content: str, role: str = "assistant") -> MessageRecord:
    return MessageRecord(conversation_id="conv", message_index=0, role=role, content=content)


def test_classifier_identifies_meta_from_user_role() -> None:
    classifier = SentenceClassifier()
    label = classifier.classify_sentence("Thanks!", make_message("", role="user"))
    assert label == ClassificationLabel.NARRATIVE_META


def test_classifier_identifies_measurements_as_confirmed() -> None:
    classifier = SentenceClassifier()
    label = classifier.classify_sentence("The circuit draws 3.2 mA at 5V.", make_message(""))
    assert label == ClassificationLabel.EMPIRICAL_CONFIRMED


def test_classify_messages_generates_records(tmp_path: Path) -> None:
    classifier = SentenceClassifier()
    message = make_message("The sensor is calibrated. It might fail in vacuum.")
    records = classifier.classify_messages([message], tmp_path / "conversations.json")
    assert len(records) == 2
    assert records[0].classification == ClassificationLabel.EMPIRICAL_PLAUSIBLE
    assert records[1].classification == ClassificationLabel.SPECULATIVE


def test_segmenter_fallback_segment() -> None:
    segmenter = SentenceSegmenter()
    segmenter._nlp = None  # force fallback path
    sentences = segmenter.segment("One. Two? Three!")
    assert sentences == ["One.", "Two?", "Three!"]
