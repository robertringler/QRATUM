"""Sentence classification logic for the Empirical Evidence Extractor."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import spacy
except Exception as error:  # pragma: no cover - exercised indirectly via fallback
    spacy = None  # type: ignore[assignment]
    _SPACY_IMPORT_ERROR = error
else:
    _SPACY_IMPORT_ERROR = None

from .domains import DomainTagger
from .models import ClassificationLabel, MessageRecord, SentenceRecord

LOGGER = logging.getLogger(__name__)

CONFIRMED_KEYWORDS = (
    "measured",
    "observed",
    "verified",
    "standard",
    "iso",
    "experiment",
    "protocol",
    "dataset",
    "statistically",
    "evidence",
)

PLAUSIBLE_KEYWORDS = (
    "can",
    "able to",
    "typically",
    "commonly",
    "often",
    "design",
    "implementation",
    "calibrated",
)

SPECULATIVE_KEYWORDS = (
    "might",
    "could",
    "future",
    "plan",
    "potential",
    "hypothetical",
    "aim",
    "will",
)

META_KEYWORDS = (
    "thank you",
    "please",
    "i'm",
    "i am",
    "glad",
    "sorry",
    "hello",
    "instructions",
    "guidelines",
    "persona",
)

MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(hz|khz|ghz|ms|s|kg|g|mg|ma|a|v|mv|kv|m|mm|cm|km|nm|%|percent)\b"
)


@dataclass
class SentenceSegmenter:
    """Segments text into sentences using spaCy with a deterministic fallback."""

    model: str = "en"
    _nlp: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if spacy is None:
            LOGGER.warning(
                "spaCy is unavailable (%s). Using regex-based sentence segmentation fallback.",
                _SPACY_IMPORT_ERROR,
            )
            self._nlp = None
            return

        LOGGER.info("Initializing spaCy model '%s' for segmentation", self.model)
        self._nlp = spacy.blank(self.model)
        if "sentencizer" not in self._nlp.pipe_names:
            self._nlp.add_pipe("sentencizer")

    def segment(self, text: str) -> list[str]:
        if not text.strip():
            return []
        if self._nlp is None:
            return self._fallback_segment(text)
        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _fallback_segment(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [sentence for sentence in sentences if sentence]


@dataclass
class SentenceClassifier:
    """Classifies sentences and attaches domain metadata."""

    domain_tagger: DomainTagger = field(default_factory=DomainTagger)
    segmenter: SentenceSegmenter = field(default_factory=SentenceSegmenter)

    def classify_messages(self, messages: Iterable[MessageRecord], source_path: Path) -> list[SentenceRecord]:
        records: list[SentenceRecord] = []
        for message in messages:
            sentences = self.segmenter.segment(message.content)
            for sentence_index, sentence in enumerate(sentences):
                label = self.classify_sentence(sentence, message)
                domains = self.domain_tagger.tag(sentence)
                record = SentenceRecord(
                    statement_id=str(uuid.uuid4()),
                    conversation_id=message.conversation_id,
                    message_index=message.message_index,
                    sentence_index=sentence_index,
                    text=sentence,
                    classification=label,
                    domains=domains,
                    source_path=source_path,
                )
                records.append(record)
        LOGGER.info("Classified %d sentences", len(records))
        return records

    def classify_sentence(self, sentence: str, message: MessageRecord) -> ClassificationLabel:
        text = sentence.lower()
        if _contains_any(text, META_KEYWORDS) or message.role not in {"assistant", "system"}:
            return ClassificationLabel.NARRATIVE_META
        if _contains_any(text, SPECULATIVE_KEYWORDS):
            return ClassificationLabel.SPECULATIVE
        if MEASUREMENT_PATTERN.search(text) or _contains_any(text, CONFIRMED_KEYWORDS):
            return ClassificationLabel.EMPIRICAL_CONFIRMED
        if _contains_any(text, PLAUSIBLE_KEYWORDS):
            return ClassificationLabel.EMPIRICAL_PLAUSIBLE
        if len(text.split()) >= 5:
            return ClassificationLabel.EMPIRICAL_PLAUSIBLE
        return ClassificationLabel.UNCLASSIFIABLE


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)

