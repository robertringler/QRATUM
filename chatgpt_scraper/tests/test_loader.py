import zipfile
from pathlib import Path

from chatgpt_scraper.loader import find_export_root, iter_raw_messages, load_raw_conversations

FIXTURE_DIR = Path(__file__).parent / "data" / "export_dir"


def _make_export_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "export.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        for file_path in FIXTURE_DIR.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(FIXTURE_DIR))
    return archive_path


def test_find_export_root_directory():
    root = find_export_root(FIXTURE_DIR)
    assert root == FIXTURE_DIR


def test_find_export_root_zip(tmp_path):
    zip_path = _make_export_zip(tmp_path)
    root = find_export_root(zip_path)
    assert (root / "conversations.json").exists()


def test_load_conversations_and_iter_messages():
    conversations = load_raw_conversations(FIXTURE_DIR)
    assert len(conversations) == 2
    first_messages = list(iter_raw_messages(conversations[0]))
    assert len(first_messages) == 3
    second_messages = list(iter_raw_messages(conversations[1]))
    assert len(second_messages) == 2
