from qconstitution.charter import default_charter


def test_default_charter_active():
    charter = default_charter()
    assert charter.active.version_id == "v1"
    assert "safety-envelope" in charter.article_index()
