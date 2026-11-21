from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_app_startup():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception


def test_keyword_association_counts():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app import keyword_association_counts

    results = [
        {"keywords": ["transformer model", "attention"]},
        {"keywords": ["transformer model", "bert"]},
    ]
    assert keyword_association_counts(results, "transformer") == {"transformer model": 2}


def test_search_save_and_saved_papers():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                            default_timeout=30).run()
    app.radio[0].set_value("⌕  Research Search").run()
    app.text_input[0].set_value("deep learning").run()
    assert not app.exception
    assert app.button
    app.button[0].click().run()
    assert not app.exception
    app.radio[0].set_value("★  Saved Papers (1)").run()
    assert not app.exception


def test_topic_keyword_and_analytics_pages():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                            default_timeout=30).run()
    app.radio[0].set_value("◈  Topic Explorer").run()
    app.selectbox[0].set_value(app.selectbox[0].options[3]).run()
    assert not app.exception

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                            default_timeout=30).run()
    app.radio[0].set_value("✦  Keyword Explorer").run()
    app.text_input[0].set_value("transformer").run()
    assert not app.exception

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                            default_timeout=30).run()
    app.radio[0].set_value("▦  Research Analytics").run()
    assert not app.exception


def test_artifact_signature_tracks_file_changes(tmp_path):
    import sys
    import os

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import app

    first = app.artifact_signature()
    metadata = app.ARTIFACTS_DIR / "metadata.json"
    stat = metadata.stat()
    os.utime(metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = app.artifact_signature()
    os.utime(metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert first != second
