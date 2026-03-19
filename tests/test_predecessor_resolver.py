"""Regression tests for shared travel predecessor-resolution helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from source.task_handlers.travel import predecessor_resolver as resolver


def test_resolve_generation_id_prefers_first_truthy_source():
    assert resolver.resolve_generation_id(
        "child_generation_id",
        {"child_generation_id": ""},
        {"child_generation_id": "child-2"},
        {"child_generation_id": "child-3"},
    ) == "child-2"
    assert resolver.resolve_generation_id(
        "parent_generation_id",
        None,
        {"parent_generation_id": "parent-1"},
    ) == "parent-1"
    assert resolver.resolve_generation_id("missing_key", {"missing_key": None}, {}) is None


def test_predecessor_result_found_contract():
    assert resolver.PredecessorResult(task_id="pred-1", output_url="/tmp/out.mp4").found is True
    assert resolver.PredecessorResult(task_id="pred-1", output_url=None).found is False
    assert resolver.PredecessorResult(task_id=None, output_url="/tmp/out.mp4").found is False


def test_resolve_segment_predecessor_wraps_db_lookup():
    with patch.object(
        resolver,
        "get_segment_predecessor_output",
        return_value=("pred-42", "https://example.com/output.mp4"),
    ) as get_predecessor:
        result = resolver.resolve_segment_predecessor(
            task_id="task-1",
            parent_generation_id="parent-1",
            child_generation_id="child-1",
            child_order=3,
            segment_index=4,
        )

    get_predecessor.assert_called_once_with(
        task_id="task-1",
        parent_generation_id="parent-1",
        child_generation_id="child-1",
        child_order=3,
        segment_index=4,
    )
    assert result == resolver.PredecessorResult(
        task_id="pred-42",
        output_url="https://example.com/output.mp4",
    )
    assert result.found is True


def test_download_predecessor_video_redownloads_empty_cached_file(tmp_path):
    cached_file = tmp_path / "prev_remote.mp4"
    cached_file.write_bytes(b"")

    def fake_download(url: str, dest_folder: Path, filename: str) -> bool:
        assert url == "https://example.com/remote.mp4"
        assert Path(dest_folder) == tmp_path
        (Path(dest_folder) / filename).write_bytes(b"video-bytes")
        return True

    with patch.object(resolver, "download_file", side_effect=fake_download) as download:
        resolved = resolver.download_predecessor_video(
            "https://example.com/remote.mp4",
            tmp_path,
            prefix="prev",
        )

    download.assert_called_once()
    assert resolved == str(cached_file.resolve())
    assert cached_file.read_bytes() == b"video-bytes"
