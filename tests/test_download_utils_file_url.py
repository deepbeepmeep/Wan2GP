from pathlib import Path

import pytest

from source.utils.download_utils import DownloadError, download_image_if_url, download_video_if_url


@pytest.fixture()
def materialization_dir(tmp_path, monkeypatch):
    mat_dir = tmp_path / "mat"
    mat_dir.mkdir()
    monkeypatch.setenv("REIGH_LOCAL_WORKER_DIR", str(mat_dir))
    return mat_dir


def test_download_image_accepts_file_url_inside_materialization_dir(materialization_dir, tmp_path):
    source_file = materialization_dir / "input.png"
    source_file.write_bytes(b"image")
    target_dir = tmp_path / "downloads"

    result = download_image_if_url(source_file.as_uri(), target_dir)

    assert result == str(source_file.resolve())
    assert not target_dir.exists()


def test_download_video_accepts_file_url_inside_materialization_dir(materialization_dir, tmp_path):
    source_file = materialization_dir / "input.mp4"
    source_file.write_bytes(b"video")
    target_dir = tmp_path / "downloads"

    result = download_video_if_url(source_file.as_uri(), target_dir)

    assert result == str(source_file.resolve())
    assert not target_dir.exists()


def test_rejects_file_url_outside_materialization_dir(materialization_dir, tmp_path):
    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"outside")

    with pytest.raises(DownloadError, match="file:// path not allowed"):
        download_image_if_url(outside_file.as_uri(), tmp_path / "downloads")


def test_rejects_file_url_with_literal_dotdot_component(materialization_dir, tmp_path):
    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"outside")
    dotdot_url = (materialization_dir / ".." / outside_file.name).as_uri()

    with pytest.raises(DownloadError, match="file:// path not allowed"):
        download_image_if_url(dotdot_url, tmp_path / "downloads")


def test_rejects_file_url_with_traversal_escape(materialization_dir):
    with pytest.raises(DownloadError, match="file:// path not allowed"):
        download_image_if_url("file:///etc/passwd", None)


def test_rejects_missing_file_url(materialization_dir):
    missing_file = materialization_dir / "missing.png"

    with pytest.raises(DownloadError, match="file:// path not allowed"):
        download_image_if_url(missing_file.as_uri(), None)


def test_rejects_non_file_url(materialization_dir):
    with pytest.raises(DownloadError, match="file:// path not allowed"):
        download_image_if_url(materialization_dir.as_uri(), None)


def test_file_url_works_when_download_target_dir_is_none(materialization_dir):
    source_file = materialization_dir / "input.png"
    source_file.write_bytes(b"image")

    result = download_image_if_url(source_file.as_uri(), None)

    assert result == str(source_file.resolve())
