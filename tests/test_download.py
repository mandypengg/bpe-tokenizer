"""
The caching downloader. Everything here runs against `file://` urls, so the
tests never touch the network.

The cases that matter are the failure ones: a hash mismatch has to be caught
at download time rather than surfacing later as a wrong token id, and an
interrupted fetch must not leave anything behind that a later run would treat
as cached.
"""

from __future__ import annotations

import urllib.request

import pytest

from bpe.download import download_file, sha256

PAYLOAD = b"the quick brown fox\n"
PAYLOAD_SHA = "6b93cffb6d9c5d0d1a2a1b5b8c3a1b1e9b5b8d0d2fbb1b3a9c9c2f9a1e0e8b5a"


@pytest.fixture
def source(tmp_path):
    """A file:// url serving PAYLOAD."""
    path = tmp_path / "source.bin"
    path.write_bytes(PAYLOAD)
    return path.as_uri()


@pytest.fixture
def digest(tmp_path):
    path = tmp_path / "digest.bin"
    path.write_bytes(PAYLOAD)
    return sha256(path)


def test_downloads_when_missing(source, tmp_path):
    target = download_file(source, tmp_path / "out" / "file.bin")
    assert target.read_bytes() == PAYLOAD


def test_creates_missing_parent_directories(source, tmp_path):
    target = download_file(source, tmp_path / "a" / "b" / "c" / "file.bin")
    assert target.exists()


def test_cached_file_is_not_refetched(source, tmp_path):
    target = tmp_path / "file.bin"
    download_file(source, target)
    target.write_bytes(b"local edit")
    # no hash to check against, so the cached bytes are taken as-is
    assert download_file(source, target).read_bytes() == b"local edit"


def test_matching_hash_is_accepted(source, tmp_path, digest):
    target = download_file(source, tmp_path / "file.bin", expected_sha256=digest)
    assert target.read_bytes() == PAYLOAD


def test_wrong_hash_is_rejected(source, tmp_path):
    with pytest.raises(ValueError, match="expected"):
        download_file(source, tmp_path / "file.bin", expected_sha256=PAYLOAD_SHA)


def test_failed_download_leaves_no_file_behind(source, tmp_path):
    with pytest.raises(ValueError):
        download_file(source, tmp_path / "file.bin", expected_sha256=PAYLOAD_SHA)
    # not even a stray .part file
    assert list(tmp_path.iterdir()) == [tmp_path / "source.bin"]


def test_corrupt_cache_is_refetched(source, tmp_path, digest):
    target = tmp_path / "file.bin"
    target.write_bytes(b"truncated")
    assert download_file(source, target, expected_sha256=digest).read_bytes() == PAYLOAD


def test_interrupted_download_leaves_no_partial_file(source, tmp_path, monkeypatch):
    real_urlopen = urllib.request.urlopen

    def explode(*args, **kwargs):
        real_urlopen(*args, **kwargs).close()
        raise TimeoutError("connection dropped")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    with pytest.raises(TimeoutError):
        download_file(source, tmp_path / "file.bin")
    assert not (tmp_path / "file.bin").exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "source.bin"]


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "file.bin"
    path.write_bytes(PAYLOAD * 100)
    assert sha256(path) == hashlib.sha256(PAYLOAD * 100).hexdigest()
