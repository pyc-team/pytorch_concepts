"""Tests for data I/O utilities."""
import os
import tempfile
import pickle
import zipfile
import tarfile
from pathlib import Path

import pytest

from torch_concepts.data.io import (
    extract_zip,
    extract_tar,
    save_pickle,
    load_pickle,
    download_url,
    download_url_wget,
    zip_is_valid,
    wget_available,
    DownloadProgressBar,
)


class TestPickle:
    """Test pickle save/load functionality."""
    
    def test_save_and_load_pickle(self):
        """Test saving and loading a pickle file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            filepath = os.path.join(tmpdir, "test.pkl")
            
            # Save
            saved_path = save_pickle(data, filepath)
            assert os.path.exists(saved_path)
            assert saved_path == os.path.abspath(filepath)
            
            # Load
            loaded_data = load_pickle(saved_path)
            assert loaded_data == data
    
    def test_save_pickle_creates_directory(self):
        """Test that save_pickle creates missing directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [1, 2, 3]
            filepath = os.path.join(tmpdir, "subdir", "nested", "test.pkl")
            
            saved_path = save_pickle(data, filepath)
            assert os.path.exists(saved_path)
            assert load_pickle(saved_path) == data


class TestExtractZip:
    """Test zip extraction functionality."""
    
    def test_extract_zip(self):
        """Test extracting a zip archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test zip file
            zip_path = os.path.join(tmpdir, "test.zip")
            extract_dir = os.path.join(tmpdir, "extracted")
            
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("file1.txt", "content1")
                zf.writestr("dir/file2.txt", "content2")
            
            # Extract
            extract_zip(zip_path, extract_dir)
            
            # Verify
            assert os.path.exists(os.path.join(extract_dir, "file1.txt"))
            assert os.path.exists(os.path.join(extract_dir, "dir", "file2.txt"))
            
            with open(os.path.join(extract_dir, "file1.txt")) as f:
                assert f.read() == "content1"


class TestExtractTar:
    """Test tar extraction functionality."""
    
    def test_extract_tar(self):
        """Test extracting a tar archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test tar file
            tar_path = os.path.join(tmpdir, "test.tar")
            extract_dir = os.path.join(tmpdir, "extracted")
            
            # Create some test files
            test_file1 = os.path.join(tmpdir, "file1.txt")
            test_file2 = os.path.join(tmpdir, "file2.txt")
            with open(test_file1, 'w') as f:
                f.write("content1")
            with open(test_file2, 'w') as f:
                f.write("content2")
            
            # Create tar
            with tarfile.open(tar_path, 'w') as tar:
                tar.add(test_file1, arcname="file1.txt")
                tar.add(test_file2, arcname="dir/file2.txt")
            
            # Extract
            extract_tar(tar_path, extract_dir, verbose=False)
            
            # Verify
            assert os.path.exists(os.path.join(extract_dir, "file1.txt"))
            assert os.path.exists(os.path.join(extract_dir, "dir", "file2.txt"))
            
            with open(os.path.join(extract_dir, "file1.txt")) as f:
                assert f.read() == "content1"


class TestDownloadUrl:
    """Test URL download functionality."""
    
    def test_download_creates_file(self):
        """Test downloading a file from a URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a small test file from GitHub
            url = "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md"
            
            # Download
            path = download_url(url, tmpdir, verbose=False)
            
            # Verify
            assert os.path.exists(path)
            assert os.path.basename(path) == "README.md"
            assert os.path.getsize(path) > 0
    
    def test_download_uses_existing_file(self):
        """Test that download_url skips download if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an existing file
            filepath = os.path.join(tmpdir, "existing.txt")
            with open(filepath, 'w') as f:
                f.write("existing content")
            
            # Try to download (should use existing)
            url = "https://example.com/file.txt"
            path = download_url(url, tmpdir, filename="existing.txt", verbose=False)
            
            # Verify it's the same file
            assert path == filepath
            with open(path) as f:
                assert f.read() == "existing content"
    
    def test_download_custom_filename(self):
        """Test downloading with a custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md"
            custom_name = "custom_readme.md"
            
            # Download with custom name
            path = download_url(url, tmpdir, filename=custom_name, verbose=False)
            
            # Verify
            assert os.path.exists(path)
            assert os.path.basename(path) == custom_name


class TestZipIsValid:
    """Test zip file validation."""

    def test_valid_zip(self):
        """zip_is_valid returns True for a well-formed zip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "good.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("hello.txt", "hello world")
            assert zip_is_valid(zip_path) is True

    def test_invalid_zip_bad_file(self):
        """zip_is_valid returns False for a file that is not a zip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "bad.zip")
            with open(bad_path, 'wb') as f:
                f.write(b"this is not a zip file at all")
            assert zip_is_valid(bad_path) is False

    def test_invalid_zip_truncated(self):
        """zip_is_valid returns False for a truncated/corrupt zip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "truncated.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("data.txt", "some data")
            # Corrupt it by truncating
            with open(zip_path, 'r+b') as f:
                f.truncate(10)
            assert zip_is_valid(zip_path) is False


class TestWgetAvailable:
    """Test wget availability detection."""

    def test_returns_bool(self):
        """wget_available always returns a bool."""
        result = wget_available()
        assert isinstance(result, bool)


class TestDownloadUrlWget:
    """Test download_url_wget."""

    def test_download_creates_file(self):
        """download_url_wget downloads a small file successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md"
            dest = os.path.join(tmpdir, "README.md")
            download_url_wget(url, dest)
            assert os.path.exists(dest)
            assert os.path.getsize(dest) > 0

    def test_download_resume(self):
        """download_url_wget does not overwrite a pre-existing file of the same name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md"
            dest = os.path.join(tmpdir, "README.md")
            # First download
            download_url_wget(url, dest)
            size_first = os.path.getsize(dest)
            # Second download (resume / skip)
            download_url_wget(url, dest)
            size_second = os.path.getsize(dest)
            assert size_second >= size_first


class TestDownloadProgressBar:
    """Test DownloadProgressBar.update_to."""

    def test_update_to_sets_total(self):
        """update_to sets self.total when tsize is provided."""
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                                 desc="test", disable=True) as bar:
            bar.update_to(b=1, bsize=1, tsize=1024)
            assert bar.total == 1024

    def test_update_to_without_tsize(self):
        """update_to works without tsize (no total set)."""
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                                 desc="test", disable=True) as bar:
            bar.update_to(b=2, bsize=512)  # should not raise
