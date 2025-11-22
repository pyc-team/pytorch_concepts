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
