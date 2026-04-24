"""
Input/output utilities for data handling.

This module provides utilities for downloading, extracting, and saving/loading
data files, including support for zip/tar archives and pickle files.
"""
import os
import pickle
import tarfile
import urllib.request
import zipfile
import logging
from typing import Any, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_zip(path: str, folder: str):
    """
    Extract a zip archive to a specific folder.

    Args:
        path: The path to the zip archive.
        folder: The destination folder.
    """
    logger.info(f"Extracting {path}")
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def extract_tar(path: str, folder: str, verbose: bool = True):
    """
    Extract a tar (or tar.gz) archive to a specific folder.

    Args:
        path: The path to the tar(gz) archive.
        folder: The destination folder.
        verbose: If False, will not show progress bars (default: True).
    """
    logger.info(f"Extracting {path}")
    with tarfile.open(path, 'r') as tar:
        for member in tqdm(iterable=tar.getmembers(),
                           total=len(tar.getmembers()),
                           disable=not verbose):
            tar.extract(member=member, path=folder)


def save_pickle(obj: Any, filename: str) -> str:
    """
    Save object to file as pickle.

    Args:
        obj: Object to be saved.
        filename: Where to save the file.

    Returns:
        str: The absolute path to the saved pickle.
    """
    abspath = os.path.abspath(filename)
    directory = os.path.dirname(abspath)
    os.makedirs(directory, exist_ok=True)
    with open(abspath, 'wb') as fp:
        pickle.dump(obj, fp)
    return abspath


def load_pickle(filename: str) -> Any:
    """
    Load object from pickle file.

    Args:
        filename: The absolute path to the saved pickle.

    Returns:
        Any: The loaded object.
    """
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data


class DownloadProgressBar(tqdm):
    """
    Progress bar for file downloads.

    Extends tqdm to show download progress with file size information.
    Adapted from https://stackoverflow.com/a/53877507
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update progress bar based on download progress.

        Args:
            b: Number of blocks transferred so far (default: 1).
            bsize: Size of each block in bytes (default: 1).
            tsize: Total size in blocks (default: None).
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_urllib(url: str,
                    folder: str,
                    filename: Optional[str] = None,
                    verbose: bool = True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        filename (string, optional): The filename. If :obj:`None`, inferred from
            url.
        verbose (bool, optional): If :obj:`False`, will not show progress bars.
            (default: :obj:`True`)
    """
    if filename is None:
        filename = url.rpartition('/')[2].split('?')[0]
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        logger.info(f'Using existing file {filename}')
        return path

    logger.info(f'Downloading {url}')

    os.makedirs(folder, exist_ok=True)

    # From https://stackoverflow.com/a/53877507
    with DownloadProgressBar(unit='B',
                             unit_scale=True,
                             miniters=1,
                             desc=url.split('/')[-1],
                             disable=not verbose) as t:
        urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)
    return path


def zip_is_valid(path: str) -> bool:
    """Return True if *path* is a structurally valid zip with correct CRCs."""
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()  # returns first bad filename or None
        return bad is None
    except zipfile.BadZipFile:
        return False


def wget_available() -> bool:
    import shutil as _shutil
    return _shutil.which("wget") is not None


def download_file(url: str, dest: str) -> None:
    """Download *url* to *dest*.

    Uses ``wget --continue`` when available (handles large files and
    network interruptions much better than urllib).  Falls back to a
    pure-Python streaming download with ``Range`` resume support.
    """
    if wget_available():
        import subprocess
        print(f"\nDownloading {os.path.basename(dest)} via wget ...")
        subprocess.run(
            [
                "wget",
                "--continue",          # resume partial downloads
                "--tries=10",          # retry up to 10 times on error
                "--retry-connrefused",
                "--waitretry=5",       # wait 5 s between retries
                "--show-progress",
                "-O", dest,
                url,
            ],
            check=True,
        )
    else:
        downloaded = os.path.getsize(dest) if os.path.exists(dest) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={downloaded}-"})
        with urllib.request.urlopen(req) as r:
            total = downloaded + int(r.headers.get("Content-Length", 0))
            print(f"\nDownloading {os.path.basename(dest)} ({total / 1e9:.2f} GB)")
            with open(dest, "ab") as f:
                while chunk := r.read(1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded / total * 100
                    bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
                    print(f"\r  [{bar}] {pct:.1f}%  {downloaded/1e9:.2f}/{total/1e9:.2f} GB", end="")
        print()