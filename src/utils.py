"""
utils.py | Author : Catherine Wong
General purpose utilities.
"""
import datetime
from pathlib import Path


def escaped_timestamp():
    """[ret]: escaped string timestamp."""
    timestamp = datetime.datetime.now().isoformat()
    # Escape the timestamp.
    timestamp = timestamp.replace(":", "-")
    timestamp = timestamp.replace(".", "-")
    return timestamp


def mkdir_if_necessary(path):
    """Creates a directory if necessary.
    [ret]: string to directory path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
