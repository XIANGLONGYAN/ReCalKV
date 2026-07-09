"""Centralized dataset paths for ReCalKV.

The datasets prepared by ``prepare_data.py`` are stored under ``DATA_ROOT`` using
``datasets.save_to_disk``. Override the location with the ``RECALKV_DATA_ROOT``
environment variable if you keep the data elsewhere.
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.environ.get("RECALKV_DATA_ROOT", os.path.join(_REPO_ROOT, "data"))


def dataset_dir(*parts):
    return os.path.join(DATA_ROOT, *parts)
