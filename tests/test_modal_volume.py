"""Pytest tests for Modal volume CRUD operations.

Run with:
    /home/agrov/gram/bin/python -m pytest tests/test_modal_volume.py -v
"""
import os
import pytest

modal = pytest.importorskip("modal")

from scripts.modal.test_volume_crud import app, test_add_and_delete


@pytest.fixture(scope="module")
def modal_app():
    """Run the Modal app context for the duration of the test module."""
    with app.run():
        yield app


def test_volume_add_and_delete(modal_app):
    """Test that we can add a file to the volume and then delete it."""
    # Should complete without raising (asserts are inside the function)
    test_add_and_delete.remote()
