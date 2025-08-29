"""Test pygotcha."""

import pygotcha


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(pygotcha.__name__, str)
