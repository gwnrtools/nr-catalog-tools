"""Tests for url_exists retry-with-exponential-backoff behaviour."""

from unittest.mock import MagicMock, patch

from nrcatalogtools.utils import url_exists


def test_url_exists_returns_false_after_all_retries_fail():
    """url_exists returns False when every attempt raises a network error."""
    with patch(
        "nrcatalogtools.utils.requests.head", side_effect=OSError("network error")
    ), patch("nrcatalogtools.utils.time.sleep"):
        result = url_exists("https://example.invalid/file.h5", num_retries=5)
    assert result is False


def test_url_exists_returns_true_after_transient_failure():
    """url_exists returns True when the second attempt succeeds."""
    import requests as _req

    ok = MagicMock()
    ok.status_code = _req.codes.ok
    side_effects = [OSError("transient"), ok]

    with patch("nrcatalogtools.utils.requests.head", side_effect=side_effects), patch(
        "nrcatalogtools.utils.time.sleep"
    ):
        result = url_exists("https://example.invalid/file.h5", num_retries=2)
    assert result is True


def test_url_exists_returns_false_on_non_200_status():
    """url_exists returns False immediately for a non-200 HTTP response."""
    bad = MagicMock()
    bad.status_code = 404
    with patch("nrcatalogtools.utils.requests.head", return_value=bad):
        result = url_exists("https://example.invalid/missing.h5", num_retries=5)
    assert result is False


def test_url_exists_retry_count_is_respected():
    """url_exists makes exactly num_retries attempts before giving up."""
    with patch(
        "nrcatalogtools.utils.requests.head", side_effect=OSError("fail")
    ) as mock_head, patch("nrcatalogtools.utils.time.sleep"):
        url_exists("https://example.invalid/file.h5", num_retries=3)
    assert mock_head.call_count == 3
