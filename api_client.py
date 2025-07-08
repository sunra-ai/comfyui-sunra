"""
Sunra API client wrapper.

This module provides a unified interface for calling Sunra AI APIs,
handling authentication, error handling, and response processing.
"""

import os
from typing import Dict, Any, Optional, Callable

import requests

from .constants import DEFAULT_API_URL, API_TIMEOUT, USER_AGENT

try:
    import sunra_client

    SUNRA_CLIENT_AVAILABLE = True
except ImportError:
    SUNRA_CLIENT_AVAILABLE = False
    print("Warning: sunra_client not available. Using fallback HTTP client.")


class SunraAPIClient:
    """
    Unified API client for Sunra AI services.

    This client can use either the official sunra_client library
    or fall back to direct HTTP calls if needed.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            api_key: Optional API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("SUNRA_KEY")
        if not self.api_key:
            raise ValueError(
                "SUNRA_KEY environment variable is required. "
                "Get your API key from https://sunra.ai"
            )

        self.use_sunra_client = SUNRA_CLIENT_AVAILABLE
        self.base_url = DEFAULT_API_URL

    def call(
        self,
        endpoint: str,
        arguments: Dict[str, Any],
        with_logs: bool = True,
        on_progress: Optional[Callable] = None,
    ) -> Any:
        """
        Call a Sunra API endpoint.

        Args:
            endpoint: The API endpoint (e.g., "black-forest-labs/flux-kontext-dev/image-to-image")
            arguments: Dictionary of arguments for the API call
            with_logs: Whether to include logs in the response
            on_progress: Optional callback for progress updates

        Returns:
            API response (format depends on the endpoint)
        """
        if self.use_sunra_client:
            return self._call_with_sunra_client(
                endpoint, arguments, with_logs, on_progress
            )
        else:
            return self._call_with_http(endpoint, arguments)

    def _call_with_sunra_client(
        self,
        endpoint: str,
        arguments: Dict[str, Any],
        with_logs: bool = True,
        on_progress: Optional[Callable] = None,
    ) -> Any:
        """Call API using the official sunra_client library."""
        # Set up callbacks
        callbacks = {}
        if on_progress:
            callbacks["on_queue_update"] = on_progress

        if with_logs:
            callbacks["on_enqueue"] = lambda x: print(f"Request enqueued: {x}")

        # Make the API call
        try:
            response = sunra_client.subscribe(
                endpoint,
                arguments=arguments,  # Pass arguments directly - processing done elsewhere
                with_logs=with_logs,
                **callbacks,
            )
            return response
        except Exception as e:
            self._raise_api_error(e)

    def _call_with_http(self, endpoint: str, arguments: Dict[str, Any]) -> Any:
        """Fallback HTTP client implementation."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        # Construct URL
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.post(
                url,
                headers=headers,
                json=arguments,  # Pass arguments directly - processing done elsewhere
                timeout=API_TIMEOUT,
            )
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            self._raise_api_error(e)

    def _raise_api_error(self, exception: Exception) -> None:
        """Extract request ID and raise a formatted error."""
        request_id = None

        # Try different ways to get request ID
        if hasattr(exception, "request_id"):
            request_id = exception.request_id
        elif hasattr(exception, "response") and exception.response is not None:
            request_id = exception.response.headers.get("x-request-id")

        error_msg = str(exception)
        if request_id:
            error_msg = f"{error_msg} (Request ID: {request_id})"

        raise RuntimeError(f"Sunra API call failed: {error_msg}")

    def check_status(self, request_id: str) -> Dict[str, Any]:
        """
        Check the status of a request.

        Args:
            request_id: The request ID to check

        Returns:
            Status information dictionary
        """
        if SUNRA_CLIENT_AVAILABLE:
            try:
                return sunra_client.status(request_id)
            except Exception as e:
                return {"status": "error", "error": str(e), "progress": 0.0}
        else:
            # Fallback HTTP implementation
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
            }

            try:
                response = requests.get(
                    f"{self.base_url}/status/{request_id}", headers=headers, timeout=30
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"status": "error", "error": str(e), "progress": 0.0}


# Global client instance
_global_client = None


def get_api_client() -> SunraAPIClient:
    """Get or create the global API client instance."""
    global _global_client
    if _global_client is None:
        _global_client = SunraAPIClient()
    return _global_client
