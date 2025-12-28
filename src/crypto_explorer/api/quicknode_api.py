import logging
import json
import time
import requests
from crypto_explorer.custom_exceptions import ApiError


class QuickNodeAPI:
    """
    API client for making requests to QuickNode's Bitcoin RPC endpoints.

    Uses multiple API keys for redundancy and implements rate limiting.

    Parameters
    ----------
    api_keys : list
        List of QuickNode API endpoint URLs.
    default_api_key_idx : int
        Starting index in the api_keys list to begin requests from.
    """

    RATE_LIMIT_SECONDS = 1
    CONNECTION_RETRY_SECONDS = 300
    TIMEOUT_RETRY_SECONDS = 120
    REQUEST_TIMEOUT = 60

    def __init__(self, api_keys: list, default_api_key_idx: int):
        """
        Initialize QuickNodeAPI client with multiple API endpoints.

        Parameters
        ----------
        api_keys : list
            List of QuickNode API endpoint URLs
        default_api_key_idx : int
            Initial index position in api_keys list to start making
            requests from. Must be between 0 and len(api_keys)-1.

        Raises
        ------
        ValueError
            If api_keys list is empty or default_api_key_idx is out of bounds.
        """
        if not api_keys:
            raise ValueError("api_keys list cannot be empty")
        if not 0 <= default_api_key_idx < len(api_keys):
            raise ValueError(
                f"default_api_key_idx must be between 0 and {len(api_keys) - 1}"
            )

        self.api_keys = api_keys
        self.default_api_key_idx = default_api_key_idx

        self.logger = logging.getLogger("quicknode_API")
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(handler)
        self.logger.propagate = True

    def _check_response(self, response: requests.Response) -> dict | None:
        """
        Check and handle API response errors.

        Parameters
        ----------
        response : requests.Response
            The response object from the API request.

        Returns
        -------
        dict or None
            The JSON result if successful, None if 403 (skip to next key).

        Raises
        ------
        ApiError
            If the response indicates an error or cannot be parsed.
        """
        if response.ok:
            return response.json()["result"]

        if response.status_code == 403:
            self.logger.critical("Forbidden error, skipping key")
            self.logger.critical(response.content)
            return None

        try:
            error_data = response.json()
            raise ApiError(f"{error_data}")
        except json.JSONDecodeError:
            raise ApiError(response.text)

    def _handle_request_exception(self, exception: Exception) -> int | None:
        """
        Handle request exceptions and return retry delay.

        Parameters
        ----------
        exception : Exception
            The exception that was raised during the request.

        Returns
        -------
        int or None
            Seconds to wait before retry, or None to skip to next key.

        Raises
        ------
        ApiError
            If the error cannot be recovered from.
        """
        if isinstance(exception, requests.exceptions.SSLError):
            self.logger.critical("SSLError, skipping key")
            self.logger.critical("Error message: %s", exception)
            return None

        if isinstance(exception, requests.exceptions.ConnectionError):
            self.logger.critical(
                "Connection error, retrying in %d seconds",
                self.CONNECTION_RETRY_SECONDS
            )
            return self.CONNECTION_RETRY_SECONDS

        if isinstance(exception, requests.exceptions.Timeout):
            self.logger.critical(
                "Timeout error, retrying in %d seconds",
                self.TIMEOUT_RETRY_SECONDS
            )
            return self.TIMEOUT_RETRY_SECONDS

        raise ApiError(f"Unexpected error: {exception}")

    def _make_request(self, payload: str) -> dict:
        """
        Make a request with automatic failover and rate limiting.

        Iterates through API keys starting from default_api_key_idx,
        handles exceptions with retry logic, and enforces rate limiting.

        Parameters
        ----------
        payload : str
            JSON-encoded payload for the POST request.

        Returns
        -------
        dict
            The result from the successful API response.

        Raises
        ------
        ApiError
            If all API keys are exhausted without success.
        """
        headers = {"Content-Type": "application/json"}

        for idx in range(self.default_api_key_idx, len(self.api_keys)):
            api_key = self.api_keys[idx]
            self.default_api_key_idx = idx

            start = time.perf_counter()

            try:
                response = requests.request(
                    "POST",
                    api_key,
                    headers=headers,
                    data=payload,
                    timeout=self.REQUEST_TIMEOUT
                )
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout
            ) as e:
                retry_delay = self._handle_request_exception(e)
                if retry_delay is None:
                    continue
                time.sleep(retry_delay)
                return self._make_request(payload)

            result = self._check_response(response)
            if result is None:
                continue

            self._enforce_rate_limit(start)
            return result

        raise ApiError("All API keys exhausted")

    def _enforce_rate_limit(self, start_time: float) -> None:
        """
        Enforce rate limiting by sleeping if request was too fast.

        Parameters
        ----------
        start_time : float
            The time.perf_counter() value from when the request started.
        """
        elapsed = time.perf_counter() - start_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)

    def get_block_stats(self, block_height: int) -> dict:
        """
        Retrieve statistics for a Bitcoin block by height.

        Makes POST requests to QuickNode API endpoints with automatic
        failover between provided API keys. Implements rate limiting.

        Parameters
        ----------
        block_height : int
            Block height to get statistics for.

        Returns
        -------
        dict
            Block statistics from the Bitcoin RPC getblockstats method.

        Raises
        ------
        ApiError
            If all API key requests fail.
        """
        payload = json.dumps({
            "method": "getblockstats",
            "params": [block_height],
        })
        return self._make_request(payload)

    def get_blockchain_info(self) -> dict:
        """
        Retrieve information about the Bitcoin blockchain.

        Makes POST requests to QuickNode API endpoints with automatic
        failover between provided API keys.

        Returns
        -------
        dict
            Information about the Bitcoin blockchain from the Bitcoin
            RPC getblockchaininfo method.

        Raises
        ------
        ApiError
            If all API key requests fail.
        """
        payload = json.dumps({"method": "getblockchaininfo"})
        return self._make_request(payload)
