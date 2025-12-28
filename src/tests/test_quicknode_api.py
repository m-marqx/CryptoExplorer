import unittest
from unittest.mock import patch, MagicMock
import requests
import json
from crypto_explorer import QuickNodeAPI
from crypto_explorer.custom_exceptions import ApiError


class TestQuickNodeAPIInit(unittest.TestCase):
    """Tests for QuickNodeAPI initialization and validation."""

    def test_init_with_valid_params(self):
        """Test successful initialization with valid parameters."""
        api_keys = ["http://endpoint1", "http://endpoint2"]
        client = QuickNodeAPI(api_keys=api_keys, default_api_key_idx=0)
        self.assertEqual(client.api_keys, api_keys)
        self.assertEqual(client.default_api_key_idx, 0)

    def test_init_with_valid_non_zero_index(self):
        """Test initialization with non-zero default index."""
        api_keys = ["http://endpoint1", "http://endpoint2", "http://endpoint3"]
        client = QuickNodeAPI(api_keys=api_keys, default_api_key_idx=2)
        self.assertEqual(client.default_api_key_idx, 2)

    def test_init_empty_api_keys_raises_value_error(self):
        """Test that empty api_keys list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            QuickNodeAPI(api_keys=[], default_api_key_idx=0)
        self.assertEqual(str(context.exception), "api_keys list cannot be empty")

    def test_init_negative_index_raises_value_error(self):
        """Test that negative default_api_key_idx raises ValueError."""
        with self.assertRaises(ValueError) as context:
            QuickNodeAPI(api_keys=["http://endpoint1"], default_api_key_idx=-1)
        self.assertIn("must be between 0 and", str(context.exception))

    def test_init_index_out_of_bounds_raises_value_error(self):
        """Test that out-of-bounds index raises ValueError."""
        with self.assertRaises(ValueError) as context:
            QuickNodeAPI(api_keys=["http://endpoint1", "http://endpoint2"], default_api_key_idx=5)
        self.assertEqual(str(context.exception), "default_api_key_idx must be between 0 and 1")

    def test_init_clears_existing_handlers(self):
        """Test that existing logger handlers are cleared on init."""
        # Create first instance to add handlers
        api_keys = ["http://endpoint1"]
        client1 = QuickNodeAPI(api_keys=api_keys, default_api_key_idx=0)
        initial_handler_count = len(client1.logger.handlers)

        # Create second instance - should clear and re-add handlers
        client2 = QuickNodeAPI(api_keys=api_keys, default_api_key_idx=0)
        self.assertEqual(len(client2.logger.handlers), initial_handler_count)


class TestCheckResponse(unittest.TestCase):
    """Tests for _check_response method."""

    def setUp(self):
        self.api_keys = ["http://endpoint1", "http://endpoint2"]
        self.client = QuickNodeAPI(api_keys=self.api_keys, default_api_key_idx=0)

    def test_check_response_success(self):
        """Test successful response returns result."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": {"data": "test"}}

        result = self.client._check_response(mock_response)
        self.assertEqual(result, {"data": "test"})

    def test_check_response_403_returns_none(self):
        """Test 403 response logs and returns None."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 403
        mock_response.content = b"Forbidden"

        with patch.object(self.client, 'logger') as mock_logger:
            result = self.client._check_response(mock_response)
            self.assertIsNone(result)
            self.assertEqual(mock_logger.critical.call_count, 2)

    def test_check_response_error_with_json(self):
        """Test error response with valid JSON raises ApiError."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal Server Error"}

        with self.assertRaises(ApiError) as context:
            self.client._check_response(mock_response)
        self.assertIn("Internal Server Error", str(context.exception))

    def test_check_response_error_with_invalid_json(self):
        """Test error response with invalid JSON raises ApiError with text."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)
        mock_response.text = "Plain text error"

        with self.assertRaises(ApiError) as context:
            self.client._check_response(mock_response)
        self.assertEqual(str(context.exception), "Plain text error")


class TestHandleRequestException(unittest.TestCase):
    """Tests for _handle_request_exception method."""

    def setUp(self):
        self.api_keys = ["http://endpoint1", "http://endpoint2"]
        self.client = QuickNodeAPI(api_keys=self.api_keys, default_api_key_idx=0)

    def test_ssl_error_returns_none(self):
        """Test SSLError logs and returns None to skip key."""
        ssl_error = requests.exceptions.SSLError("SSL Certificate Error")

        with patch.object(self.client, 'logger') as mock_logger:
            result = self.client._handle_request_exception(ssl_error)
            self.assertIsNone(result)
            self.assertEqual(mock_logger.critical.call_count, 2)

    def test_connection_error_returns_retry_delay(self):
        """Test ConnectionError returns CONNECTION_RETRY_SECONDS."""
        conn_error = requests.exceptions.ConnectionError("Connection refused")

        with patch.object(self.client, 'logger') as mock_logger:
            result = self.client._handle_request_exception(conn_error)
            self.assertEqual(result, self.client.CONNECTION_RETRY_SECONDS)
            mock_logger.critical.assert_called_once()

    def test_timeout_error_returns_retry_delay(self):
        """Test Timeout returns TIMEOUT_RETRY_SECONDS."""
        timeout_error = requests.exceptions.Timeout("Request timed out")

        with patch.object(self.client, 'logger') as mock_logger:
            result = self.client._handle_request_exception(timeout_error)
            self.assertEqual(result, self.client.TIMEOUT_RETRY_SECONDS)
            mock_logger.critical.assert_called_once()

    def test_unexpected_error_raises_api_error(self):
        """Test unexpected exception raises ApiError."""
        unexpected_error = RuntimeError("Unexpected error")

        with self.assertRaises(ApiError) as context:
            self.client._handle_request_exception(unexpected_error)
        self.assertIn("Unexpected error", str(context.exception))


class TestEnforceRateLimit(unittest.TestCase):
    """Tests for _enforce_rate_limit method."""

    def setUp(self):
        self.api_keys = ["http://endpoint1"]
        self.client = QuickNodeAPI(api_keys=self.api_keys, default_api_key_idx=0)

    @patch("time.sleep")
    @patch("time.perf_counter")
    def test_enforce_rate_limit_sleeps_when_fast(self, mock_perf_counter, mock_sleep):
        """Test rate limiting sleeps when request was too fast."""
        mock_perf_counter.return_value = 0.5  # 0.5 seconds elapsed
        start_time = 0.0

        self.client._enforce_rate_limit(start_time)
        mock_sleep.assert_called_once_with(0.5)  # Should sleep for remaining 0.5s

    @patch("time.sleep")
    @patch("time.perf_counter")
    def test_enforce_rate_limit_no_sleep_when_slow(self, mock_perf_counter, mock_sleep):
        """Test rate limiting doesn't sleep when request took long enough."""
        mock_perf_counter.return_value = 2.0  # 2 seconds elapsed (> 1s limit)
        start_time = 0.0

        self.client._enforce_rate_limit(start_time)
        mock_sleep.assert_not_called()


class TestMakeRequest(unittest.TestCase):
    """Tests for _make_request method."""

    def setUp(self):
        self.api_keys = ["http://endpoint1", "http://endpoint2"]
        self.client = QuickNodeAPI(api_keys=self.api_keys, default_api_key_idx=0)

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_make_request_success(self, mock_request, mock_perf_counter, mock_sleep):
        """Test successful request returns result."""
        mock_perf_counter.side_effect = [0.0, 0.5]  # Start and end times
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": {"data": "test"}}
        mock_request.return_value = mock_response

        payload = json.dumps({"method": "test"})
        result = self.client._make_request(payload)

        self.assertEqual(result, {"data": "test"})
        mock_request.assert_called_once()

    @patch("time.sleep")
    @patch("requests.request")
    def test_make_request_all_keys_exhausted(self, mock_request, mock_sleep):
        """Test ApiError raised when all keys return 403."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 403
        mock_response.content = b"Forbidden"
        mock_request.return_value = mock_response

        with self.assertRaises(ApiError) as context:
            self.client._make_request(json.dumps({"method": "test"}))
        self.assertEqual(str(context.exception), "All API keys exhausted")

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_make_request_failover_to_second_key(self, mock_request, mock_perf_counter, mock_sleep):
        """Test failover to second key on 403."""
        mock_perf_counter.side_effect = [0.0, 0.0, 0.5]

        # First key returns 403, second key succeeds
        mock_response_403 = MagicMock()
        mock_response_403.ok = False
        mock_response_403.status_code = 403
        mock_response_403.content = b"Forbidden"

        mock_response_success = MagicMock()
        mock_response_success.ok = True
        mock_response_success.json.return_value = {"result": {"data": "success"}}

        mock_request.side_effect = [mock_response_403, mock_response_success]

        result = self.client._make_request(json.dumps({"method": "test"}))
        self.assertEqual(result, {"data": "success"})
        self.assertEqual(mock_request.call_count, 2)

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_make_request_ssl_error_skips_key(self, mock_request, mock_perf_counter, mock_sleep):
        """Test SSLError causes failover to next key."""
        mock_perf_counter.side_effect = [0.0, 0.0, 0.5]

        mock_response_success = MagicMock()
        mock_response_success.ok = True
        mock_response_success.json.return_value = {"result": {"data": "success"}}

        mock_request.side_effect = [
            requests.exceptions.SSLError("SSL Error"),
            mock_response_success
        ]

        result = self.client._make_request(json.dumps({"method": "test"}))
        self.assertEqual(result, {"data": "success"})
        self.assertEqual(mock_request.call_count, 2)

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_make_request_connection_error_retries(self, mock_request, mock_perf_counter, mock_sleep):
        """Test ConnectionError waits and retries."""
        mock_perf_counter.side_effect = [0.0, 0.0, 0.5]

        mock_response_success = MagicMock()
        mock_response_success.ok = True
        mock_response_success.json.return_value = {"result": {"data": "success"}}

        mock_request.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            mock_response_success
        ]

        result = self.client._make_request(json.dumps({"method": "test"}))
        self.assertEqual(result, {"data": "success"})
        # Verify sleep was called with CONNECTION_RETRY_SECONDS
        mock_sleep.assert_any_call(self.client.CONNECTION_RETRY_SECONDS)

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_make_request_timeout_error_retries(self, mock_request, mock_perf_counter, mock_sleep):
        """Test Timeout waits and retries."""
        mock_perf_counter.side_effect = [0.0, 0.0, 0.5]

        mock_response_success = MagicMock()
        mock_response_success.ok = True
        mock_response_success.json.return_value = {"result": {"data": "success"}}

        mock_request.side_effect = [
            requests.exceptions.Timeout("Request timed out"),
            mock_response_success
        ]

        result = self.client._make_request(json.dumps({"method": "test"}))
        self.assertEqual(result, {"data": "success"})
        # Verify sleep was called with TIMEOUT_RETRY_SECONDS
        mock_sleep.assert_any_call(self.client.TIMEOUT_RETRY_SECONDS)


class TestGetBlockStats(unittest.TestCase):
    """Tests for get_block_stats method."""

    def setUp(self):
        self.api_keys = ["http://endpoint1", "http://endpoint2"]
        self.client = QuickNodeAPI(api_keys=self.api_keys, default_api_key_idx=0)

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_get_block_stats_success(self, mock_request, mock_perf_counter, mock_sleep):
        """Test get_block_stats returns correct data."""
        mock_perf_counter.side_effect = [0.0, 0.5]
        expected_result = {
            "avgfee": 125685,
            "avgfeerate": 340,
            "height": 500000,
        }

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": expected_result}
        mock_request.return_value = mock_response

        result = self.client.get_block_stats(500000)
        self.assertEqual(result, expected_result)

        # Verify correct payload was sent
        call_args = mock_request.call_args
        payload = json.loads(call_args.kwargs['data'])
        self.assertEqual(payload["method"], "getblockstats")
        self.assertEqual(payload["params"], [500000])

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_get_block_stats_connection_error_retry(self, mock_request, mock_perf_counter, mock_sleep):
        """Test get_block_stats retries on connection error."""
        mock_perf_counter.side_effect = [0.0, 0.0, 0.5]
        expected_result = {"height": 500000}

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": expected_result}

        mock_request.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            mock_response
        ]

        with patch.object(self.client, 'logger') as mock_logger:
            result = self.client.get_block_stats(500000)
            self.assertEqual(result, expected_result)
            mock_logger.critical.assert_called_with(
                "Connection error, retrying in %d seconds",
                self.client.CONNECTION_RETRY_SECONDS
            )

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_get_block_stats_timeout_error_retry(self, mock_request, mock_perf_counter, mock_sleep):
        """Test get_block_stats retries on timeout."""
        mock_perf_counter.side_effect = [0.0, 0.0, 0.5]
        expected_result = {"height": 500000}

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": expected_result}

        mock_request.side_effect = [
            requests.exceptions.Timeout("Timeout"),
            mock_response
        ]

        with patch.object(self.client, 'logger') as mock_logger:
            result = self.client.get_block_stats(500000)
            self.assertEqual(result, expected_result)
            mock_logger.critical.assert_called_with(
                "Timeout error, retrying in %d seconds",
                self.client.TIMEOUT_RETRY_SECONDS
            )


class TestGetBlockchainInfo(unittest.TestCase):
    """Tests for get_blockchain_info method."""

    def setUp(self):
        self.api_keys = ["http://endpoint1", "http://endpoint2"]
        self.client = QuickNodeAPI(api_keys=self.api_keys, default_api_key_idx=0)

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_get_blockchain_info_success(self, mock_request, mock_perf_counter, mock_sleep):
        """Test get_blockchain_info returns correct data."""
        mock_perf_counter.side_effect = [0.0, 0.5]
        expected_result = {
            "chain": "main",
            "blocks": 887250,
            "headers": 887250,
        }

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": expected_result}
        mock_request.return_value = mock_response

        result = self.client.get_blockchain_info()
        self.assertEqual(result, expected_result)

        # Verify correct payload was sent
        call_args = mock_request.call_args
        payload = json.loads(call_args.kwargs['data'])
        self.assertEqual(payload["method"], "getblockchaininfo")

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_get_blockchain_info_connection_error_retry(self, mock_request, mock_perf_counter, mock_sleep):
        """Test get_blockchain_info retries on connection error."""
        mock_perf_counter.side_effect = [0.0, 0.0, 0.5]
        expected_result = {"chain": "main"}

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": expected_result}

        mock_request.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            mock_response
        ]

        with patch.object(self.client, 'logger') as mock_logger:
            result = self.client.get_blockchain_info()
            self.assertEqual(result, expected_result)
            mock_logger.critical.assert_called_with(
                "Connection error, retrying in %d seconds",
                self.client.CONNECTION_RETRY_SECONDS
            )

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_get_blockchain_info_timeout_error_retry(self, mock_request, mock_perf_counter, mock_sleep):
        """Test get_blockchain_info retries on timeout."""
        mock_perf_counter.side_effect = [0.0, 0.0, 0.5]
        expected_result = {"chain": "main"}

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": expected_result}

        mock_request.side_effect = [
            requests.exceptions.Timeout("Timeout"),
            mock_response
        ]

        with patch.object(self.client, 'logger') as mock_logger:
            result = self.client.get_blockchain_info()
            self.assertEqual(result, expected_result)
            mock_logger.critical.assert_called_with(
                "Timeout error, retrying in %d seconds",
                self.client.TIMEOUT_RETRY_SECONDS
            )


class TestQuickNodeAPIIntegration(unittest.TestCase):
    """Integration tests for QuickNodeAPI."""

    @patch("time.sleep")
    @patch("time.perf_counter")
    @patch("requests.request")
    def test_full_failover_chain(self, mock_request, mock_perf_counter, mock_sleep):
        """Test complete failover from first to last key."""
        api_keys = ["http://endpoint1", "http://endpoint2", "http://endpoint3"]
        client = QuickNodeAPI(api_keys=api_keys, default_api_key_idx=0)

        mock_perf_counter.side_effect = [0.0, 0.0, 0.0, 0.5]

        # First two keys fail with 403, third succeeds
        mock_response_403 = MagicMock()
        mock_response_403.ok = False
        mock_response_403.status_code = 403
        mock_response_403.content = b"Forbidden"

        mock_response_success = MagicMock()
        mock_response_success.ok = True
        mock_response_success.json.return_value = {"result": {"success": True}}

        mock_request.side_effect = [
            mock_response_403,
            mock_response_403,
            mock_response_success
        ]

        result = client.get_block_stats(100)
        self.assertEqual(result, {"success": True})
        self.assertEqual(mock_request.call_count, 3)
        # Verify final key index is updated
        self.assertEqual(client.default_api_key_idx, 2)

    @patch("time.sleep")
    @patch("requests.request")
    def test_start_from_non_zero_index(self, mock_request, mock_sleep):
        """Test starting from non-zero index skips earlier keys."""
        api_keys = ["http://endpoint1", "http://endpoint2", "http://endpoint3"]
        client = QuickNodeAPI(api_keys=api_keys, default_api_key_idx=1)

        mock_response_403 = MagicMock()
        mock_response_403.ok = False
        mock_response_403.status_code = 403
        mock_response_403.content = b"Forbidden"

        mock_request.return_value = mock_response_403

        with self.assertRaises(ApiError):
            client.get_block_stats(100)

        # Should only try endpoint2 and endpoint3, not endpoint1
        self.assertEqual(mock_request.call_count, 2)