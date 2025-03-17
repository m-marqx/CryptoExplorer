import unittest
from unittest.mock import patch, MagicMock
import requests
import json
from src.crypto_explorer import QuickNodeAPI


class TestQuickNodeAPI(unittest.TestCase):
    def setUp(self):
        self.api_keys = ["http://dummy-endpoint1", "http://dummy-endpoint2"]
        self.api_client = QuickNodeAPI(api_keys=self.api_keys, default_api_key_idx=0)

    @patch("requests.request")
    def test_get_block_stats(self, mock_request):
        expected_result = {
            "avgfee": 125685,
            "avgfeerate": 340,
            "avgtxsize": 388,
            "blockhash": "00000000000000000024fb37364cbf81fd49cc2d51c09c75c35433c3a1945d04",
            "feerate_percentiles": [317, 329, 340, 346, 393],
            "height": 500000,
            "ins": 4370,
            "maxfee": 6470750,
            "maxfeerate": 884,
            "maxtxsize": 19850,
            "medianfee": 85898,
            "mediantime": 1513620886,
            "mediantxsize": 226,
            "minfee": 1704,
            "minfeerate": 12,
            "mintxsize": 189,
            "outs": 6269,
            "subsidy": 1250000000,
            "swtotal_size": 111337,
            "swtotal_weight": 243925,
            "swtxs": 166,
            "time": 1513622125,
            "total_out": 1401737618054,
            "total_size": 1048257,
            "total_weight": 3991605,
            "totalfee": 339351625,
            "txs": 2701,
            "utxo_increase": 1899,
            "utxo_size_inc": 142171,
            "utxo_increase_actual": 1685,
            "utxo_size_inc_actual": 126712,
        }

        mock_request.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": expected_result},
        )
        result = self.api_client.get_block_stats(500000)
        self.assertEqual(result, expected_result)

    @patch("requests.request")
    def test_get_block_info(self, mock_request):
        expected_result = {
            "chain": "main",
            "blocks": 887250,
            "headers": 887250,
            "bestblockhash": "000000000000000000012c249deea126ed1d9fb2581e0bb42e679c3cb9ca3780",
            "difficulty": 112149504190349.3,
            "time": 1741652595,
            "mediantime": 1741651138,
            "verificationprogress": 0.9999959151971047,
            "initialblockdownload": False,
            "chainwork": "0000000000000000000000000000000000000000b3a243f0021d24f772091011",
            "size_on_disk": 732614355026,
            "pruned": False,
            "warnings": "",
        }

        mock_request.return_value = MagicMock(status_code=200, json=lambda: {"result": expected_result})
        result = self.api_client.get_blockchain_info()
        self.assertEqual(result, expected_result)

    @patch("time.sleep")
    @patch("requests.request")
    def test_get_block_stats_connection_error(self, mock_request, mock_sleep):
        # Set up mock to raise ConnectionError on first call, then return success on second call
        expected_result = {
            "avgfee": 125685,
            "avgfeerate": 340,
            "avgtxsize": 388,
            "blockhash": "00000000000000000024fb37364cbf81fd49cc2d51c09c75c35433c3a1945d04",
            "feerate_percentiles": [317, 329, 340, 346, 393],
            "height": 500000,
            "ins": 4370,
            "maxfee": 6470750,
            "maxfeerate": 884,
            "maxtxsize": 19850,
            "medianfee": 85898,
            "mediantime": 1513620886,
            "mediantxsize": 226,
            "minfee": 1704,
            "minfeerate": 12,
            "mintxsize": 189,
            "outs": 6269,
            "subsidy": 1250000000,
            "swtotal_size": 111337,
            "swtotal_weight": 243925,
            "swtxs": 166,
            "time": 1513622125,
            "total_out": 1401737618054,
            "total_size": 1048257,
            "total_weight": 3991605,
            "totalfee": 339351625,
            "txs": 2701,
            "utxo_increase": 1899,
            "utxo_size_inc": 142171,
            "utxo_increase_actual": 1685,
            "utxo_size_inc_actual": 126712,
        }

        mock_response = MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {"result": expected_result},
        )

        # Configure the mock to raise ConnectionError first, then return successful response
        mock_request.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            mock_response
        ]

        # Mock the logger to verify messages
        with patch.object(self.api_client, 'logger') as mock_logger:
            result = self.api_client.get_block_stats(500000)

            # Verify the correct error message was logged
            mock_logger.critical.assert_called_once_with("Connection error, retrying in 5 minutes")

            # Verify the result is correctly returned after retry
            self.assertEqual(result, expected_result)

            # Verify request was called twice (once for error, once for success)
            self.assertEqual(mock_request.call_count, 2)

            # Verify the request parameters were correct
            headers = {"Content-Type": "application/json"}
            payload = json.dumps({"method": "getblockstats", "params": [500000]})
            mock_request.assert_called_with(
                "POST", self.api_keys[0], headers=headers, data=payload, timeout=60
            )

    @patch("time.sleep")
    @patch("requests.request")
    def test_get_block_stats_timeout_error(self, mock_request, mock_sleep):
        # Set up mock to raise Timeout on first call, then return success on second call
        expected_result = {
            "avgfee": 125685,
            "avgfeerate": 340,
            "avgtxsize": 388,
            "blockhash": "00000000000000000024fb37364cbf81fd49cc2d51c09c75c35433c3a1945d04",
            "feerate_percentiles": [317, 329, 340, 346, 393],
            "height": 500000,
            "ins": 4370,
            "maxfee": 6470750,
            "maxfeerate": 884,
            "maxtxsize": 19850,
            "medianfee": 85898,
            "mediantime": 1513620886,
            "mediantxsize": 226,
            "minfee": 1704,
            "minfeerate": 12,
            "mintxsize": 189,
            "outs": 6269,
            "subsidy": 1250000000,
            "swtotal_size": 111337,
            "swtotal_weight": 243925,
            "swtxs": 166,
            "time": 1513622125,
            "total_out": 1401737618054,
            "total_size": 1048257,
            "total_weight": 3991605,
            "totalfee": 339351625,
            "txs": 2701,
            "utxo_increase": 1899,
            "utxo_size_inc": 142171,
            "utxo_increase_actual": 1685,
            "utxo_size_inc_actual": 126712,
        }
        mock_response = MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {"result": expected_result},
        )

        # Configure the mock to raise Timeout first, then return successful response
        mock_request.side_effect = [
            requests.exceptions.Timeout("Request timed out"),
            mock_response
        ]

        # Mock the logger to verify messages
        with patch.object(self.api_client, 'logger') as mock_logger:
            result = self.api_client.get_block_stats(500000)

            # Verify the correct error message was logged
            mock_logger.critical.assert_called_once_with("Connection error, retrying in 2 minutes")

            # Verify the result is correctly returned after retry
            self.assertEqual(result, expected_result)

            # # Verify request was called twice
            self.assertEqual(mock_request.call_count, 2)

    @patch("time.sleep")
    @patch("requests.request")
    def test_get_blockchain_info_connection_error(self, mock_request, mock_sleep):
        # Set up mock to raise ConnectionError on first call, then return success on second call
        expected_result = {
            "chain": "main",
            "blocks": 887250,
            "headers": 887250,
            "bestblockhash": "000000000000000000012c249deea126ed1d9fb2581e0bb42e679c3cb9ca3780",
            "difficulty": 112149504190349.3,
            "time": 1741652595,
            "mediantime": 1741651138,
            "verificationprogress": 0.9999959151971047,
            "initialblockdownload": False,
            "chainwork": "0000000000000000000000000000000000000000b3a243f0021d24f772091011",
            "size_on_disk": 732614355026,
            "pruned": False,
            "warnings": "",
        }

        mock_response = MagicMock(
            status_code=200, ok=True, json=lambda: expected_result
        )

        result = self.api_client.get_blockchain_info()

        # Configure the mock to raise ConnectionError first, then return successful response
        mock_request.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            mock_response,
        ]

        # Mock the logger to verify messages
        with patch.object(self.api_client, "logger") as mock_logger:
            result = self.api_client.get_blockchain_info()

            # Verify the correct error message was logged
            mock_logger.critical.assert_called_once_with(
                "Connection error, retrying in 5 minutes"
            )

            # Verify the result is correctly returned after retry
            self.assertEqual(result, expected_result)

            # # Verify request was called three times (once for error, once for success)
            # self.assertEqual(mock_request.call_count, 3)

            # Verify the request parameters were correct
            headers = {"Content-Type": "application/json"}
            payload = json.dumps({"method": "getblockchaininfo"})
            mock_request.assert_called_with(
                "POST",
                self.api_keys[0],
                headers=headers,
                data=payload,
                timeout=60,
            )

    @patch("time.sleep")
    @patch("requests.request")
    def test_get_blockchain_info_timeout_error(self, mock_request, mock_sleep):
        # Set up mock to raise Timeout on first call, then return success on second call
        expected_result = {"height": 500000, "avgfee": 125685}
        mock_response = MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {"height": 500000, "avgfee": 125685},
        )

        # Configure the mock to raise Timeout first, then return successful response
        mock_request.side_effect = [
            requests.exceptions.Timeout("Request timed out"),
            mock_response,
        ]

        # Mock the logger to verify messages
        with patch.object(self.api_client, "logger") as mock_logger:
            result = self.api_client.get_blockchain_info()

            # Verify the correct error message was logged
            mock_logger.critical.assert_called_once_with(
                "Connection error, retrying in 2 minutes"
            )

            # Verify the result is correctly returned after retry
            self.assertEqual(result, expected_result)

            # # Verify request was called twice
            self.assertEqual(mock_request.call_count, 2)
