import unittest
from unittest.mock import patch, MagicMock
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
