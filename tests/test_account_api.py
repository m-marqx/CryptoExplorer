import unittest
from unittest.mock import patch, Mock
import warnings

import pandas as pd

from src.crypto_explorer.api.account_api import MoralisHandler, BlockscoutHandler, AccountAPI

class TestMoralisHandler(unittest.TestCase):
    def setUp(self):
        self.moralis_handler = MoralisHandler(verbose=False, api_key="dummy_key")

        self.transactions = pd.read_parquet(
            "tests/test_data/transactions.parquet"
        )

        self.aligned_transactions = [
            self.transactions.iloc[column].to_dict()
            for column in range(self.transactions.shape[0])
        ]

        self.dummy_balances = pd.read_parquet("tests/test_data/dummy_balances.parquet")[
            0
        ].to_numpy()

        self.dummy_balances = [list(x) for x in self.dummy_balances]

    @patch("src.crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps(self, mock_transactions):
        mock_transactions.return_value = self.aligned_transactions

        result = self.moralis_handler.get_account_swaps(
            wallet="0x1",
            coin_name=False,
        )

        expected_result = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("src.crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps_with_coin_name(self, mock_get):
        mock_get.return_value = self.aligned_transactions

        result = self.moralis_handler.get_account_swaps(
            wallet="0x1",
            coin_name=True,
        )

        expected_result = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result_coin_name.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)


class TestBlockscouHandler(unittest.TestCase):
    def setUp(self):
        self.blockscout_handler = BlockscoutHandler(verbose=False)

    @patch("requests.get")
    def test_get_account_transactions_no_coin_names(self, mock_get):
        mock_response_1 = Mock()
        mock_response_1.json.return_value = {
            "items": [
                {
                    "hash": "0xf0",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "7022820013343358"},
                    "status": "ok",
                },
                {
                    "hash": "0xf1",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "4945950006759465"},
                    "status": "ok",
                },
                {
                    "hash": "0xf2",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "6832756486003916"},
                    "status": "ok",
                },
                {
                    "hash": "0xf3",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "7861015823084520"},
                    "status": "ok",
                },
                {
                    "hash": "0xf4",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "4802190080676792"},
                    "status": "ok",
                },
                {
                    "hash": "0xf5",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "9602190080676792"},
                    "status": "error",
                },
            ]
        }

        mock_response_2 = Mock()
        mock_response_2.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7757769"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12859"},
                },
            ]
        }

        mock_response_3 = Mock()
        mock_response_3.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7740000"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12831"},
                },
            ]
        }

        mock_response_4 = Mock()
        mock_response_4.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12602"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7594096"},
                },
            ]
        }

        mock_response_5 = Mock()
        mock_response_5.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12647"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7903673"},
                },
            ]
        }

        mock_response_6 = Mock()
        mock_response_6.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7664795"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12589"},
                },
            ]
        }

        mock_get.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_3,
            mock_response_4,
            mock_response_5,
            mock_response_6,
        ]

        result = self.blockscout_handler.get_account_swaps(
            "0x1",
            False,
        )

        expected_result = pd.DataFrame([
            {
                "from": 7.757769,
                "to": 0.00012859,
                "USD Price": 60329.489073800454,
                "txn_fee": 0.007022820013343358,
            },
            {
                "from": 7.74,
                "to": 0.00012831,
                "USD Price": 60322.656067336924,
                "txn_fee": 0.004945950006759465,
            },
            {
                "from": 0.00012602,
                "to": 7.594096,
                "USD Price": 60261.03793048723,
                "txn_fee": 0.006832756486003916,
            },
            {
                "from": 0.00012647,
                "to": 7.903673,
                "USD Price": 62494.44927650827,
                "txn_fee": 0.00786101582308452,
            },
            {
                "from": 7.664795,
                "to": 0.00012589,
                "USD Price": 60884.85979823655,
                "txn_fee": 0.004802190080676792,
            },
        ])

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("requests.get")
    def test_get_account_transactions_coin_names(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "hash": "0xf0",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "7022820013343358"},
                    "status": "ok",
                },
                {
                    "hash": "0xf1",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "4945950006759465"},
                    "status": "ok",
                },
                {
                    "hash": "0xf2",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "6832756486003916"},
                    "status": "ok",
                },
                {
                    "hash": "0xf3",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "7861015823084520"},
                    "status": "ok",
                },
                {
                    "hash": "0xf4",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "4802190080676792"},
                    "status": "ok",
                },
                {
                    "hash": "0xf5",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "9602190080676792"},
                    "status": "error",
                },
            ]
        }

        mock_response_1 = Mock()
        mock_response_1.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7757769"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12859"},
                },
            ]
        }

        mock_response_2 = Mock()
        mock_response_2.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7740000"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12831"},
                },
            ]
        }

        mock_response_3 = Mock()
        mock_response_3.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12602"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7594096"},
                },
            ]
        }

        mock_response_4 = Mock()
        mock_response_4.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12647"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7903673"},
                },
            ]
        }

        mock_response_5 = Mock()
        mock_response_5.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7664795"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12589"},
                },
            ]
        }

        mock_get.side_effect = [
            mock_response,
            mock_response_1,
            mock_response_2,
            mock_response_3,
            mock_response_4,
            mock_response_5,
        ]

        result = self.blockscout_handler.get_account_swaps(
            "0x1",
            True,
        )

        expected_result = pd.DataFrame([
            {
                "from": 7.757769,
                "to": 0.00012859,
                "USD Price": 60329.489073800454,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.007022820013343358,
            },
            {
                "from": 7.74,
                "to": 0.00012831,
                "USD Price": 60322.656067336924,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.004945950006759465,
            },
            {
                "from": 0.00012602,
                "to": 7.594096,
                "USD Price": 60261.03793048723,
                "from_coin_name": "WBTC",
                "to_coin_name": "USDT",
                "txn_fee": 0.006832756486003916,
            },
            {
                "from": 0.00012647,
                "to": 7.903673,
                "USD Price": 62494.44927650827,
                "from_coin_name": "WBTC",
                "to_coin_name": "USDT",
                "txn_fee": 0.00786101582308452,
            },
            {
                "from": 7.664795,
                "to": 0.00012589,
                "USD Price": 60884.85979823655,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.004802190080676792,
            },
        ])

        pd.testing.assert_frame_equal(result, expected_result)


class TestAccountAPI(unittest.TestCase):
    def setUp(self):
        self.account_api = AccountAPI(verbose=False, api_key="dummy_key")
        self.blockscout_handler = BlockscoutHandler(verbose=False)
        self.moralis_handler = MoralisHandler(
            verbose=False, api_key="dummy_key"
        )

        self.transactions = pd.read_parquet(
            "tests/test_data/transactions.parquet"
        )

        self.aligned_transactions = [
            self.transactions.iloc[column].to_dict()
            for column in range(self.transactions.shape[0])
        ]

    @patch("src.crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps_coin_names(self, mock_get):
        mock_get.return_value = self.aligned_transactions

        result = self.moralis_handler.get_account_swaps(
            wallet="0x1",
            coin_name=True,
        )

        expected_result = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result_coin_name.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("src.crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps_no_coin_names(self, mock_transactions):
        mock_transactions.return_value = self.aligned_transactions

        result = self.account_api.get_wallet_swaps(
            wallet="0x1",
            coin_name=False,
        )

        expected_result = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("requests.get")
    def test_get_account_swaps_blockscout_coin_names(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "hash": "0xf0",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "7022820013343358"},
                    "status": "ok",
                },
                {
                    "hash": "0xf1",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "4945950006759465"},
                    "status": "ok",
                },
                {
                    "hash": "0xf2",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "6832756486003916"},
                    "status": "ok",
                },
                {
                    "hash": "0xf3",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "7861015823084520"},
                    "status": "ok",
                },
                {
                    "hash": "0xf4",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "4802190080676792"},
                    "status": "ok",
                },
                {
                    "hash": "0xf5",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "9602190080676792"},
                    "status": "error",
                },
            ]
        }

        mock_response_1 = Mock()
        mock_response_1.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7757769"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12859"},
                },
            ]
        }

        mock_response_2 = Mock()
        mock_response_2.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7740000"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12831"},
                },
            ]
        }

        mock_response_3 = Mock()
        mock_response_3.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12602"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7594096"},
                },
            ]
        }

        mock_response_4 = Mock()
        mock_response_4.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12647"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7903673"},
                },
            ]
        }

        mock_response_5 = Mock()
        mock_response_5.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7664795"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12589"},
                },
            ]
        }

        mock_get.side_effect = [
            mock_response,
            mock_response_1,
            mock_response_2,
            mock_response_3,
            mock_response_4,
            mock_response_5,
        ]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = self.account_api.get_wallet_swaps(
                wallet="0x1",
                coin_name=True,
            )

        expected_result = pd.DataFrame([
            {
                "from": 7.757769,
                "to": 0.00012859,
                "USD Price": 60329.489073800454,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.007022820013343358,
            },
            {
                "from": 7.74,
                "to": 0.00012831,
                "USD Price": 60322.656067336924,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.004945950006759465,
            },
            {
                "from": 0.00012602,
                "to": 7.594096,
                "USD Price": 60261.03793048723,
                "from_coin_name": "WBTC",
                "to_coin_name": "USDT",
                "txn_fee": 0.006832756486003916,
            },
            {
                "from": 0.00012647,
                "to": 7.903673,
                "USD Price": 62494.44927650827,
                "from_coin_name": "WBTC",
                "to_coin_name": "USDT",
                "txn_fee": 0.00786101582308452,
            },
            {
                "from": 7.664795,
                "to": 0.00012589,
                "USD Price": 60884.85979823655,
                "from_coin_name": "USDT",
                "to_coin_name": "WBTC",
                "txn_fee": 0.004802190080676792,
            },
        ])

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("requests.get")
    def test_get_account_swaps_blockscout_no_coin_names(self, mock_get):
        mock_response_1 = Mock()
        mock_response_1.json.return_value = {
            "items": [
                {
                    "hash": "0xf0",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "7022820013343358"},
                    "status": "ok",
                },
                {
                    "hash": "0xf1",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "4945950006759465"},
                    "status": "ok",
                },
                {
                    "hash": "0xf2",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "6832756486003916"},
                    "status": "ok",
                },
                {
                    "hash": "0xf3",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "7861015823084520"},
                    "status": "ok",
                },
                {
                    "hash": "0xf4",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "4802190080676792"},
                    "status": "ok",
                },
                {
                    "hash": "0xf5",
                    "method": "processRouteWithTransferValueOutput",
                    "fee": {"type": "actual", "value": "9602190080676792"},
                    "status": "error",
                },
            ]
        }

        mock_response_2 = Mock()
        mock_response_2.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7757769"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12859"},
                },
            ]
        }

        mock_response_3 = Mock()
        mock_response_3.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7740000"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12831"},
                },
            ]
        }

        mock_response_4 = Mock()
        mock_response_4.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12602"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7594096"},
                },
            ]
        }

        mock_response_5 = Mock()
        mock_response_5.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12647"},
                },
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7903673"},
                },
            ]
        }

        mock_response_6 = Mock()
        mock_response_6.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "7664795"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "12589"},
                },
            ]
        }

        mock_get.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_3,
            mock_response_4,
            mock_response_5,
            mock_response_6,
        ]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = self.account_api.get_wallet_swaps(
                wallet="0x1",
                coin_name=False,
            )

        expected_result = pd.DataFrame([
            {
                "from": 7.757769,
                "to": 0.00012859,
                "USD Price": 60329.489073800454,
                "txn_fee": 0.007022820013343358,
            },
            {
                "from": 7.74,
                "to": 0.00012831,
                "USD Price": 60322.656067336924,
                "txn_fee": 0.004945950006759465,
            },
            {
                "from": 0.00012602,
                "to": 7.594096,
                "USD Price": 60261.03793048723,
                "txn_fee": 0.006832756486003916,
            },
            {
                "from": 0.00012647,
                "to": 7.903673,
                "USD Price": 62494.44927650827,
                "txn_fee": 0.00786101582308452,
            },
            {
                "from": 7.664795,
                "to": 0.00012589,
                "USD Price": 60884.85979823655,
                "txn_fee": 0.004802190080676792,
            },
        ])

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("src.crypto_explorer.api.account_api.AccountAPI.get_wallet_swaps")
    def test_get_buys(self, mock_get_wallet_swaps):
        mock_get_wallet_swaps.return_value = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result_coin_name.parquet"
        )

        expected_result = pd.DataFrame({
            "from": {
                1: 28.526519,
                2: 28.526519,
                3: 28.526519,
                4: 28.526519,
                5: 28.526519,
            },
            "to": {
                1: 0.0003421,
                2: 0.0003421,
                3: 0.0003421,
                4: 0.0003421,
                5: 0.0003421,
            },
            "USD Price": {
                1: 83386.49225372697,
                2: 83386.49225372697,
                3: 83386.49225372697,
                4: 83386.49225372697,
                5: 83386.49225372697,
            },
            "from_coin_name": {1: "USDT", 2: "USDT", 3: "USDT", 4: "USDT", 5: "USDT"},
            "to_coin_name": {1: "WBTC", 2: "WBTC", 3: "WBTC", 4: "WBTC", 5: "WBTC"},
            "txn_fee": {
                1: 0.019240015994738333,
                2: 0.013693531986716086,
                3: 0.064321688,
                4: 0.046096036130056674,
                5: 0.032091566188846524,
            },
        })

        result = self.account_api.get_buys(wallet_address="0x1", asset_name="WBTC")
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("src.crypto_explorer.api.account_api.AccountAPI.get_wallet_swaps")
    def test_get_sells(self, mock_get_wallet_swaps):
        mock_get_wallet_swaps.return_value = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result_coin_name.parquet"
        )

        expected_result = pd.DataFrame(
            {
                "from": {
                    1: 0.0003421,
                    2: 0.0003421,
                    3: 0.0003421,
                    4: 0.0003421,
                    5: 0.0003421,
                },
                "to": {
                    1: 28.526519,
                    2: 28.526519,
                    3: 28.526519,
                    4: 28.526519,
                    5: 28.526519,
                },
                "USD Price": {
                    1: 83386.49225372697,
                    2: 83386.49225372697,
                    3: 83386.49225372697,
                    4: 83386.49225372697,
                    5: 83386.49225372697,
                },
                "from_coin_name": {
                    1: "WBTC",
                    2: "WBTC",
                    3: "WBTC",
                    4: "WBTC",
                    5: "WBTC",
                },
                "to_coin_name": {
                    1: "USDT",
                    2: "USDT",
                    3: "USDT",
                    4: "USDT",
                    5: "USDT",
                },
                "txn_fee": {
                    1: 0.022341900917041562,
                    2: 0.012738525181396596,
                    3: 0.008895849581408056,
                    4: 0.01429167502286668,
                    5: 0.018425064080938674,
                },
            }
        )

        result = self.account_api.get_sells(
            wallet_address="0x1", asset_name="WBTC"
        )
        pd.testing.assert_frame_equal(result, expected_result)
