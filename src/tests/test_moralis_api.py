import unittest
from unittest.mock import patch, MagicMock, Mock
from crypto_explorer import MoralisAPI
from crypto_explorer.custom_exceptions import ApiError, InvalidArgumentError
import pandas as pd
import numpy as np
import os
from itertools import cycle
import pytest

class TestMoralisAPI(unittest.TestCase):
    def setUp(self):
        self.api_client = MoralisAPI(verbose=False, api_key="dummy_key")

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

    def test_get_swaps_without_summary(self):
        result = pd.DataFrame(
            self.api_client.get_swaps(
                swaps=self.aligned_transactions,
                add_summary=False
            )
        )

        expected_result = pd.read_parquet(
            "tests/test_data/expected_get_swaps.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_swaps_with_summary(self):
        result = pd.DataFrame(
            self.api_client.get_swaps(
                swaps=self.aligned_transactions, add_summary=True
            )
        )

        expected_result = pd.read_parquet(
            "tests/test_data/expected_get_swaps_summary.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps(self, mock_transactions):
        mock_transactions.return_value = self.aligned_transactions

        result = self.api_client.get_account_swaps(
            wallet="0x1",
            coin_name=False,
            add_summary=False,
        )

        expected_result = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps_with_summary(self, mock_get):
        mock_transactions = MagicMock()
        mock_transactions.return_value = self.aligned_transactions
        mock_get.return_value = mock_transactions()

        result = self.api_client.get_account_swaps(
            wallet="0x1",
            coin_name=False,
            add_summary=True,
        )

        expected_result = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result_summary.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps_with_coin_name(self, mock_get):
        mock_transactions = MagicMock()
        mock_transactions.return_value = self.aligned_transactions
        mock_get.return_value = mock_transactions()

        result = self.api_client.get_account_swaps(
            wallet="0x1",
            coin_name=True,
            add_summary=False,
        )

        expected_result = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result_coin_name.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps_with_coin_name_and_summary(self, mock_get):
        mock_transactions = MagicMock()
        mock_transactions.return_value = self.aligned_transactions
        mock_get.return_value = mock_transactions()

        result = self.api_client.get_account_swaps(
            wallet="0x1",
            coin_name=True,
            add_summary=True,
        )

        expected_result = pd.read_parquet(
            "tests/test_data/account_swaps_expected_result_summary_coin_name.parquet"
        )

        pd.testing.assert_frame_equal(result, expected_result)

    def test_process_transaction_data_two_data(self):
        data = [
            {
                "from_address": "0x1",
                "to_address": "0x2",
                "value": 1,
                "block_timestamp": 1,
                "transaction_hash": "0x3",
                "token_name": "ETH",
                "token_symbol": "ETH",
            },
            {
                "from_address": "0x2",
                "to_address": "0x1",
                "value": 1,
                "block_timestamp": 2,
                "transaction_hash": "0x4",
                "token_name": "ETH",
                "token_symbol": "ETH",
            },
        ]

        result = self.api_client.process_transaction_data(data)
        self.assertListEqual(result, data)

    def test_process_transaction_data_one_data(self):
        data = [
            {
                "from_address": "0x1",
                "to_address": "0x2",
                "value": 1,
                "block_timestamp": 1,
                "transaction_hash": "0x3",
                "token_name": "ETH",
                "token_symbol": "ETH",
            },
        ]

        # check if when run the code below returns raise ValueError with message: "data has less than 2 elements"
        with self.assertRaises(ValueError) as context:
            self.api_client.process_transaction_data(data)

        self.assertEqual(str(context.exception), "data has less than 2 elements")

    def test_process_transaction_data_np_array_data(self):
        data = np.array([
            {
                "from_address": "0x1",
                "to_address": "0x2",
                "value": 1,
                "block_timestamp": 1,
                "transaction_hash": "0x3",
                "token_name": "ETH",
                "token_symbol": "ETH",
            },
            {
                "from_address": "0x2",
                "to_address": "0x1",
                "value": 1,
                "block_timestamp": 2,
                "transaction_hash": "0x4",
                "token_name": "ETH",
                "token_symbol": "ETH",
            },
        ])

        result = self.api_client.process_transaction_data(data)
        self.assertListEqual(result, data.tolist())

    def test_process_transaction_data_pd_series(self):
        data = pd.Series(
            [
                {
                    "from_address": "0x1",
                    "to_address": "0x2",
                    "value": 1,
                    "block_timestamp": 1,
                    "transaction_hash": "0x3",
                    "token_name": "ETH",
                    "token_symbol": "ETH",
                },
                {
                    "from_address": "0x2",
                    "to_address": "0x1",
                    "value": 1,
                    "block_timestamp": 2,
                    "transaction_hash": "0x4",
                    "token_name": "ETH",
                    "token_symbol": "ETH",
                },
            ]
        )

        result = self.api_client.process_transaction_data(data)
        self.assertListEqual(result, data.tolist())

    def test_process_transaction_data_three_data(self):
        data = [
            {
                "from_address": "0x1",
                "to_address": "0x2",
                "value": 1,
                "block_timestamp": 1,
                "transaction_hash": "0x3",
                "token_name": "ETH",
                "token_symbol": "ETH",
                "value": 1305,
                "value_formatted": 1.305,
                "direction": "send",
            },
            {
                "from_address": "0x2",
                "to_address": "0x1",
                "value": 1,
                "block_timestamp": 2,
                "transaction_hash": "0x4",
                "token_name": "ETH",
                "token_symbol": "ETH",
                "value": 1305,
                "value_formatted": 1.305,
                "direction": "send",
            },
            {
                "from_address": "0x2",
                "to_address": "0x1",
                "block_timestamp": 3,
                "transaction_hash": "0x5",
                "token_name": "ETH",
                "token_symbol": "ETH",
                "value": 2610,
                "value_formatted": 2.610,
                "direction": "receive",
            },
        ]

        expected_result = [
            {
                "from_address": "0x1",
                "to_address": "0x2",
                "value": 2610,
                "block_timestamp": 1,
                "transaction_hash": "0x3",
                "token_name": "ETH",
                "token_symbol": "ETH",
                "value_formatted": 2.610,
                "direction": "send",
            },
            {
                "from_address": "0x2",
                "to_address": "0x1",
                "value": 2610,
                "block_timestamp": 3,
                "transaction_hash": "0x5",
                "token_name": "ETH",
                "token_symbol": "ETH",
                "value_formatted": 2.610,
                "direction": "receive",
            },
        ]

        result = self.api_client.process_transaction_data(data)
        self.assertListEqual(result, expected_result)

    @patch(
        "crypto_explorer.api.moralis_api.MoralisAPI.fetch_wallet_token_balances"
    )
    def test_get_wallet_token_balances(self, mock_balance):
        mock_balance.return_value = self.dummy_balances[0]

        result = self.api_client.get_wallet_token_balances(
            wallet_address="0x1", block_number=5_900_000
        )

        expected_result = pd.DataFrame(
            [5.88430469, 326940.373369],
            index=pd.Index(["WBTC", "USDT"], name="symbol"),
            columns=[str(5_900_000)],
        )

        expected_result.index.name = "symbol"

        pd.testing.assert_frame_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_wallet_token_balances")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_block")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_token_price")
    def test_get_wallet_token_balances_history(
        self,
        mock_price,
        mock_block,
        mock_balance,
        mock_transactions,
    ):
        transactions = self.transactions

        rng = np.random.default_rng(42)
        transactions['block_number'] = rng.integers(low=100_000, high=150_000, size=len(transactions))

        aligned_transactions = [
            transactions.iloc[column].to_dict()
            for column in range(transactions.shape[0])
        ]

        mock_price.return_value = {
            "usdPrice": 100_000,
            "blockTimestamp": 1741827878000,
        }

        mock_block.return_value = {"block" : 500_000}

        mock_balance.side_effect = cycle(self.dummy_balances)
        mock_transactions.return_value = aligned_transactions

        result = self.api_client.get_wallet_token_balances_history(
            wallet_address="0x1", token_address="0x2"
        )

        expected_result = pd.DataFrame(
            [
                {
                    "WBTC": 5.884305,
                    "USDT": 326940.373369,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 2.050381,
                    "USDT": 148.035889,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 8.218896,
                    "USDT": 62523.502209,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 8.438703,
                    "USDT": 2565.726409,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 5.710890,
                    "USDT": 1544.804416,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 5.182523,
                    "USDT": 4096.000000,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 9.808711,
                    "USDT": 3010.936384,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 1.660195,
                    "USDT": 243087.455521,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 9.496194,
                    "USDT": 113.379904,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 6.045387,
                    "USDT": 15625.000000,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
                {
                    "WBTC": 5.884305,
                    "USDT": 326940.373369,
                    "usdPrice": 100000,
                    "blockTimestamp": pd.Timestamp("2025-03-13 01:04:38"),
                },
            ],
            pd.Index(
                [
                    "104462",
                    "138697",
                    "132728",
                    "121943",
                    "121650",
                    "142929",
                    "104297",
                    "134868",
                    "110073",
                    "104708",
                    "500000",
                ],
            ),
        )

        # Ensure the blockTimestamp column uses the same dtype as in the
        # result, preventing potential dtype mismatch issues across
        # different pandas versions.
        expected_result["blockTimestamp"] = expected_result[
            "blockTimestamp"
        ].astype(result['blockTimestamp'].dtype) 

        # Ensure the blockTimestamp column uses the same dtype as in the
        # result, preventing potential dtype mismatch issues across
        # different pandas versions.
        expected_result["blockTimestamp"] = expected_result[
            "blockTimestamp"
        ].astype(result['blockTimestamp'].dtype) 

        expected_result.columns.name = "symbol"

        pd.testing.assert_frame_equal(
            result, expected_result,
        )

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_block")
    def test_fetch_unpaginated_transactions(
        self,
        mock_block,
        mock_transactions,
    ):
        transactions_df = pd.DataFrame(self.aligned_transactions)

        transactions_df["block_number"] = np.random.default_rng(33).integers(
            10_000_000, 100_000_000, size=len(self.aligned_transactions)
        )

        transactions_df["block_number"] = (
            transactions_df["block_number"].astype(int)
        )

        mock_transactions.return_value = transactions_df
        mock_block.return_value = {"block": 70507977}

        result = self.api_client.fetch_unpaginated_transactions(
            wallet_address="0x1",
        )

        expected_results = [
            89104395,
            49927801,
            44202656,
            61164207,
            84777770,
            91729339,
            36996145,
            32882459,
            57736195,
            62990314,
            70507977,
        ]

        self.assertListEqual(result, expected_results)

class TestFetchPaginatedTransactions(unittest.TestCase):
    def setUp(self):
        self.api_client = MoralisAPI(verbose=False, api_key="dummy_key")

    @patch.object(MoralisAPI, "fetch_transactions")
    def test_fetch_paginated_transactions_normal(self, mock_fetch):
        mock_fetch.side_effect = [[{"block_number": 1}], [{"block_number": 2}], []]
        initial_block = 0
        final_block = 2_000_000

        result = self.api_client.fetch_paginated_transactions(
            wallet_address="0xabc",
            initial_block=initial_block,
            final_block=final_block,
        )

        self.assertListEqual(result, [{"block_number": 1}, {"block_number": 2}])

        call_args = [call.kwargs for call in mock_fetch.call_args_list]

        self.assertEqual(call_args[0]["from_block"], 0)
        self.assertEqual(call_args[0]["to_block"], 0)

        self.assertEqual(call_args[1]["from_block"], 1)
        self.assertEqual(call_args[1]["to_block"], 1_000_000)

        self.assertEqual(call_args[2]["from_block"], 1_000_001)
        self.assertEqual(call_args[2]["to_block"], 2_000_000)

class TestGetWalletBlocks(unittest.TestCase):
    def setUp(self):
        self.api_client = MoralisAPI(verbose=False, api_key="dummy_key")

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_paginated_transactions")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_block")
    def test_get_wallet_blocks_with_from_block(self, mock_fetch_block, mock_fetch_paginated):
        mock_fetch_paginated.return_value = [{"block_number": 100}, {"block_number": 200}]
        mock_fetch_block.return_value = {"block": 300}

        result = self.api_client.get_wallet_blocks(
            wallet_address="0x123", from_block=1, to_block=250
        )

        mock_fetch_paginated.assert_called_once_with(
            wallet_address="0x123", initial_block=1, final_block=250, from_block=1, to_block=250
        )
        mock_fetch_block.assert_called_once_with("now")
        self.assertEqual(result, [100, 200, 300])

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_unpaginated_transactions")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_block")
    def test_get_wallet_blocks_without_from_block(self, mock_fetch_block, mock_fetch_unpaginated):
        mock_fetch_unpaginated.return_value = [{"block_number": 400}, {"block_number": 500}]
        mock_fetch_block.return_value = {"block": 600}

        result = self.api_client.get_wallet_blocks(wallet_address="0x123")

        mock_fetch_unpaginated.assert_called_once_with(wallet_address="0x123")
        mock_fetch_block.assert_called_once_with("now")
        self.assertEqual(result, [400, 500, 600])

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_paginated_transactions")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_block")
    def test_get_wallet_blocks_with_missing_to_block(self, mock_fetch_block, mock_fetch_paginated):
        mock_fetch_paginated.return_value = [{"block_number": 700}]
        mock_fetch_block.return_value = {"block": 800}

        result = self.api_client.get_wallet_blocks(
            wallet_address="0x123", from_block=1
        )

        mock_fetch_paginated.assert_called_once_with(
            wallet_address="0x123", initial_block=1, final_block=800, from_block=1, to_block=800
        )
        # Called twice: once for the missing to_block, once for the last_block
        self.assertEqual(mock_fetch_block.call_count, 2)
        self.assertEqual(result, [700, 800])
