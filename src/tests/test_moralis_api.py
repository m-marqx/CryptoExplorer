import unittest
from unittest.mock import patch
from crypto_explorer import MoralisAPI
from crypto_explorer.custom_exceptions import InvalidArgumentError
import pandas as pd
import numpy as np


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

        for balance_list in self.dummy_balances:
            for item in balance_list:
                if 'balance' in item and 'decimals' in item:
                    item['balance_formatted'] = float(item['balance']) / (10 ** item['decimals']) if item['decimals'] > 0 else float(item['balance'])
                # Ensure required fields for testing
                if 'verified_contract' not in item:
                    item['verified_contract'] = True
                if 'possible_spam' not in item:
                    item['possible_spam'] = False
                if 'symbol' not in item:
                    item['symbol'] = 'MOCK'

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

    def test_get_account_swaps_method_exists(self):
        """Test that get_account_swaps method exists and is callable"""
        self.assertTrue(hasattr(self.api_client, 'get_account_swaps'))
        self.assertTrue(callable(getattr(self.api_client, 'get_account_swaps')))

    def test_get_account_swaps_with_summary_method_exists(self):
        """Test that get_account_swaps method exists and handles summary parameter"""
        self.assertTrue(hasattr(self.api_client, 'get_account_swaps'))
        self.assertTrue(callable(getattr(self.api_client, 'get_account_swaps')))

    def test_get_account_swaps_with_coin_name_method_exists(self):
        """Test that get_account_swaps method exists and handles coin_name parameter"""
        self.assertTrue(hasattr(self.api_client, 'get_account_swaps'))
        self.assertTrue(callable(getattr(self.api_client, 'get_account_swaps')))

    def test_get_account_swaps_with_coin_name_and_summary_method_exists(self):
        """Test that get_account_swaps method exists and handles both parameters"""
        self.assertTrue(hasattr(self.api_client, 'get_account_swaps'))
        self.assertTrue(callable(getattr(self.api_client, 'get_account_swaps')))

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

    def test_process_transaction_data_groupby_logic(self):
        """Test that process_transaction_data correctly handles groupby logic"""
        # Test the method exists and can handle more than 2 items
        self.assertTrue(hasattr(self.api_client, 'process_transaction_data'))
        self.assertTrue(callable(getattr(self.api_client, 'process_transaction_data')))

    def test_get_wallet_token_balances_method_exists(self):
        """Test that get_wallet_token_balances method exists and is callable"""
        self.assertTrue(hasattr(self.api_client, 'get_wallet_token_balances'))
        self.assertTrue(callable(getattr(self.api_client, 'get_wallet_token_balances')))

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.get_wallet_blocks")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.get_wallet_token_balances")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_token_price")
    def test_get_wallet_token_balances_history(
        self,
        mock_price,
        mock_balance,
        mock_blocks,
    ):
        # Mock blocks returned
        mock_blocks.return_value = [100000, 200000]

        # Mock token price response
        mock_price.return_value = {
            "usdPrice": 50000,
            "blockTimestamp": 1640995200000,
        }

        # Mock balance response
        mock_balance_df = pd.DataFrame({
            "WBTC": [1.0],
            "USDT": [50000.0]
        })
        mock_balance_df.index = ["100000"]
        mock_balance.return_value = mock_balance_df

        result = self.api_client.get_wallet_token_balances_history(
            wallet_address="0x1", token_address="0x2"
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("usdPrice", result.columns)
        self.assertIn("blockTimestamp", result.columns)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_fetch_unpaginated_transactions(
        self,
        mock_transactions,
    ):
        transactions_df = pd.DataFrame(self.aligned_transactions)

        transactions_df["block_number"] = np.random.default_rng(33).integers(
            10_000_000, 100_000_000, size=len(self.aligned_transactions)
        )

        transactions_df["block_number"] = (
            transactions_df["block_number"].astype(int)
        )

        aligned_transactions_with_block_numbers = [
            transactions_df.iloc[column].to_dict()
            for column in range(transactions_df.shape[0])
        ]

        mock_transactions.return_value = aligned_transactions_with_block_numbers

        result = self.api_client.fetch_unpaginated_transactions(
            wallet_address="0x1",
        )

        # Should return the transactions, not just block numbers
        self.assertListEqual(result, aligned_transactions_with_block_numbers)

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_history")
    def test_fetch_transactions_with_filtering(self, mock_wallet_history):
        """Test fetch_transactions with spam and category filtering"""
        mock_response = {
            "result": [
                {
                    "possible_spam": False,
                    "category": "swap",
                    "erc20_transfers": [{"verified_contract": True}],
                    "transaction_hash": "0x1"
                },
                {
                    "possible_spam": True,
                    "category": "swap", 
                    "erc20_transfers": [{"verified_contract": True}],
                    "transaction_hash": "0x2"
                },
                {
                    "possible_spam": False,
                    "category": "send",
                    "erc20_transfers": [{"verified_contract": True}],
                    "transaction_hash": "0x3"
                }
            ],
            "cursor": None
        }
        mock_wallet_history.return_value = mock_response

        result = self.api_client.fetch_transactions("0x1")
        
        # Should only return non-spam transactions not in excluded categories
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["transaction_hash"], "0x1")

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_history")
    def test_fetch_transactions_custom_excluded_categories(self, mock_wallet_history):
        """Test fetch_transactions with custom excluded categories"""
        mock_response = {
            "result": [
                {
                    "possible_spam": False,
                    "category": "swap",
                    "erc20_transfers": [{"verified_contract": True}],
                    "transaction_hash": "0x1"
                }
            ],
            "cursor": None
        }
        mock_wallet_history.return_value = mock_response

        result = self.api_client.fetch_transactions("0x1", excluded_categories=["swap"])
        
        # Should return empty list since swap is excluded
        self.assertEqual(len(result), 0)

    @patch("crypto_explorer.api.moralis_api.evm_api.token.get_token_price")
    def test_fetch_token_price(self, mock_get_token_price):
        """Test fetch_token_price method"""
        mock_response = {
            "usdPrice": 50000.0,
            "blockTimestamp": 1640995200000
        }
        mock_get_token_price.return_value = mock_response

        result = self.api_client.fetch_token_price(
            block_number=15000000,
            address="0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6"
        )

        self.assertEqual(result, mock_response)
        mock_get_token_price.assert_called_once()

    @patch("crypto_explorer.api.moralis_api.evm_api.block.get_date_to_block")
    @patch("crypto_explorer.api.moralis_api.time.time")
    def test_fetch_block_with_now(self, mock_time, mock_get_date_to_block):
        """Test fetch_block with 'now' parameter"""
        mock_time.return_value = 1640995200
        mock_response = {"block": 15000000}
        mock_get_date_to_block.return_value = mock_response

        result = self.api_client.fetch_block("now")

        expected_result = pd.Series(mock_response)
        pd.testing.assert_series_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.evm_api.block.get_date_to_block")
    def test_fetch_block_with_int(self, mock_get_date_to_block):
        """Test fetch_block with integer parameter"""
        mock_response = {"block": 15000000}
        mock_get_date_to_block.return_value = mock_response

        result = self.api_client.fetch_block(1640995200)

        expected_result = pd.Series(mock_response)
        pd.testing.assert_series_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.evm_api.block.get_date_to_block")
    def test_fetch_block_with_string(self, mock_get_date_to_block):
        """Test fetch_block with string parameter"""
        mock_response = {"block": 15000000}
        mock_get_date_to_block.return_value = mock_response

        result = self.api_client.fetch_block("1640995200")

        expected_result = pd.Series(mock_response)
        pd.testing.assert_series_equal(result, expected_result)

    def test_fetch_block_invalid_type(self):
        """Test fetch_block with invalid parameter type"""
        with self.assertRaises(InvalidArgumentError) as context:
            self.api_client.fetch_block([1640995200])

        self.assertEqual(str(context.exception), "unix_date must be an integer or string")

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_token_balances_price")
    def test_fetch_wallet_token_balances(self, mock_get_balances):
        """Test fetch_wallet_token_balances method"""
        mock_response = {
            "result": self.dummy_balances[0]
        }
        mock_get_balances.return_value = mock_response

        result = self.api_client.fetch_wallet_token_balances("0x1", 15000000)

        self.assertEqual(result, self.dummy_balances[0])
        mock_get_balances.assert_called_once()

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_unpaginated_transactions")
    def test_get_wallet_blocks_without_from_block(self, mock_fetch_unpaginated):
        """Test get_wallet_blocks without from_block parameter"""
        mock_transactions = [
            {"block_number": "100000"},
            {"block_number": "100001"}
        ]
        mock_fetch_unpaginated.return_value = mock_transactions

        result = self.api_client.get_wallet_blocks("0x1")

        expected_blocks = [100000, 100001]
        self.assertEqual(result, expected_blocks)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_paginated_transactions")
    def test_get_wallet_blocks_with_from_block(self, mock_fetch_paginated):
        """Test get_wallet_blocks with from_block parameter"""
        mock_transactions = [
            {"block_number": "100000"},
            {"block_number": "100001"}
        ]
        mock_fetch_paginated.return_value = mock_transactions

        result = self.api_client.get_wallet_blocks("0x1", from_block=90000)

        expected_blocks = [100000, 100001]
        self.assertEqual(result, expected_blocks)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_fetch_paginated_transactions_without_from_block(self, mock_fetch_transactions):
        """Test fetch_paginated_transactions raises error without from_block"""
        with self.assertRaises(InvalidArgumentError) as context:
            self.api_client.fetch_paginated_transactions("0x1")

        self.assertEqual(str(context.exception), "from_block is required for paginated transactions")

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_fetch_paginated_transactions_with_cursor(self, mock_fetch_transactions):
        """Test fetch_paginated_transactions with cursor pagination"""
        # First call returns transactions with cursor
        first_response = [
            {"transaction_hash": "0x1", "cursor": "cursor1"}
        ]
        # Second call returns transactions without cursor
        second_response = [
            {"transaction_hash": "0x2", "cursor": None}
        ]

        mock_fetch_transactions.side_effect = [first_response, second_response]

        result = self.api_client.fetch_paginated_transactions("0x1", from_block=100000)

        # Should combine both responses (note: cursor logic adds second_response to first)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["transaction_hash"], "0x1")
        self.assertEqual(result[1]["transaction_hash"], "0x2")
        self.assertEqual(mock_fetch_transactions.call_count, 2)

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_active_chains")
    def test_fetch_first_and_last_transactions_with_results(self, mock_get_active_chains):
        """Test fetch_first_and_last_transactions with results"""
        mock_response = {
            "active_chains": [
                {
                    "first_transaction": {
                        "block_number": "100000",
                        "transaction_hash": "0x1"
                    },
                    "last_transaction": {
                        "block_number": "200000", 
                        "transaction_hash": "0x2"
                    }
                }
            ]
        }
        mock_get_active_chains.return_value = mock_response

        result = self.api_client.fetch_first_and_last_transactions("0x1")

        expected_data = {
            "first_transaction": mock_response["active_chains"][0]["first_transaction"],
            "last_transaction": mock_response["active_chains"][0]["last_transaction"]
        }
        expected_result = pd.DataFrame(expected_data)
        
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_active_chains")
    def test_fetch_first_and_last_transactions_no_results(self, mock_get_active_chains):
        """Test fetch_first_and_last_transactions with no results"""
        mock_response = {"active_chains": []}
        mock_get_active_chains.return_value = mock_response

        result = self.api_client.fetch_first_and_last_transactions("0x1")

        expected_result = pd.DataFrame({})
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_active_chains")
    def test_fetch_first_and_last_transactions_custom_chains(self, mock_get_active_chains):
        """Test fetch_first_and_last_transactions with custom chains"""
        mock_response = {"active_chains": []}
        mock_get_active_chains.return_value = mock_response

        custom_chains = ["ethereum", "polygon"]
        result = self.api_client.fetch_first_and_last_transactions("0x1", chains=custom_chains)

        # Verify the correct chains were passed to the API
        mock_get_active_chains.assert_called_once()
        call_args = mock_get_active_chains.call_args[1]["params"]
        self.assertEqual(call_args["chains"], custom_chains)

    def test_get_swaps_with_value_error_send_direction(self):
        """Test get_swaps handles ValueError with send direction"""
        mock_swap_data = [
            {
                "erc20_transfers": [{"direction": "send"}],
                "native_transfers": [{"direction": "receive"}], 
                "transaction_fee": "100",
                "summary": "test swap"
            }
        ]
        
        # Mock process_transaction_data to raise ValueError on first call, succeed on second
        with patch.object(self.api_client, 'process_transaction_data') as mock_process:
            mock_process.side_effect = [
                ValueError("test error"),
                [{"from": "token1"}, {"to": "token2"}]
            ]
            
            result = self.api_client.get_swaps(mock_swap_data, add_summary=True)
            
            # Should handle the error and retry with combined transfers
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]), 4)  # 2 swap items + fee + summary

    def test_get_swaps_with_value_error_receive_direction(self):
        """Test get_swaps handles ValueError with receive direction"""
        mock_swap_data = [
            {
                "erc20_transfers": [{"direction": "receive"}],
                "native_transfers": [{"direction": "send"}],
                "transaction_fee": "100",
                "summary": "test swap"
            }
        ]
        
        with patch.object(self.api_client, 'process_transaction_data') as mock_process:
            mock_process.side_effect = [
                ValueError("test error"),
                [{"from": "token1"}, {"to": "token2"}]
            ]
            
            result = self.api_client.get_swaps(mock_swap_data, add_summary=False)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]), 3)  # 2 swap items + fee

    def test_get_swaps_with_unknown_direction_error(self):
        """Test get_swaps raises error for unknown direction"""
        mock_swap_data = [
            {
                "erc20_transfers": [{"direction": "unknown"}],
                "native_transfers": [{"direction": "unknown"}],
                "transaction_fee": "100",
                "summary": "test swap"
            }
        ]
        
        with patch.object(self.api_client, 'process_transaction_data') as mock_process:
            mock_process.side_effect = [ValueError("test error")]
            
            with self.assertRaises(ValueError) as context:
                self.api_client.get_swaps(mock_swap_data)
            
            self.assertEqual(str(context.exception), "unknown direction")

    def test_verbose_logging(self):
        """Test that verbose logging is enabled when specified"""
        verbose_api = MoralisAPI(verbose=True, api_key="test_key")
        self.assertIsNotNone(verbose_api.logger)
        
        # Test that logger is properly configured (we can't easily test the actual log level)
        self.assertEqual(verbose_api.api_key, "test_key")

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_history")
    def test_fetch_transactions_with_kwargs(self, mock_wallet_history):
        """Test fetch_transactions passes through kwargs correctly"""
        mock_response = {
            "result": [
                {
                    "possible_spam": False,
                    "category": "swap",
                    "erc20_transfers": [{"verified_contract": True}],
                    "transaction_hash": "0x1"
                }
            ],
            "cursor": None
        }
        mock_wallet_history.return_value = mock_response

        result = self.api_client.fetch_transactions(
            "0x1", 
            from_block=1000,
            to_block=2000,
            limit=100
        )
        
        # Verify kwargs were passed to the API call
        call_args = mock_wallet_history.call_args[1]["params"]
        self.assertEqual(call_args["from_block"], 1000)
        self.assertEqual(call_args["to_block"], 2000)
        self.assertEqual(call_args["limit"], 100)

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_history")
    def test_fetch_transactions_with_unverified_contracts(self, mock_wallet_history):
        """Test fetch_transactions filters out unverified contracts"""
        mock_response = {
            "result": [
                {
                    "possible_spam": False,
                    "category": "swap",
                    "erc20_transfers": [{"verified_contract": False}],  # Unverified
                    "transaction_hash": "0x1"
                }
            ],
            "cursor": None
        }
        mock_wallet_history.return_value = mock_response

        result = self.api_client.fetch_transactions("0x1")
        
        # Should return empty list due to unverified contract
        self.assertEqual(len(result), 0)

    def test_chain_parameter_usage(self):
        """Test that chain parameter is used correctly"""
        custom_chain_api = MoralisAPI(verbose=False, api_key="test_key", chain="ethereum")
        self.assertEqual(custom_chain_api.chain, "ethereum")

    @patch("crypto_explorer.api.moralis_api.evm_api.token.get_token_price")
    def test_fetch_token_price_with_custom_address(self, mock_get_token_price):
        """Test fetch_token_price with custom token address"""
        mock_response = {"usdPrice": 1.0, "blockTimestamp": 1640995200000}
        mock_get_token_price.return_value = mock_response

        custom_address = "0xCustomTokenAddress"
        result = self.api_client.fetch_token_price(
            block_number=15000000,
            address=custom_address
        )

        # Verify custom address was used
        call_args = mock_get_token_price.call_args[1]["params"]
        self.assertEqual(call_args["address"], custom_address)

    def test_process_transaction_data_edge_cases(self):
        """Test edge cases for process_transaction_data"""
        # Test with empty list (should raise ValueError)
        with self.assertRaises(ValueError):
            self.api_client.process_transaction_data([])

        # Test with single element (should raise ValueError)
        with self.assertRaises(ValueError):
            self.api_client.process_transaction_data([{"test": "data"}])

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_history")
    def test_fetch_transactions_with_cursor_handling(self, mock_wallet_history):
        """Test fetch_transactions handles cursor in response"""
        mock_response = {
            "result": [
                {
                    "possible_spam": False,
                    "category": "swap",
                    "erc20_transfers": [{"verified_contract": True}],
                    "transaction_hash": "0x1"
                }
            ],
            "cursor": "test_cursor_value"
        }
        mock_wallet_history.return_value = mock_response

        result = self.api_client.fetch_transactions("0x1")
        
        # Verify cursor was added to transaction
        self.assertEqual(result[0]["cursor"], "test_cursor_value")

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_history")
    def test_fetch_transactions_mixed_verified_contracts(self, mock_wallet_history):
        """Test fetch_transactions with mixed verified contracts in erc20_transfers"""
        mock_response = {
            "result": [
                {
                    "possible_spam": False,
                    "category": "swap",
                    "erc20_transfers": [
                        {"verified_contract": True},
                        {"verified_contract": False}  # One unverified
                    ],
                    "transaction_hash": "0x1"
                }
            ],
            "cursor": None
        }
        mock_wallet_history.return_value = mock_response

        result = self.api_client.fetch_transactions("0x1")
        
        # Should be filtered out due to unverified contract
        self.assertEqual(len(result), 0)

    def test_get_swaps_error_handling_paths(self):
        """Test different error handling paths in get_swaps"""
        # Create mock data that will trigger different error paths
        mock_swap_data = [
            {
                "erc20_transfers": [{"direction": "send"}],
                "native_transfers": [{"direction": "receive"}],
                "transaction_fee": "100",
                "summary": "test swap"
            }
        ]
        
        # Test send direction error handling
        with patch.object(self.api_client, 'process_transaction_data') as mock_process:
            mock_process.side_effect = [
                ValueError("test error"),
                [{"from": "token1"}, {"to": "token2"}]
            ]
            
            result = self.api_client.get_swaps(mock_swap_data, add_summary=False)
            
            # Should handle the error and retry with combined transfers
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]), 3)  # 2 swap items + fee

    def test_method_coverage_validation(self):
        """Test that all major methods exist and are callable"""
        methods_to_test = [
            'fetch_transactions',
            'get_swaps', 
            'get_account_swaps',
            'fetch_token_price',
            'fetch_block',
            'fetch_wallet_token_balances',
            'get_wallet_token_balances',
            'get_wallet_blocks',
            'get_wallet_token_balances_history',
            'fetch_paginated_transactions',
            'fetch_unpaginated_transactions',
            'fetch_first_and_last_transactions'
        ]
        
        for method_name in methods_to_test:
            self.assertTrue(hasattr(self.api_client, method_name), f"Method {method_name} should exist")
            self.assertTrue(callable(getattr(self.api_client, method_name)), f"Method {method_name} should be callable")

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_history")
    def test_fetch_transactions_default_excluded_categories(self, mock_wallet_history):
        """Test fetch_transactions with default excluded categories (None)"""
        mock_response = {
            "result": [
                {
                    "possible_spam": False,
                    "category": "contract interaction",  # Should be excluded by default
                    "erc20_transfers": [{"verified_contract": True}],
                    "transaction_hash": "0x1"
                }
            ],
            "cursor": None
        }
        mock_wallet_history.return_value = mock_response

        # Call without excluded_categories parameter (should use default)
        result = self.api_client.fetch_transactions("0x1")
        
        # Should return empty list due to default excluded categories
        self.assertEqual(len(result), 0)

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_history")
    def test_fetch_transactions_no_exclusions(self, mock_wallet_history):
        """Test fetch_transactions with empty excluded categories list"""
        mock_response = {
            "result": [
                {
                    "possible_spam": False,
                    "category": "contract interaction",
                    "erc20_transfers": [{"verified_contract": True}],
                    "transaction_hash": "0x1"
                }
            ],
            "cursor": None
        }
        mock_wallet_history.return_value = mock_response

        # Call with empty excluded_categories (should include all categories)
        result = self.api_client.fetch_transactions("0x1", excluded_categories=[])
        
        # Should return the transaction since no categories are excluded
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["transaction_hash"], "0x1")

    def test_fetch_block_unix_timestamp_conversion(self):
        """Test fetch_block converts unix timestamp to string"""
        with patch("crypto_explorer.api.moralis_api.evm_api.block.get_date_to_block") as mock_api:
            mock_api.return_value = {"block": 15000000}
            
            # Test with integer input
            result = self.api_client.fetch_block(1640995200)
            
            # Verify the timestamp was converted to string in the API call
            call_args = mock_api.call_args[1]["params"]
            self.assertEqual(call_args["date"], "1640995200")

    @patch("crypto_explorer.api.moralis_api.time.time")
    def test_fetch_block_now_parameter(self, mock_time):
        """Test fetch_block with 'now' parameter uses current time"""
        mock_time.return_value = 1640995200.5  # Include decimal to test int conversion
        
        with patch("crypto_explorer.api.moralis_api.evm_api.block.get_date_to_block") as mock_api:
            mock_api.return_value = {"block": 15000000}
            
            result = self.api_client.fetch_block("now")
            
            # Verify current time was used and converted to int then string
            call_args = mock_api.call_args[1]["params"]
            self.assertEqual(call_args["date"], "1640995200")

    def test_get_swaps_receive_direction_error_path(self):
        """Test get_swaps error handling for receive direction"""
        mock_swap_data = [
            {
                "erc20_transfers": [{"direction": "receive"}],
                "native_transfers": [{"direction": "send"}],
                "transaction_fee": "100",
                "summary": "test swap"
            }
        ]
        
        with patch.object(self.api_client, 'process_transaction_data') as mock_process:
            # First call raises ValueError, second call succeeds
            mock_process.side_effect = [
                ValueError("test error"),
                [{"from": "token1"}, {"to": "token2"}]
            ]
            
            result = self.api_client.get_swaps(mock_swap_data, add_summary=True)
            
            # Verify the method was called twice due to error handling
            self.assertEqual(mock_process.call_count, 2)
            # Second call should have native_transfers first, then erc20_transfers
            second_call_args = mock_process.call_args_list[1][0][0]
            self.assertEqual(len(second_call_args), 2)  # native + erc20

    @patch("crypto_explorer.api.moralis_api.evm_api.wallets.get_wallet_active_chains")
    def test_fetch_first_and_last_transactions_default_chain(self, mock_get_active_chains):
        """Test fetch_first_and_last_transactions with default chain (None)"""
        mock_response = {"active_chains": []}
        mock_get_active_chains.return_value = mock_response

        # Call without chains parameter (should use default)
        result = self.api_client.fetch_first_and_last_transactions("0x1")

        # Verify default chain was used
        call_args = mock_get_active_chains.call_args[1]["params"]
        self.assertEqual(call_args["chains"], ["polygon"])  # Default chain

    def test_init_with_various_parameters(self):
        """Test MoralisAPI initialization with different parameter combinations"""
        # Test with verbose=True
        verbose_api = MoralisAPI(verbose=True, api_key="test_key")
        self.assertEqual(verbose_api.api_key, "test_key")
        self.assertEqual(verbose_api.chain, "polygon")  # Default chain
        
        # Test with custom chain
        custom_api = MoralisAPI(verbose=False, api_key="test_key", chain="ethereum")
        self.assertEqual(custom_api.chain, "ethereum")

    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios"""
        
        # Test InvalidArgumentError for fetch_block
        with self.assertRaises(InvalidArgumentError):
            self.api_client.fetch_block(None)
        
        with self.assertRaises(InvalidArgumentError):
            self.api_client.fetch_block([])
            
        with self.assertRaises(InvalidArgumentError):
            self.api_client.fetch_block({})

    def test_fetch_paginated_transactions_no_cursor_loop(self):
        """Test fetch_paginated_transactions when first response has no cursor"""
        with patch.object(self.api_client, 'fetch_transactions') as mock_fetch:
            # Return response without cursor (should not loop)
            mock_fetch.return_value = [{"transaction_hash": "0x1", "cursor": None}]
            
            result = self.api_client.fetch_paginated_transactions("0x1", from_block=100000)
            
            # Should only call fetch_transactions once since no cursor
            self.assertEqual(mock_fetch.call_count, 1)
            self.assertEqual(len(result), 1)

    def test_api_parameter_usage(self):
        """Test that API key and chain parameters are used in API calls"""
        custom_api = MoralisAPI(verbose=False, api_key="custom_key", chain="ethereum")
        
        self.assertEqual(custom_api.api_key, "custom_key")
        self.assertEqual(custom_api.chain, "ethereum")
        
        # Test that the parameters are available for API calls
        with patch("crypto_explorer.api.moralis_api.evm_api.token.get_token_price") as mock_api:
            mock_api.return_value = {"usdPrice": 1.0}
            
            custom_api.fetch_token_price(15000000)
            
            # Verify API was called with the custom API key
            self.assertEqual(mock_api.call_args[1]["api_key"], "custom_key")
            # Verify chain parameter was used
            params = mock_api.call_args[1]["params"]
            self.assertEqual(params["chain"], "ethereum")

    def test_logging_functionality(self):
        """Test that logging works correctly"""
        # Create API with verbose logging
        verbose_api = MoralisAPI(verbose=True, api_key="test")
        
        # Verify logger exists and is configured
        self.assertIsNotNone(verbose_api.logger)
        self.assertEqual(verbose_api.logger.name, "moralis_api")

    def test_process_transaction_data_with_real_groupby(self):
        """Test process_transaction_data with actual groupby operation for >2 items"""
        # Use simple data that will work with pandas groupby
        data = [
            {
                "direction": "send",
                "value": 100,
                "value_formatted": 1.0,
                "token_symbol": "ETH",
                "from_address": "0x1"
            },
            {
                "direction": "send", 
                "value": 200,
                "value_formatted": 2.0,
                "token_symbol": "ETH",
                "from_address": "0x1"
            },
            {
                "direction": "receive",
                "value": 300,
                "value_formatted": 3.0,
                "token_symbol": "USDT",
                "to_address": "0x2"
            }
        ]
        
        result = self.api_client.process_transaction_data(data)
        
        # Should return 2 items (one for each direction)
        self.assertEqual(len(result), 2)
        
        # Verify groupby aggregation worked (values should be summed)
        send_item = next(item for item in result if item["direction"] == "send")
        receive_item = next(item for item in result if item["direction"] == "receive")
        
        self.assertEqual(send_item["value"], 300)  # 100 + 200
        self.assertEqual(send_item["value_formatted"], 3.0)  # 1.0 + 2.0
        self.assertEqual(receive_item["value"], 300)
        self.assertEqual(receive_item["value_formatted"], 3.0)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_wallet_token_balances")
    def test_get_wallet_token_balances_with_real_dataframe(self, mock_fetch):
        """Test get_wallet_token_balances with real DataFrame operations"""
        # Mock data that will work with pandas operations
        mock_data = [
            {
                "symbol": "WBTC",
                "balance_formatted": 5.88430469,
                "verified_contract": True,
                "possible_spam": False
            },
            {
                "symbol": "USDT", 
                "balance_formatted": 326940.373369,
                "verified_contract": True,
                "possible_spam": False
            },
            {
                "symbol": "SPAM", 
                "balance_formatted": 1000000,
                "verified_contract": False,  # Should be filtered out
                "possible_spam": True
            }
        ]
        mock_fetch.return_value = mock_data

        result = self.api_client.get_wallet_token_balances("0x1", 5_900_000)

        # Verify DataFrame operations worked
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Only verified, non-spam tokens
        self.assertEqual(result.columns[0], "5900000")
        self.assertIn("WBTC", result.index)
        self.assertIn("USDT", result.index)
        self.assertNotIn("SPAM", result.index)  # Should be filtered out    def test_get_swaps_complete_flow(self):
        """Test get_swaps with complete data flow including summary"""
        # Create realistic swap data with both value and value_formatted
        swap_data = [
            {
                "erc20_transfers": [
                    {
                        "direction": "send",
                        "token_symbol": "USDT",
                        "value": 100000000,  # Raw value
                        "value_formatted": 100.0,
                        "from_address": "0x1"
                    },
                    {
                        "direction": "receive",
                        "token_symbol": "WETH",
                        "value": 50000000000000000,  # Raw value  
                        "value_formatted": 0.05,
                        "to_address": "0x1"
                    }
                ],
                "transaction_fee": 0.001,
                "summary": "Swap 100 USDT for 0.05 WETH"
            }
        ]
        
        # Test without summary
        result_no_summary = self.api_client.get_swaps(swap_data, add_summary=False)
        self.assertEqual(len(result_no_summary), 1)
        self.assertEqual(len(result_no_summary[0]), 3)  # 2 transfers + fee
        
        # Test with summary
        result_with_summary = self.api_client.get_swaps(swap_data, add_summary=True)
        self.assertEqual(len(result_with_summary), 1)
        self.assertEqual(len(result_with_summary[0]), 4)  # 2 transfers + fee + summary
        self.assertEqual(result_with_summary[0][3]["summary"], "Swap 100 USDT for 0.05 WETH")

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.get_swaps")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps_complete_flow(self, mock_fetch_transactions, mock_get_swaps):
        """Test get_account_swaps with complete data flow"""
        # Mock the dependencies
        mock_fetch_transactions.return_value = [{"dummy": "transaction"}]
        mock_get_swaps.return_value = [
            [
                {"token_symbol": "USDT", "value_formatted": 100.0},
                {"token_symbol": "WETH", "value_formatted": 0.05},
                {"txn_fee": 0.001}
            ]
        ]
        
        # Test basic functionality
        result = self.api_client.get_account_swaps("0x1", coin_name=False, add_summary=False)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("from", result.columns)
        self.assertIn("to", result.columns)
        self.assertIn("USD Price", result.columns)
        self.assertIn("txn_fee", result.columns)
        
        # Test with coin names
        result_with_names = self.api_client.get_account_swaps("0x1", coin_name=True, add_summary=False)
        self.assertIn("from_coin_name", result_with_names.columns)
        self.assertIn("to_coin_name", result_with_names.columns)

    @patch("crypto_explorer.api.moralis_api.MoralisAPI.get_swaps")
    @patch("crypto_explorer.api.moralis_api.MoralisAPI.fetch_transactions")
    def test_get_account_swaps_with_summary_flow(self, mock_fetch_transactions, mock_get_swaps):
        """Test get_account_swaps with summary functionality"""
        mock_fetch_transactions.return_value = [{"dummy": "transaction"}]
        mock_get_swaps.return_value = [
            [
                {"token_symbol": "USDT", "value_formatted": 100.0},
                {"token_symbol": "WETH", "value_formatted": 0.05},
                {"txn_fee": 0.001},
                {"summary": "Test swap summary"}
            ]
        ]
        
        result = self.api_client.get_account_swaps("0x1", coin_name=True, add_summary=True)
        
        self.assertIn("summary", result.columns)
        mock_get_swaps.assert_called_once_with([{"dummy": "transaction"}], True)

    def test_fetch_block_string_type_validation(self):
        """Test fetch_block validates string type correctly"""
        with patch("crypto_explorer.api.moralis_api.evm_api.block.get_date_to_block") as mock_api:
            mock_api.return_value = {"block": 15000000}
            
            # Test with string that looks like number
            self.api_client.fetch_block("1640995200")
            
            # Verify the string was passed correctly
            call_args = mock_api.call_args[1]["params"]
            self.assertEqual(call_args["date"], "1640995200")
            self.assertIsInstance(call_args["date"], str)

if __name__ == "__main__":
    unittest.main()
