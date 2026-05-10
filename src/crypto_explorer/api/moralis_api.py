from typing import Literal
import time

import pandas as pd
import numpy as np

from moralis import evm_api
from crypto_explorer.custom_exceptions import InvalidArgumentError
from crypto_explorer.utils import create_logger

class MoralisAPI:
    """
    A class to interact with the Moralis API for retrieving transaction
    data.

    Parameters
    ----------
    verbose : bool
        If True, sets the logger to INFO level.
    api_key : str
        The API key for the Moralis API.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for logging information.
    api_key : str
        The API key for the Moralis API.

    Methods
    -------
    process_transaction_data(data)
        Processes transaction data for a given transaction.
    get_swaps(wallet)
        Retrieves all swaps for a given wallet address.
    get_swaps_data(swaps)
        Retrieves all swaps data for a given wallet address.
    get_account_swaps(wallet)
        Retrieves all swaps for a given wallet address.
    """

    def __init__(self, verbose: bool, api_key: str, chain: str = "polygon"):
        """
        Initialize the MoralisAPI object.

        Parameters
        ----------
        api_key : str
            The API key for the Moralis API.
        chain : str
            The blockchain to retrieve data for.
        logger : logging.Logger
            Logger instance for logging information.
        """
        self.api_key = api_key
        self.chain = chain
        self.logger = create_logger("moralis_api", verbose)

    def process_transaction_data(self, data: list) -> list:
        """
        Processes transaction data for a given transaction.

        Parameters
        ----------
        data : list
            The transaction data to process.

        Returns
        -------
        list
            The processed transaction data.

        Raises
        ------
        ValueError
            If the data has less than 2 elements.
        """
        if isinstance(data, (np.ndarray, pd.Series)):
            data = data.tolist()

        if len(data) == 2 and isinstance(data, list):
            return data

        if len(data) > 2:
            df = pd.DataFrame(data)
            default_columns = df.columns.tolist()
            value_columns = [
                "value",
                "value_formatted",
            ]

            df[value_columns] = df[value_columns].astype(float)
            df = df.groupby("direction").agg(
                {
                    col: "sum" if col in value_columns else "first"
                    for col in default_columns
                }
            )

            ordened_df = df.loc[["send", "receive"]][default_columns]

            return [ordened_df.iloc[x].to_dict() for x in range(df.shape[0])]

        raise ValueError("data has less than 2 elements")

    def fetch_transactions(
        self,
        wallet: str,
        excluded_categories: list | None = None,
        **kwargs,
    ) -> list:
        """
        Retrieves transaction history for the specified wallet address
        while filtering out transactions that are marked as spam or
        belong to any of the excluded categories.

        Parameters
        ----------
        wallet : str
            The wallet address for which to retrieve the transaction
            history.
        excluded_categories : list or None, optional
            A list of transaction categories to exclude. If None, a
            default list of categories including "contract interaction",
            "token receive", "airdrop", "receive", "approve", and "send"
            will be used.
        **kwargs : dict
            Additional keyword arguments to filter transactions, such
            as:
                - **from_block**: int
                    The minimum block number to start retrieving
                    transactions.
                - **to_block**: int
                    The maximum block number to stop retrieving
                    transactions.
                - **from_date**: str
                    The start date
                    (in seconds or a momentjs-compatible datestring).
                - **to_date**: str
                    The end date
                    (in seconds or a momentjs-compatible datestring).
                - **include_internal_transactions**: bool
                    Whether to include internal transactions in the
                    results.
                - **nft_metadata**: bool
                    Whether to include NFT metadata in the results.
                - **cursor**: str
                    A pagination cursor returned from previous
                    responses.
                - **order**: str
                    The order of transactions, either "ASC" for
                    ascending or "DESC" for descending.
                - **limit**: int
                    The maximum number of transactions to retrieve.

        Returns
        -------
        list
            A list of transaction dictionaries that have been filtered to exclude
            spam and the specified categories.

        Side Effects
        ------------
        Logs the start and completion of the transaction retrieval process.
        """
        self.logger.info("Retrieving transactions for wallet: %s", wallet)

        params = {**kwargs}
        params["chain"] = kwargs.get("chain", self.chain)
        params["address"] = kwargs.get("address", wallet)
        params["order"] = kwargs.get("order", "DESC")

        txn_infos = evm_api.wallets.get_wallet_history(
            api_key=self.api_key,
            params=params,
        )

        transactions = []

        if excluded_categories is None:
            excluded_categories = [
                "contract interaction",
                "token receive",
                "airdrop",
                "receive",
                "approve",
                "send",
            ]

        for txn in txn_infos["result"]:
            is_verified_contract = all(
                [item["verified_contract"] for item in txn["erc20_transfers"]]
            )
            is_not_spam = not txn["possible_spam"] and is_verified_contract
            in_excluded_categories = txn["category"] in excluded_categories
            txn['cursor'] = txn_infos.get("cursor", None)

            if is_not_spam and not in_excluded_categories:
                transactions.append(txn)

        self.logger.info("Retrieved %d transactions", len(transactions))

        return transactions

    def get_swaps(self, swaps: list, add_summary: bool = False) -> list:
        """
        Retrieves all swaps data for a given wallet address.

        Parameters
        ----------
        swaps : list
            The swaps to retrieve data for.

        Returns
        -------
        list
            A list of dictionaries, each containing details of a swap
            transaction.
        """
        swaps_data = []

        infos_df = pd.DataFrame(swaps)
        infos_df["transaction_fee"] = infos_df["transaction_fee"].astype(float)
        infos_df["summary"] = infos_df["summary"]

        for idx, x in enumerate(swaps):
            try:
                swap = self.process_transaction_data(x["erc20_transfers"])

            except ValueError as exc:
                erc20_transfer_direction = x["erc20_transfers"][0]["direction"]

                if erc20_transfer_direction == "send":
                    x = x["erc20_transfers"] + x["native_transfers"]

                elif erc20_transfer_direction == "receive":
                    x = x["native_transfers"] + x["erc20_transfers"]

                else:
                    raise ValueError("unknown direction") from exc

                swap = self.process_transaction_data(x)

            swap.extend([{"txn_fee": infos_df.loc[idx, "transaction_fee"]}])

            if add_summary:
                swap.extend([{"summary": infos_df.loc[idx, "summary"]}])

            swaps_data.append(swap)

        return swaps_data

    def get_account_swaps(
        self,
        wallet: str,
        coin_name: bool = False,
        add_summary: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves all swaps for a given wallet address.

        Parameters
        ----------
        wallet : str
            The wallet address to retrieve swaps for.
        coin_name : bool
            Whether to include the names of the coins being swapped.


        Returns
        -------
        pandas.DataFrame
            A DataFrame containing details of all swaps for the given
            wallet address.
        """
        swaps_list = self.fetch_transactions(wallet)
        swaps_data = self.get_swaps(swaps_list, add_summary)

        swap_columns = ["token_symbol", "value_formatted"]
        from_df = pd.DataFrame(pd.DataFrame(swaps_data)[0].tolist())[
            swap_columns
        ]
        from_df = from_df.rename(
            columns={
                "token_symbol": "from_coin_name",
                "value_formatted": "from",
            }
        )

        to_df = pd.DataFrame(pd.DataFrame(swaps_data)[1].tolist())[
            swap_columns
        ]
        to_df = to_df.rename(
            columns={"token_symbol": "to_coin_name", "value_formatted": "to"}
        )

        fee_df = pd.DataFrame(pd.DataFrame(swaps_data)[2].tolist())

        columns_name = [
            "from",
            "to",
            "USD Price",
            "from_coin_name",
            "to_coin_name",
            "txn_fee",
        ]

        data_dfs = [from_df, to_df, fee_df]

        if add_summary:
            columns_name.append("summary")
            summary_df = pd.DataFrame(pd.DataFrame(swaps_data)[3].tolist())
            data_dfs.append(summary_df)

        swaps_df = pd.concat(data_dfs, axis=1)

        swaps_df[["from", "to"]] = swaps_df[["from", "to"]].astype(float)

        swaps_df["USD Price"] = np.where(
            swaps_df["to_coin_name"].str.startswith("USD"),
            swaps_df["to"] / swaps_df["from"],
            swaps_df["from"] / swaps_df["to"],
        )

        swaps_df = swaps_df[columns_name]

        if not coin_name:
            coin_name_columns = ["from_coin_name", "to_coin_name"]
            swaps_df = swaps_df.drop(columns=coin_name_columns)

        return swaps_df

    def fetch_erc20_balances_at_block(
        self,
        wallet_address: str,
        token_addresses: list[str],
        block_number: int | None = None,
    ) -> dict[str, float]:
        """
        Retrieve formatted ERC-20 balances for an explicit token list.

        Bypasses the spam/unverified-contract filters of
        :py:meth:`fetch_wallet_token_balances` by querying tokens by
        contract address. This is robust to Moralis intermittently
        flagging stable-coin contracts (e.g. USDT) as unverified.

        Parameters
        ----------
        wallet_address : str
            The wallet address to query.
        token_addresses : list[str]
            Lowercase ERC-20 contract addresses to fetch.
        block_number : int or None, optional
            The block number to query at. ``None`` for latest.

        Returns
        -------
        dict[str, float]
            Mapping of lowercase token address to decimal-normalized
            balance. Tokens absent from the response default to ``0.0``.
        """
        params: dict = {
            "chain": self.chain,
            "address": wallet_address,
            "token_addresses": token_addresses,
        }
        if block_number is not None:
            params["to_block"] = block_number

        result = evm_api.token.get_wallet_token_balances(
            api_key=self.api_key,
            params=params,
        )

        balances = {addr.lower(): 0.0 for addr in token_addresses}
        for entry in result:
            addr = entry["token_address"].lower()
            decimals = int(entry.get("decimals", 0) or 0)
            raw = int(entry.get("balance", 0) or 0)
            balances[addr] = raw / (10 ** decimals) if decimals else float(raw)
        return balances

    def fetch_native_balance_at_block(
        self,
        wallet_address: str,
        block_number: int | None = None,
        decimals: int = 18,
    ) -> float:
        """
        Retrieve the native-coin balance at a given block.

        Parameters
        ----------
        wallet_address : str
            The wallet address to query.
        block_number : int or None, optional
            The block number to query at. ``None`` for latest.
        decimals : int, optional
            Decimals of the native coin (default 18 for EVM chains).

        Returns
        -------
        float
            Decimal-normalized native balance.
        """
        params: dict = {"chain": self.chain, "address": wallet_address}
        if block_number is not None:
            params["to_block"] = block_number

        result = evm_api.balance.get_native_balance(
            api_key=self.api_key,
            params=params,
        )
        raw = int(result.get("balance", 0) or 0)
        return raw / (10 ** decimals)

    def get_account_balance_usd(
        self,
        wallet: str,
        token_map: dict[str, str] | None = None,
        native_price_address: str = (
            "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270"
        ),
        wbtc_address: str = (
            "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6"
        ),
        excluded_categories: list[str] | None = None,
        from_block: int | None = None,
        include_current_block: bool = True,
    ) -> pd.DataFrame:
        """
        Build a per-block USD balance history for a wallet.

        Produces a DataFrame analogous to
        :py:meth:`DataPipelineAPI.get_wallet_transactions_unsynced` but
        sourced directly from explicit token-address lookups, which
        avoids the silent USDT/stable drop-outs that occur when Moralis
        flags stable contracts as unverified.

        Strategy
        --------
        1. Discover swap/receive blocks via
           :py:meth:`fetch_paginated_transactions`.
        2. Optionally append the current latest block.
        3. For each block: fetch tracked ERC-20 balances by address,
           native balance, WBTC USD price, and native USD price.
        4. Compose ``total_usd`` (stables + WBTC * price) and
           ``formatted_total_usd`` (``total_usd`` + native * native_price).

        Parameters
        ----------
        wallet : str
            The wallet address to evaluate.
        token_map : dict[str, str] or None, optional
            Mapping of display column name to token contract address.
            Defaults to Polygon stables + WBTC
            (``USDC``, ``USDT``, ``WBTC``).
        native_price_address : str, optional
            Contract address used to price the native coin
            (default WPOL on Polygon).
        wbtc_address : str, optional
            WBTC contract address used to compute WBTC USD value.
        excluded_categories : list[str] or None, optional
            Transaction categories to exclude when discovering blocks.
            Defaults to ``["approve"]`` (mirrors the unsynced pipeline).
        from_block : int or None, optional
            Starting block. If ``None``, derived from the wallet's first
            on-chain transaction via
            :py:meth:`fetch_first_and_last_transactions`.
        include_current_block : bool, optional
            If ``True`` (default), append a row for the current latest
            block so the tail reflects the live balance.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``height`` (block number, ascending) with columns
            ``<token symbols>``, native symbol, ``usdPrice``,
            ``polPrice``, ``blockTimestamp``, ``total_usd``, and
            ``formatted_total_usd``.
        """
        if token_map is None:
            token_map = {
                "USDC": "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
                "USDT": "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
                "WBTC": "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6",
            }
        if excluded_categories is None:
            excluded_categories = ["approve"]

        token_map_lower = {sym: addr.lower() for sym, addr in token_map.items()}
        token_addresses = list(token_map_lower.values())
        wbtc_addr_lower = wbtc_address.lower()

        if from_block is None:
            first_last = self.fetch_first_and_last_transactions(
                wallet_address=wallet
            )
            from_block = int(first_last["first_transaction"]["block_number"])

        self.logger.info(
            "Discovering blocks for wallet %s starting at block %d",
            wallet,
            from_block,
        )
        transactions = self.fetch_paginated_transactions(
            wallet_address=wallet,
            excluded_categories=excluded_categories,
            order="ASC",
            from_block=from_block,
        )

        txn_df = pd.DataFrame(transactions)
        if txn_df.empty:
            self.logger.warning("No transactions discovered for wallet %s", wallet)
            block_meta = pd.DataFrame(
                columns=["block_number", "block_timestamp"]
            )
        else:
            block_meta = (
                txn_df[["block_number", "block_timestamp"]]
                .drop_duplicates(subset="block_number")
                .copy()
            )
            block_meta["block_number"] = block_meta["block_number"].astype(int)
            block_meta["block_timestamp"] = pd.to_datetime(
                block_meta["block_timestamp"], utc=True
            )

        if include_current_block:
            try:
                latest = self.fetch_block("now")
                latest_block = int(latest["block"])
                latest_ts = pd.to_datetime(latest["block_timestamp"], utc=True)
                if latest_block not in set(block_meta["block_number"].tolist()):
                    block_meta = pd.concat(
                        [
                            block_meta,
                            pd.DataFrame(
                                [{
                                    "block_number": latest_block,
                                    "block_timestamp": latest_ts,
                                }]
                            ),
                        ],
                        ignore_index=True,
                    )
            except Exception as exc:
                self.logger.warning(
                    "Could not append current block: %s", exc
                )

        block_meta = block_meta.sort_values("block_number").reset_index(drop=True)

        rows: list[dict] = []
        total = len(block_meta)
        for idx, meta in block_meta.iterrows():
            block = int(meta["block_number"])
            try:
                erc20 = self.fetch_erc20_balances_at_block(
                    wallet_address=wallet,
                    token_addresses=token_addresses,
                    block_number=block,
                )
                native = self.fetch_native_balance_at_block(
                    wallet_address=wallet,
                    block_number=block,
                )
                wbtc_price = float(
                    self.fetch_token_price(block, wbtc_address)["usdPrice"]
                )
                pol_price = float(
                    self.fetch_token_price(block, native_price_address)[
                        "usdPrice"
                    ]
                )
            except Exception as exc:
                self.logger.warning(
                    "Skipping block %d due to error: %s", block, exc
                )
                continue

            row: dict = {"height": block, "POL": native}
            for sym, addr in token_map_lower.items():
                row[sym] = erc20.get(addr, 0.0)
            row["usdPrice"] = wbtc_price
            row["polPrice"] = pol_price
            row["blockTimestamp"] = meta["block_timestamp"]

            stable_total = sum(
                row[sym]
                for sym, addr in token_map_lower.items()
                if addr != wbtc_addr_lower
            )
            wbtc_balance = next(
                (row[sym] for sym, addr in token_map_lower.items()
                 if addr == wbtc_addr_lower),
                0.0,
            )
            row["total_usd"] = stable_total + wbtc_balance * wbtc_price
            row["formatted_total_usd"] = row["total_usd"] + native * pol_price

            rows.append(row)

            progress = (idx + 1) / total if total else 1.0
            self.logger.info(
                "Progress: %.2f%% - %d / %d (block %d)",
                progress * 100,
                idx + 1,
                total,
                block,
            )

        if not rows:
            self.logger.warning("No balance rows produced for wallet %s", wallet)
            return pd.DataFrame()

        result_df = pd.DataFrame(rows).set_index("height").sort_index()
        return result_df

    def fetch_token_price(
        self,
        block_number: int,
        address: str = "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
    ) -> pd.Series:
        """
        Retrieves the token price at a specified block number using the
        Moralis API.

        Parameters
        ----------
        block_number : int
            The block number at which to fetch the token price.
        address : str
            The address of the token to retrieve the price for.

        Returns
        -------
        pandas.Series
            A Series containing the token price data as returned by the
            Moralis API.
        """
        params = {
            "chain": self.chain,
            "to_block": block_number,
            "address": address,
        }

        result = evm_api.token.get_token_price(
            api_key=self.api_key,
            params=params,
        )

        return result

    def fetch_block(self, unix_date: int | str | Literal["now"]) -> pd.Series:
        """
        Retrieves block information corresponding to a given Unix
        timestamp.

        Parameters
        ----------
        unix_date : int or str
            The Unix timestamp to retrieve the block information for.

        Returns
        -------
        dict
            A dictionary containing block information as returned by the
            Moralis API.
        """
        if isinstance(unix_date, int):
            unix_date = str(unix_date)
        if not isinstance(unix_date, (str, int)):
            raise InvalidArgumentError(
                "unix_date must be an integer or string"
            )

        if unix_date == "now":
            unix_date = str(int(time.time()))

        params = {"chain": self.chain, "date": unix_date}

        result = evm_api.block.get_date_to_block(
        api_key=self.api_key,
        params=params,
        )

        return pd.Series(result)

    def fetch_wallet_token_balances(
            self,
            wallet_address: str,
            block_number: int,
        ) -> pd.DataFrame:
        """
        Retrieves the token balances for a specified wallet address at
        a given block number.

        Parameters
        ----------
        wallet_address : str
            The wallet address for which to fetch token balances.
        block_number : int
            The block number at which to evaluate the wallet's token balances.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by token symbol with a single column
            (named after the block number) showing the balance of
            each token.
        """
        params = {
            "chain": self.chain,
            "to_block": block_number,
            "exclude_spam": True,
            "exclude_unverified_contracts": True,
            "address": wallet_address,
        }

        result = evm_api.wallets.get_wallet_token_balances_price(
            api_key=self.api_key,
            params=params,
        )['result']

        return result

    def get_wallet_token_balances(
        self,
        wallet_address: str,
        block_number: int,
    ) -> pd.DataFrame:
        """
        Retrieves and processes token balances for a wallet at a
        specific block.

        This method fetches the token balances for a given wallet
        address at a specific block number. It then filters the results
        to include only verified, non-spam tokens with a security score.
        The raw balance is adjusted using the token's decimals to get
        the actual token balance. The final output is a DataFrame
        formatted for easy analysis, with token symbols as the index.

        Parameters
        ----------
        wallet_address : str
            The wallet address to query.
        block_number : int
            The block number at which to fetch the balances.

        Returns
        -------
        pd.DataFrame
            A DataFrame with token symbols as the index and a single
            column named after the `block_number`, containing the
            calculated token balances.
        """
        result = self.fetch_wallet_token_balances(wallet_address, block_number)

        result_df = (
            pd.DataFrame(result)
            .query("verified_contract == True and possible_spam == False")
        )

        result_df["balance_formatted"] = (
            result_df["balance_formatted"].astype("float64")
        )

        inline_result = (
            result_df[["symbol", "balance_formatted"]]
            .set_index("symbol")
        )

        inline_result.columns = [str(block_number)]
        return inline_result

    def get_wallet_blocks(
        self,
        wallet_address: str,
        excluded_categories: list | None = None,
        **kwargs: dict,
    ) -> list:
        """
        Retrieves the historical token balances for a specific wallet.

        This method gathers all transactions for the given wallet
        address, extracts the block numbers, and then for each block
        (including the latest block), it queries the token balances and
        token price. The resulting data includes the token balance,
        corresponding USD price, and the block timestamp at which the
        price was retrieved.

        Parameters
        ----------
        wallet_address : str
            The wallet address for which to fetch the token balances
            history.
        token_address : str
            The address of the token to retrieve the price for at each
            block.
        **kwargs : dict
            Additional keyword arguments to filter transactions, such
            as:
                - **from_block**: int
                    The minimum block number to start retrieving
                    transactions.
                - **to_block**: int
                    The maximum block number to stop retrieving
                    transactions.
                - **from_date**: str
                    The start date
                    (in seconds or a momentjs-compatible datestring).
                - **to_date**: str
                    The end date
                    (in seconds or a momentjs-compatible datestring).
                - **include_internal_transactions**: bool
                    Whether to include internal transactions in the
                    results.
                - **nft_metadata**: bool
                    Whether to include NFT metadata in the results.
                - **cursor**: str
                    A pagination cursor returned from previous
                    responses.
                - **order**: str
                    The order of transactions, either "ASC" for
                    ascending or "DESC" for descending.
                - **limit**: int
                    The maximum number of transactions to retrieve.

        Returns
        -------
        list
            A list of transaction dictionaries that have been filtered
            to exclude spam and the specified categories.

        Raises
        ------
        InvalidArgumentError
            If `from_block` is greater than or equal to `to_block`.
        """
        if "from_block" in kwargs:
            transactions = self.fetch_paginated_transactions(
                wallet_address=wallet_address,
                excluded_categories=excluded_categories,
                **kwargs,
            )
        else:
            transactions = self.fetch_unpaginated_transactions(
                wallet_address=wallet_address,
                **kwargs,
            )

        block_numbers = (
            pd.DataFrame(transactions)['block_number']
            .astype(int)
            .tolist()
        )

        return block_numbers

    def get_wallet_token_balances_history(
        self,
        wallet_address: str,
        token_address: str,
        excluded_categories: list | None = None,
        **kwargs: dict,
    ) -> pd.DataFrame:
        """
        Retrieves the historical token balances for a specific wallet
        and token.

        This method gathers all transactions for the given wallet
        address, extracts the block numbers, and then for each block
        (including the latest block), it queries the token balances and
        token price. The resulting data includes the token balance,
        corresponding USD price, and the block timestamp at which the
        price was retrieved.

        Parameters
        ----------
        wallet_address : str
            The wallet address for which to fetch the token balances
            history.
        token_address : str
            The address of the token to retrieve the price for at each
            block.
        **kwargs : dict
            Additional keyword arguments to filter transactions, such
            as:
                - **from_block**: int
                    The minimum block number to start retrieving
                    transactions.
                - **to_block**: int
                    The maximum block number to stop retrieving
                    transactions.
                - **from_date**: str
                    The start date
                    (in seconds or a momentjs-compatible datestring).
                - **to_date**: str
                    The end date
                    (in seconds or a momentjs-compatible datestring).
                - **include_internal_transactions**: bool
                    Whether to include internal transactions in the
                    results.
                - **nft_metadata**: bool
                    Whether to include NFT metadata in the results.
                - **cursor**: str
                    A pagination cursor returned from previous
                    responses.
                - **order**: str
                    The order of transactions, either "ASC" for
                    ascending or "DESC" for descending.
                - **limit**: int
                    The maximum number of transactions to retrieve.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the token balances (transposed),
            USD price, and block timestamp for each evaluated block.
        """
        updated_blocks = self.get_wallet_blocks(
            wallet_address=wallet_address,
            excluded_categories=excluded_categories,
            **kwargs,
        )

        token_balances = []

        for block in updated_blocks:
            self.logger.info(f"Getting token balances for block {block}.")

            temp_df = self.get_wallet_token_balances(wallet_address, block).T
            token_price = self.fetch_token_price(block, token_address)

            temp_df['usdPrice'] = token_price['usdPrice']
            temp_df['blockTimestamp'] = pd.Timestamp(
                int(token_price['blockTimestamp']),
                unit='ms',
            )

            token_balances.append(temp_df)

            progress = len(token_balances) / len(updated_blocks)
            progress_abs = f"{len(token_balances)} / {len(updated_blocks)}"

            self.logger.info(f"Progress: {progress:.2%} - {progress_abs}")

        return pd.concat(token_balances)

    def fetch_paginated_transactions(
        self,
        wallet_address: str,
        excluded_categories: list | None = None,
        **kwargs,
    ) -> list:
        """
        Fetches transactions in paginated chunks between a start and end
        block.

        This method retrieves transactions for a specified wallet
        address, starting from an initial block and ending at a final
        block. It processes the transactions in chunks of 1,000,000
        blocks to avoid hitting API limits or performance issues. The
        transactions are filtered based on the provided keyword
        arguments, which can include block range, date range, and other
        transaction filters.

        Parameters
        ----------
        wallet_address : str
            The wallet address for which to fetch transactions.
        excluded_categories : list or None, optional
            A list of transaction categories to exclude from the
            results. If None, a default list of categories including
            "contract interaction", "token receive", "airdrop", "receive",
            "approve", and "send" will be used.
            If you want to include all categories, set this to an empty
            list `[]` or `None`.
            (default: None)
        **kwargs : dict
            Additional keyword arguments to filter transactions, such
            as:
                - **from_block**: int
                    The minimum block number to start retrieving
                    transactions.
                - **to_block**: int
                    The maximum block number to stop retrieving
                    transactions.
                - **from_date**: str
                    The start date
                    (in seconds or a momentjs-compatible datestring).
                - **to_date**: str
                    The end date
                    (in seconds or a momentjs-compatible datestring).
                - **include_internal_transactions**: bool
                    Whether to include internal transactions in the
                    results.
                - **nft_metadata**: bool
                    Whether to include NFT metadata in the results.
                - **cursor**: str
                    A pagination cursor returned from previous
                    responses.
                - **order**: str
                    The order of transactions, either "ASC" for
                    ascending or "DESC" for descending.
                - **limit**: int
                    The maximum number of transactions to retrieve.
        """
        if "from_block" not in kwargs:
            raise InvalidArgumentError(
                "from_block is required for paginated transactions"
            )

        response = self.fetch_transactions(
            wallet=wallet_address,
            excluded_categories=excluded_categories,
            **kwargs,
        )

        txn_list = response

        while response[0]["cursor"]:
            kwargs["cursor"] = response[0]["cursor"]
            response = self.fetch_transactions(
                wallet=wallet_address,
                excluded_categories=excluded_categories,
                **kwargs
            )
            txn_list.extend(response)

        return txn_list

    def fetch_unpaginated_transactions(
        self,
        wallet_address: str,
        **kwargs: dict,
    ) -> list:
        """
        Retrieves transactions for a wallet without pagination constraints.

        This method gathers all transactions for the given wallet
        address using a single API call without pagination.

        Parameters
        ----------
        wallet_address : str
            The wallet address for which to fetch transactions.
        **kwargs : dict
            Additional keyword arguments to filter transactions, such
            as:
                - **from_block**: int
                    The minimum block number to start retrieving
                    transactions.
                - **to_block**: int
                    The maximum block number to stop retrieving
                    transactions.
                - **from_date**: str
                    The start date
                    (in seconds or a momentjs-compatible datestring).
                - **to_date**: str
                    The end date
                    (in seconds or a momentjs-compatible datestring).
                - **include_internal_transactions**: bool
                    Whether to include internal transactions in the
                    results.
                - **nft_metadata**: bool
                    Whether to include NFT metadata in the results.
                - **cursor**: str
                    A pagination cursor returned from previous
                    responses.
                - **order**: str
                    The order of transactions, either "ASC" for
                    ascending or "DESC" for descending.
                - **limit**: int
                    The maximum number of transactions to retrieve.

        Returns
        -------
        list
            A list of transaction dictionaries that have been filtered to exclude
            spam and the specified categories.
        """
        return self.fetch_transactions(
            wallet=wallet_address,
            excluded_categories=None,
            **kwargs,
        )

    def fetch_first_and_last_transactions(
        self,
        wallet_address: str,
        chains: list[str] | None = None,
    ):
        """
        Fetches the first and last transactions for a given wallet
        address across multiple chains.

        Parameters
        ----------
        wallet_address : str
            The wallet address to query.
        chains : list[str] | None, optional
            A list of blockchain names to query.
            If None, defaults to the chain specified during the
            initialization of the MoralisAPI instance.
            (default: None)

        Returns
        -------
        dict
            A dictionary containing the first and last transactions for
            each chain.
        """
        first_and_last_transactions = {}

        if chains is None:
            chains = [self.chain]

        params = {
            "chains": chains,
            "address": wallet_address,
        }

        txn = evm_api.wallets.get_wallet_active_chains(
            api_key=self.api_key,
            params=params,
        )['active_chains']

        if txn:
            first_and_last_transactions = {
                "first_transaction": txn[0]["first_transaction"],
                "last_transaction": txn[0]["last_transaction"]
            }
        else:
            self.logger.warning(
                "No transactions found for wallet %s on chain %s",
                wallet_address,
                chains,
            )

        return pd.DataFrame(first_and_last_transactions)
