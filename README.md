## PT-BR
A versão traduzida deste README se encontra [aqui](https://github.com/m-marqx/CryptoExplorer/blob/main/README-ptbr.md)

# CryptoExplorer
## Overview

The CryptoExplorer repository offers a unified interface to query blockchain data, monitor trading activities, retrieve exchange price histories, and access Bitcoin network health metrics. Explore and extend the modules to suit your specific use-case.

It enables you to:

- Retrieve swap transaction data from the blockchain.
- Extract and convert OHLCV price data via exchanges.
- Fetch comprehensive Bitcoin on-chain metrics.

## Glossary

- **swap**: An action performed by a user when executing a trade on a DEX Exchange.
- **txid**: Transaction ID.

## API Classes & Usage

### AccountAPI

This high-level API handler chains multiple providers to retrieve swap transaction data from the blockchain.

Methods:

- __get_wallet_swaps(wallet: str, coin_name: bool = False)__  
   Extracts all swap transactions for a wallet.  
   _Note_: setting the `coin_name` param to `True` will include the names of the tokens involved.
- __get_buys(wallet_address: str, asset_name: str = "WBTC")__  
   Retrieves buy transactions from the swap data for the specified asset.
- __get_sells(wallet_address: str, asset_name: str = "WBTC")__  
   Retrieves sell transactions from the swap data for the specified asset.

Example:

```py
from crypto_explorer import AccountAPI

account_api = AccountAPI(api_key="YOUR_MORALIS_API_KEY", verbose=True)
wallet = "0xYourWalletAddress"

# Retrieve all swap transactions with coin names
swaps = account_api.get_wallet_swaps(wallet, coin_name=True)

# Retrieve buy transactions for asset "WBTC"
buys = account_api.get_buys(wallet, asset_name="WBTC")

# Retrieve sell transactions for asset "WBTC"
sells = account_api.get_sells(wallet, asset_name="WBTC")

```

### BlockscoutAPI

Provides methods to access swap data from the Blockscout API.

Methods:

- __get_transactions(txid: str, coin_name: bool = False)__  
   Extracts swap transaction data for a specific txid.
- __get_account_transactions(wallet: str, coin_names: bool = False)__  
   Retrieves all swap transactions for a wallet.

_Note_: setting the `coin_name` param to `True` will include the names of the tokens involved.

Example:

```py
from crypto_explorer import BlockscoutAPI

blockscout = BlockscoutAPI(verbose=True)
txid = "0xTransactionIDExample"

# Get swap transaction details for a specific txid
transaction = blockscout.get_transactions(txid, coin_name=True)

# Get all swap transactions for a wallet
wallet_tx = blockscout.get_account_transactions("0xYourWalletAddress", coin_names=True)

```

### MoralisAPI

Extracts swap transactions and historical token balance data using the Moralis API.

Methods:

- __get_account_swaps(wallet: str, coin_name: bool = False, add_summary: bool = False)__  
   Retrieves all swap transactions (swaps) for a wallet.  
   _Note_: setting the `coin_name` param to `True` will include the names of the tokens involved.  
   _Note 2_: setting the `add_summary` param to `True` will includes transaction summaries.
- __get_wallet_token_balances_history(wallet_address: str, token_address: str, kwargs)__  
   Retrieves a wallet’s historical token balances to track portfolio changes.

Example:

```py
from crypto_explorer import MoralisAPI

moralis = MoralisAPI(verbose=True, api_key="YOUR_MORALIS_API_KEY")
wallet = "0xYourWalletAddress"

# Get swap transactions with coin names and summary
swaps = moralis.get_account_swaps(wallet, coin_name=True, add_summary=True)

# Get historical token balances (portfolio history)
history = moralis.get_wallet_token_balances_history(wallet, token_address="0xTokenAddress")

```

### CcxtAPI

Retrieves OHLCV (price) market data from exchanges via the CCXT library.

Methods:

- __get_all_klines(until: int | None = None)__  
   Extracts OHLCV price data for the configured symbol and timeframe.
- __to_OHLCV()__  
   Converts the fetched OHLCV data into a pandas DataFrame.  
   _Note_: Call get_all_klines before to_OHLCV to avoid a ValueError.

Example:

```py
import ccxt
from crypto_explorer import CcxtAPI

# Create a CCXT API instance for BTCUSDT on Binance
ccxt_api = CcxtAPI("BTCUSDT", "2h", ccxt.binance(), verbose="Text")

# Fetch OHLCV price data
ccxt_api.get_all_klines()

# Convert fetched data to a DataFrame
ohlcv_df = ccxt_api.to_OHLCV().data_frame
print(ohlcv_df)

```

### QuickNodeAPI

Extracts Bitcoin on-chain information using QuickNode endpoints.

Methods:

- __get_blockchain_info()__  
   Extracts general on-chain Bitcoin information such as network type, block height, sync progress, and protocol upgrade status.
- __get_block_stats(block_height: int)__  
   Extracts detailed Bitcoin block statistics including transaction fees, size metrics, UTXO changes, SegWit data, and economic figures (in satoshis).

Example:

```py
from crypto_explorer import QuickNodeAPI

# List your QuickNode API URLs
api_keys = ["https://your.quicknode.endpoint"]

quicknode = QuickNodeAPI(api_keys, default_api_key_idx=0)

# Retrieve general Bitcoin blockchain information
info = quicknode.get_blockchain_info()
print(info)

# Retrieve statistics for a specific Bitcoin block
block_stats = quicknode.get_block_stats(680000)
print(block_stats)

```

## Repository Structure

- **api/**:

   - `account_api.py`: Manages API handlers and chaining via the `AccountAPI`.
   - `blockscout_api.py`: Contains the `BlockscoutAPI` for accessing blockchain swap data.
   - `ccxt_api.py`: Provides the `CcxtAPI` to extract market data (OHLCV) from exchanges.
   - `moralis_api.py`: Implements the `MoralisAPI` for querying swap transactions and wallet token balances.
   - `quicknode_api.py`: Contains the `QuickNodeAPI` for Bitcoin on-chain information.
   - **tests/**: Unit tests for modules in the `api/` directory.

- __custom_exceptions/__: Custom exception classes.
- **utils/**: Utility functions and classes (e.g., time conversions, logging, kline utilities).
- **tests/**: General unit tests for repository functionality.

## Tests

All unit tests are inside the `tests/` directory


To run the tests, execute:

```sh
cd src
python -m pytest ./tests
```

## Setup & Installation

1. Clone the repository:

```bash
git clone https://github.com/m-marqx/CryptoExplorer.git
cd CryptoExplorer 

```

2. Install dependencies:

```sh
pip install -r requirements.txt

```

**Note**: Ensure you are using Python 3.10 or greater.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
