## Visão Geral

O repositório CryptoExplorer oferece uma interface unificada para consultar dados de blockchain, monitorar atividades de negociação, recuperar históricos de preços de exchanges e acessar métricas de saúde da rede Bitcoin. Explore e estenda os módulos para atender ao seu caso de uso específico.

Ele permite que você:

- Recupere dados de transações de swap da blockchain.
- Extraia e converta dados de preços OHLCV via exchanges.
- Obtenha dados on-chain do Bitcoin.

## Glossário

- **swap**: Uma ação realizada por um usuário ao executar um trade em uma Exchange DEX.
- **txid**: ID da Transação.

## Classes da API e Uso

### AccountAPI

Este manipulador de API de alto nível encadeia múltiplos provedores para recuperar dados de transações de swap da blockchain.

Métodos:

- __get_wallet_swaps(wallet: str, coin_name: bool = False)__  
    Extrai todas as transações de swap para uma carteira.  
    _Nota_: definir o parâmetro `coin_name` como `True` incluirá os nomes dos tokens envolvidos.
- __get_buys(wallet_address: str, asset_name: str = "WBTC")__  
    Recupera transações de compra a partir dos dados de swap para o ativo especificado.
- __get_sells(wallet_address: str, asset_name: str = "WBTC")__  
    Recupera transações de venda a partir dos dados de swap para o ativo especificado.

Exemplo:

```py
from crypto_explorer import AccountAPI

account_api = AccountAPI(api_key="SUA_CHAVE_API_MORALIS", verbose=True)
wallet = "0xSeuEndereçoDeCarteira"

# Recuperar todas as transações de swap com nomes de moedas
swaps = account_api.get_wallet_swaps(wallet, coin_name=True)

# Recuperar transações de compra para o ativo "WBTC"
buys = account_api.get_buys(wallet, asset_name="WBTC")

# Recuperar transações de venda para o ativo "WBTC"
sells = account_api.get_sells(wallet, asset_name="WBTC")
```

### BlockscoutAPI

Fornece métodos para acessar dados de swap da API Blockscout.

Métodos:

- __get_transactions(txid: str, coin_name: bool = False)__  
    Extrai dados de transação de swap para um txid específico.
- __get_account_transactions(wallet: str, coin_names: bool = False)__  
    Recupera todas as transações de swap para uma carteira.

_Nota_: definir o parâmetro `coin_name` como `True` incluirá os nomes dos tokens envolvidos.

Exemplo:

```py
from crypto_explorer import BlockscoutAPI

blockscout = BlockscoutAPI(verbose=True)
txid = "0xExemploDeIDDeTransação"

# Obter detalhes da transação de swap para um txid específico
transaction = blockscout.get_transactions(txid, coin_name=True)

# Obter todas as transações de swap para uma carteira
wallet_tx = blockscout.get_account_transactions("0xSeuEndereçoDeCarteira", coin_names=True)
```

### MoralisAPI

Extrai transações de swap e dados de saldo histórico de tokens usando a API Moralis.

Métodos:

- __get_account_swaps(wallet: str, coin_name: bool = False, add_summary: bool = False)__  
    Recupera todas as transações de swap (swaps) para uma carteira.  
    _Nota_: definir o parâmetro `coin_name` como `True` incluirá os nomes dos tokens envolvidos.  
    _Nota 2_: definir o parâmetro `add_summary` como `True` incluirá resumos das transações.
- __get_wallet_token_balances_history(wallet_address: str, token_address: str, kwargs)__  
    Recupera os saldos históricos de tokens de uma carteira para rastrear mudanças no portfólio.

Exemplo:

```py
from crypto_explorer import MoralisAPI

moralis = MoralisAPI(verbose=True, api_key="SUA_CHAVE_API_MORALIS")
wallet = "0xSeuEndereçoDeCarteira"

# Obter transações de swap com nomes de moedas e resumo
swaps = moralis.get_account_swaps(wallet, coin_name=True, add_summary=True)

# Obter saldos históricos de tokens (histórico de portfólio)
history = moralis.get_wallet_token_balances_history(wallet, token_address="0xEndereçoDoToken")
```

### CcxtAPI

Recupera dados de mercado OHLCV (preço) de exchanges via biblioteca CCXT.

Métodos:

- __get_all_klines(until: int | None = None)__  
    Extrai dados de preço OHLCV para o símbolo e período de tempo configurados.
- __to_OHLCV()__  
    Converte os dados OHLCV obtidos em um DataFrame pandas.  
    _Nota_: Chame get_all_klines antes de to_OHLCV para evitar um ValueError.

Exemplo:

```py
import ccxt
from crypto_explorer import CcxtAPI

# Criar uma instância de API CCXT para BTCUSDT na Binance
ccxt_api = CcxtAPI("BTCUSDT", "2h", ccxt.binance(), verbose="Text")

# Buscar dados de preço OHLCV
ccxt_api.get_all_klines()

# Converter dados obtidos em um DataFrame
ohlcv_df = ccxt_api.to_OHLCV().data_frame
print(ohlcv_df)
```

### QuickNodeAPI

Extrai informações on-chain do Bitcoin usando endpoints do QuickNode.

Métodos:

- __get_blockchain_info()__  
    Extrai informações gerais on-chain do Bitcoin como tipo de rede, altura do bloco, progresso de sincronização e status de atualização do protocolo.
- __get_block_stats(block_height: int)__  
    Extrai estatísticas detalhadas de blocos Bitcoin, incluindo taxas de transação, métricas de tamanho, mudanças UTXO, dados SegWit e números econômicos (em satoshis).

Exemplo:

```py
from crypto_explorer import QuickNodeAPI

# Liste suas URLs de API QuickNode
api_keys = ["https://seu.endpoint.quicknode"]

quicknode = QuickNodeAPI(api_keys, default_api_key_idx=0)

# Recuperar informações gerais da blockchain Bitcoin
info = quicknode.get_blockchain_info()
print(info)

# Recuperar estatísticas para um bloco específico do Bitcoin
block_stats = quicknode.get_block_stats(680000)
print(block_stats)
```

## Estrutura do Repositório

- **api/**:

    - `account_api.py`: Gerencia manipuladores de API e encadeamento via `AccountAPI`.
    - `blockscout_api.py`: Contém o `BlockscoutAPI` para acessar dados de swap da blockchain.
    - `ccxt_api.py`: Fornece o `CcxtAPI` para extrair dados de mercado (OHLCV) de exchanges.
    - `moralis_api.py`: Implementa o `MoralisAPI` para consultar transações de swap e saldos de tokens da carteira.
    - `quicknode_api.py`: Contém o `QuickNodeAPI` para informações on-chain do Bitcoin.
    - **tests/**: Testes unitários para módulos no diretório `api/`.

- __custom_exceptions/__: Classes de exceção personalizadas.
- **utils/**: Funções e classes utilitárias (por exemplo, conversões de tempo, logging, utilitários de kline).
- **tests/**: Testes unitários gerais para funcionalidade do repositório.

## Testes

Todos os testes unitários estão dentro do diretório `tests/`

Para executar os testes, execute:

```sh
python -m pytest
```

## Configuração e Instalação

1. Clone o repositório:

```bash
git clone https://github.com/m-marqx/CryptoExplorer.git
cd CryptoExplorer 
```

2. Instale as dependências:

```sh
pip install -r requirements.txt
```

**Nota**: Certifique-se de estar usando Python 3.10 ou superior.

## Contribuindo

Contribuições são bem-vindas! Por favor, abra uma issue ou envie um pull request para quaisquer melhorias ou correções de bugs.
