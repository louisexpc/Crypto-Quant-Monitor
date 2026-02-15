import ccxt
import os
import time
import yaml
import logging
from typing import Any, Dict, Optional, Tuple, List
from utils.data_collector import ExchangeConfig

class Trader(object):
    def __init__(self, tradeConfig: Dict[str, Any],exchangeConfig: ExchangeConfig, apiKey: Optional[str] = None, apiSecret: Optional[str] = None):
        """
        ```
        Method 用途與注意事項
        - 初始化 Trader：完成 logger/auth/exchange 初始化，並預先 load_markets() 以便後續做 precision/limits 正規化。
        - 若 config 指定 sandbox=false（live），會要求 trade.allow_live=true 才允許系統啟動（安全防呆）。

        Args: 參數說明
          - tradeConfig: YAML 讀進來的設定 dict（需包含 exchange/trade/risk/futures/logging/network 等區塊）。
          - apiKey: API Key（建議由環境變數或 secrets manager 讀取後傳入）。
          - apiSecret: API Secret（建議由環境變數或 secrets manager 讀取後傳入）。

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: Binance testnet 若遇到 timestamp/recvWindow 問題，建議開啟 adjustForTimeDifference 並調整 recv_window。
          - hint 2: 若你要用期貨專屬能力（set_leverage/set_margin_mode/...），請在 config.exchange.id 使用對應 exchange class（如 binanceusdm）。
        ```
        """
        # raw config (可能為新格式：{exchange:{...}, trade:{...}}；也支援舊格式/子集)
        self.tradeConfig = tradeConfig
        self.exchangeConfig = exchangeConfig

        if apiKey is None or apiSecret is None:
            raise ValueError("必須提供 apiKey 與 apiSecret 參數（建議從 .env 或環境變數讀取）")

        self.apiKey = apiKey
        self.apiSecret = apiSecret
        self.logger = logging.getLogger(self.__class__.__name__)

        self.exchange = self._init_exchange()
        self._post_init_safety_checks()

        self._markets_loaded = False
        self.load_markets()



    # -------------------------
    # Exchange initialization
    # -------------------------
    def _init_exchange(self):
        """
        - 建立 ccxt exchange instance 並套用常用參數（rate limit/timeout/recvWindow/options/defaultType）。
        - 若 exchange.sandbox=true，會呼叫 set_sandbox_mode(True)，並允許用 urls_override 覆寫 base api endpoint。

        Returns :
          - exchange: ccxt exchange instance（例如 ccxt.binance(...)）

        Hint : 
          - hint 2: defaultType 會根據 market_type = spot/future 設定。
          - hint 3: Spot testnet 常用 urls_override.api = "https://testnet.binance.vision/api"
        """

        options = {
            "defaultType": "spot" if self.exchangeConfig.market_type == "spot" else "future",
            "adjustForTimeDifference": self.exchangeConfig.adjust_for_time_difference,
        }

        exchange_class = getattr(ccxt, self.exchangeConfig.id)
        exchange = exchange_class({
            "apiKey": self.apiKey,
            "secret": self.apiSecret,
            "enableRateLimit": self.exchangeConfig.enable_rate_limit,
            "timeout": self.exchangeConfig.timeout_ms,
            "recvWindow": self.exchangeConfig.recv_window,
            "options": options,
        })

        if self.exchangeConfig.sandbox:
          # exchange.set_sandbox_mode(True)

          urls_override = self.exchangeConfig.urls_override or {}
          if self.exchangeConfig.id == "binance" and self.exchangeConfig.market_type == "spot":
              raise NotImplementedError("暫時不支援 Binance spot testnet，如需請改用 binanceus 或 binanceusdm。")
          else:
            # Futures / 其他交易所：優先使用 ccxt 內建 sandbox mode
            exchange.set_sandbox_mode(True)

            # 若你有特別指定 urls_override，仍允許覆寫（但要注意不同 exchange class 的 urls 結構差異）
            if "api" in urls_override:
                exchange.urls = exchange.urls or {}
                exchange.urls["api"] = urls_override["api"]
                self.logger.info(f"Override exchange.urls['api'] => {urls_override['api']}")

        if self.exchangeConfig.market_type == "future" and self.exchangeConfig.id == "binance":
            self.logger.warning(
                "market_type=future 且 exchange.id=binance：可下 U 本位/永續，但不一定支援 set_leverage/set_margin_mode 等期貨專用 helper。"
            )              

        self.logger.info(
            f"Exchange initialized: id={self.exchangeConfig.id}, market_type={self.exchangeConfig.market_type}, sandbox={self.exchangeConfig.sandbox}"
        )
        return exchange

    def _post_init_safety_checks(self) -> None:
        """
        - 安全防呆：當 sandbox=false（live）時，要求 trade.allow_live=true 才允許系統啟動（避免誤打真倉）。

        Hint :
          - hint 1: 若你要在真倉跑，請同時確保 symbol_allowlist / max_notional / min_free 等風控都設定好。
        """
        trade_cfg = self.tradeConfig or {}
        ex_cfg = self.exchangeConfig or {}
        sandbox = ex_cfg.sandbox if ex_cfg else True

        if not sandbox and not bool(trade_cfg.get("allow_live", False)):
            raise RuntimeError("安全防呆：sandbox=false（Live）但 trade.allow_live!=true，已中止。")

    # -------------------------
    # Common helpers
    # -------------------------
    def load_markets(self, reload: bool = False) -> Dict[str, Any]:
        """
        - 載入交易所 markets 資料（含 symbol、precision、limits、fees 等）。
        - 建議在初始化時先載入一次，後續下單可用 amount_to_precision / price_to_precision。

        Args: 
          - reload: 是否強制重新抓取 markets。

        Returns : 
          - markets: 交易所 markets dict（ccxt 格式）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若交易對/精度更新，可用 reload=True 強制刷新。

        """
        if self._markets_loaded and not reload:
            return self.exchange.markets
        markets = self._safe_call(self.exchange.load_markets, reload)
        self._markets_loaded = True
        return markets

    def _now_ms(self) -> int:
        """\
        ```
        Method 用途與注意事項
        - 回傳當前 epoch milliseconds（常用於 debug、紀錄、以及部分需要 timestamp 的流程）。

        Args: 參數說明
          - None

        Returns : 回傳內容說明
          - now_ms: int，當前時間（毫秒）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 交易所簽名 timestamp 通常由 ccxt 內部處理；此方法偏向輔助用途。
        ```
        """
        return int(time.time() * 1000)

    def _safe_call(self, fn, *args, **kwargs):
        """\
        ```
        Method 用途與注意事項
        - 封裝 ccxt API 呼叫：遇到常見網路/交易所暫不可用/timeout/防護時做重試（指數退避）。
        - AuthenticationError 不重試（通常是 key/secret/權限問題）。

        Args: 參數說明
          - fn: 要呼叫的 function（例如 self.exchange.fetch_balance）
          - *args: fn 的 positional args
          - **kwargs: fn 的 keyword args

        Returns : 回傳內容說明
          - result: fn 的回傳值（依 ccxt method 而定）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 重試次數與基礎 sleep 可由 network.max_retry / network.base_sleep_sec 控制。
          - hint 2: 未涵蓋的 Exception 會被 logger.exception 記錄後直接拋出，方便定位 bug。
        ```
        """
        net_cfg = self.tradeConfig.get("network", {}) or {}
        max_retry = int(net_cfg.get("max_retry", 5))
        base_sleep = float(net_cfg.get("base_sleep_sec", 0.5))

        last_err = None
        for i in range(max_retry):
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
                last_err = e
                sleep_sec = base_sleep * (2 ** i)
                self.logger.warning(
                    f"Network/Exchange error on attempt {i+1}/{max_retry}: {repr(e)}; sleep={sleep_sec:.2f}s"
                )
                time.sleep(sleep_sec)
            except ccxt.AuthenticationError:
                raise
            except Exception as e:
                self.logger.exception(f"Unexpected error: {repr(e)}")
                raise
        raise last_err

    def _normalize_symbol(self, symbol: str) -> str:
        """\
        ```
        Method 用途與注意事項
        - 將輸入 symbol 正規化成 ccxt 常用格式（例如 BTC/USDT）。
        - 若輸入是 BTCUSDT 且以 USDT 結尾，會嘗試轉成 BTC/USDT。

        Args: 參數說明
          - symbol: 原始輸入交易對字串。

        Returns : 回傳內容說明
          - normalized_symbol: 正規化後的 symbol

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若你要支援更多格式（例如 ETHUSD_PERP 或 1000SHIB/USDT），建議改為依 markets mapping 做嚴格轉換。
        ```
        """
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        return symbol

    def _ensure_symbol_allowed(self, symbol: str) -> None:
        """\
        ```
        Method 用途與注意事項
        - allowlist 風控：若 trade.symbol_allowlist 有設定，則 symbol 必須在清單內才允許操作。

        Args: 參數說明
          - symbol: 已正規化的交易對（例如 BTC/USDT）

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 真倉務必設定 allowlist，避免策略/資料錯誤導致下到不該交易的標的。
        ```
        """
        allowlist = self.tradeConfig.get("symbol_allowlist", None)
        if allowlist and symbol not in allowlist:
            raise PermissionError(f"Symbol 不在 allowlist：{symbol}")

    def _to_amount(self, symbol: str, amount: float) -> str:
        """\
        ```
        Method 用途與注意事項
        - 使用 ccxt 的 amount_to_precision 依交易對精度將 amount 格式化（避免下單因精度不符被拒）。

        Args: 參數說明
          - symbol: 交易對（需已 load_markets）
          - amount: 原始下單數量（base asset amount）

        Returns : 回傳內容說明
          - amount_str: precision 後的 amount 字串

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 需先 load_markets()，否則 exchange 可能缺少 precision 資訊。
        ```
        """
        return self.exchange.amount_to_precision(symbol, amount)

    def _to_price(self, symbol: str, price: float) -> str:
        """\
        ```
        Method 用途與注意事項
        - 使用 ccxt 的 price_to_precision 依交易對精度將 price 格式化（避免下單因 tick size 不符被拒）。

        Args: 參數說明
          - symbol: 交易對（需已 load_markets）
          - price: 原始限價

        Returns : 回傳內容說明
          - price_str: precision 後的 price 字串

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 限價單價格精度不合規，是 Binance 常見的下單失敗原因之一。
        ```
        """
        return self.exchange.price_to_precision(symbol, price)

    def get_market_type(self) -> str:
        """\
        ```
        Method 用途與注意事項
        - 回傳目前 Trader 設定的市場類型（spot 或 future）。

        Args: 參數說明
          - None

        Returns : 回傳內容說明
          - market_type: "spot" 或 "future"

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 這裡取自 config.exchange.market_type。
        ```
        """
        return str((self.exchangeConfig or {}).get("market_type", "spot")).lower()

    def is_future(self) -> bool:
        """\
        ```
        Method 用途與注意事項
        - 判斷目前是否為期貨（永續/U 本位）模式。

        Args: 參數說明
          - None

        Returns : 回傳內容說明
          - is_future: bool

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 以 config.exchange.market_type == "future" 判斷。
        ```
        """
        return self.get_market_type() == "future"

    def get_market(self, symbol: str) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 取得 ccxt market metadata（precision/limits/contractSize 等）。

        Args: 參數說明
          - symbol: 交易對（BTC/USDT 或 BTCUSDT）

        Returns : 回傳內容說明
          - market: ccxt market dict

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 需先 load_markets()；本 class 在 __init__ 已先載入。
        ```
        """
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        if not self._markets_loaded:
            self.load_markets()
        return self.exchange.market(symbol)

    def get_contract_size(self, symbol: str) -> float:
        """\
        ```
        Method 用途與注意事項
        - 取得期貨合約乘數（contractSize）。
        - 對大多數 U 本位永續（linear swap），contractSize 常見為 1（代表 1 contract = 1 base coin）。

        Args: 參數說明
          - symbol: 交易對

        Returns : 回傳內容說明
          - contract_size: float

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若 market dict 缺少 contractSize，預設回傳 1.0。
        ```
        """
        m = self.get_market(symbol)
        cs = m.get("contractSize", None)
        return float(cs) if cs is not None else 1.0

    # -------------------------
    # Market data / Account
    # -------------------------
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 取得指定交易對 ticker（last/bid/ask/volume 等）。
        - 可用於估算 mid price、或作為策略輸入。

        Args: 參數說明
          - symbol: 交易對（BTC/USDT 或 BTCUSDT）

        Returns : 回傳內容說明
          - ticker: ccxt unified ticker dict

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 此方法會套用 symbol_allowlist 檢查。
        ```
        """
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        return self._safe_call(self.exchange.fetch_ticker, symbol)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 取得 order book（bids/asks），常用於估算 mid、滑點、impact。
        - limit 越大資料越多，但也可能更慢、更容易觸發 rate limit。

        Args: 參數說明
          - symbol: 交易對
          - limit: 深度（依交易所支援）

        Returns : 回傳內容說明
          - order_book: dict，包含 bids/asks

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若 bids/asks 為空，通常代表市場資料異常或該對不支援/不存在。
        ```
        """
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        return self._safe_call(self.exchange.fetch_order_book, symbol, limit)

    def fetch_balance(self) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 取得帳戶資產餘額（free/used/total）。

        Args: 參數說明
          - None

        Returns : 回傳內容說明
          - balance: ccxt unified balance dict

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: Spot testnet 有功能限制，部分資產/子帳戶等介面可能不可用。
        ```
        """
        return self._safe_call(self.exchange.fetch_balance)

    def get_free_balance(self, asset: str) -> float:
        """\
        ```
        Method 用途與注意事項
        - 取得指定 asset 的 free 餘額（盡量兼容不同 ccxt balance 格式）。

        Args: 參數說明
          - asset: 資產代號（例如 USDT / BTC）

        Returns : 回傳內容說明
          - free: float

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 部分 market/account 型態下，balance 結構會不同；此方法做 best-effort。
        ```
        """
        bal = self.fetch_balance() or {}
        if isinstance(bal.get(asset), dict):
            return float(bal.get(asset, {}).get("free", 0.0) or 0.0)
        # fallback: ccxt 常見 balance['free'] 是 dict
        free_map = bal.get("free", {}) if isinstance(bal.get("free"), dict) else {}
        return float(free_map.get(asset, 0.0) or 0.0)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """\
        ```
        Method 用途與注意事項
        - 取得目前未成交/部分成交的 open orders。
        - 常用於：下單上限控制、策略去重、撤單管理。

        Args: 參數說明
          - symbol: 可選；若提供則只查該交易對 open orders。

        Returns : 回傳內容說明
          - open_orders: list[dict]，ccxt unified orders

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若策略每輪都會下單，建議先查 open orders 做幂等控制（避免重複下單）。
        ```
        """
        if symbol:
            symbol = self._normalize_symbol(symbol)
            self._ensure_symbol_allowed(symbol)
            return self._safe_call(self.exchange.fetch_open_orders, symbol)
        return self._safe_call(self.exchange.fetch_open_orders)

    def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 依 order_id 查詢訂單狀態（open/closed/canceled 等）。
        - 部分交易所要求 symbol；此處提供 symbol 可提高成功率。

        Args: 參數說明
          - order_id: 訂單 ID
          - symbol: 可選；若提供會先做 allowlist 檢查

        Returns : 回傳內容說明
          - order: ccxt unified order dict

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若交易所端要求 symbol 而你未提供，可能會拋出錯誤；建議在你系統內保留 order_id->symbol mapping。
        ```
        """
        if symbol:
            symbol = self._normalize_symbol(symbol)
            self._ensure_symbol_allowed(symbol)
            return self._safe_call(self.exchange.fetch_order, order_id, symbol)
        return self._safe_call(self.exchange.fetch_order, order_id)

    # -------------------------
    # Risk checks
    # -------------------------
    def _estimate_mid_price(self, symbol: str) -> float:
        """\
        ```
        Method 用途與注意事項
        - 估算 mid price：優先用 order book 的 best bid/ask 平均；若 order book 不可用則 fallback 用 ticker.last。
        - 用於名目金額估算、滑點風控等。

        Args: 參數說明
          - symbol: 交易對（需已 allowlist 通過）

        Returns : 回傳內容說明
          - mid_price: float

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 這是「粗估 mid」，不等於實際成交價；若要更嚴謹，需用 order book impact 模擬成交 VWAP。
        ```
        """
        ob = self.fetch_order_book(symbol, limit=5)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            t = self.fetch_ticker(symbol)
            last = t.get("last", None)
            if last is None:
                raise RuntimeError("無法取得 mid/last price 以估算名目金額。")
            return float(last)
        return (float(bids[0][0]) + float(asks[0][0])) / 2.0

    def _check_trade_enabled(self) -> None:
        """\
        ```
        Method 用途與注意事項
        - 檢查交易總開關 trade.enabled。
        - 若為 false，所有下單/撤單會被阻擋。

        Args: 參數說明
          - None

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 部署時可用 trade.enabled=false 做「只監控不交易」模式。
        ```
        """
        trade_cfg = self.tradeConfig or {}
        if not bool(trade_cfg.get("enabled", True)):
            raise RuntimeError("trade.enabled=false：已禁止送單。")

    def _check_open_orders_cap(self, symbol: str) -> None:
        """\
        ```
        Method 用途與注意事項
        - 限制單一 symbol 的 open orders 數量，避免策略異常時爆量掛單。

        Args: 參數說明
          - symbol: 交易對

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 上限由 trade.max_open_orders_per_symbol 控制；<=0 表示不限制。
        ```
        """
        cap = int(self.tradeConfig.get("max_open_orders_per_symbol", 0))
        if cap <= 0:
            return
        opens = self.fetch_open_orders(symbol)
        if len(opens) >= cap:
            raise RuntimeError(f"Open orders 達上限：symbol={symbol}, open={len(opens)}, cap={cap}")

    def _check_min_limits(self, symbol: str, amount: float, price: Optional[float]) -> None:
        """\
        ```
        Method 用途與注意事項
        - 檢查 market 的最小下單限制（min amount / min cost）。

        Args: 參數說明
          - symbol: 交易對
          - amount: base amount（對 futures 也可能是 base amount / contracts，依 ccxt market 定義）
          - price: 用於 cost 粗估的 price（若 None 會用 mid price）

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 不同市場對 amount/cost 的定義可能不同；此方法是 best-effort，以避免最常見的「低於最小下單」錯誤。
        ```
        """
        m = self.get_market(symbol)
        limits = m.get("limits", {}) or {}
        amt_min = (limits.get("amount", {}) or {}).get("min", None)
        cost_min = (limits.get("cost", {}) or {}).get("min", None)

        if amt_min is not None and float(amount) < float(amt_min):
            raise RuntimeError(f"amount 低於最小限制：amount={amount}, min={amt_min}, symbol={symbol}")

        if cost_min is not None:
            used_price = float(price) if price is not None else self._estimate_mid_price(symbol)
            cost = float(amount) * used_price
            if cost < float(cost_min):
                raise RuntimeError(f"cost 低於最小限制：cost~={cost}, min={cost_min}, symbol={symbol}")

    def _check_balance_for_side(self, symbol: str, side: str, notional_est: float, reduce_only: bool) -> None:
        """\
        ```
        Method 用途與注意事項
        - 下單前資金/保證金門檻檢查（spot 與 futures 的概念不同）。

        Args: 參數說明
          - symbol: 交易對（已正規化）
          - side: buy / sell
          - notional_est: 估算名目金額（USDT）
          - reduce_only: 是否為 reduceOnly（只減倉）；若是，通常不必做新增保證金檢查

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: futures：檢查的是「初始保證金」粗估（notional / leverage + buffer）。
          - hint 2: spot buy：檢查 quote free；spot sell：檢查 base free。
        ```
        """
        if reduce_only:
            return

        if self.is_future():
            self._check_futures_initial_margin(notional_est)
            return

        base, quote = symbol.split("/")
        side = side.lower()
        if side == "buy":
            self._check_quote_balance(quote)
        else:
            free_base = self.get_free_balance(base)
            if free_base <= 0:
                raise RuntimeError(f"可用 {base} 餘額不足：free={free_base}")

    def _check_quote_balance(self, quote: str = "USDT") -> None:
        """\
        ```
        Method 用途與注意事項
        - 檢查 quote（例如 USDT）可用餘額是否高於 risk.min_free_quote_balance。
        - 主要用於 spot 買入 / 掛買單前的最低資金門檻（粗略風控）。

        Args: 參數說明
          - quote: quote asset symbol（預設 USDT）

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 這是「最低餘額門檻」，不是精準的下單可用額度檢查；更嚴謹做法是比對 notional + fees。
        ```
        """
        min_free = float(self.tradeConfig.get("risk", {}).get("min_free_quote_balance", 0))
        if min_free <= 0:
            return
        free = self.get_free_balance(quote)
        if free < min_free:
            raise RuntimeError(f"可用 {quote} 餘額不足：free={free}, min_required={min_free}")

    def _check_max_notional(self, symbol: str, amount: float, price: Optional[float]) -> Tuple[float, float]:
        """\
        ```
        Method 用途與注意事項
        - 檢查單筆下單名目金額（amount * price_est）不超過 risk.max_notional_per_order。
        - 若 price=None（市價單或未提供），使用 mid price 粗估。

        Args: 參數說明
          - symbol: 交易對
          - amount: 下單數量（base amount）
          - price: 限價單 price；市價單可傳 None

        Returns : 回傳內容說明
          - used_price: 用於估算 notional 的價格（price 或 mid）
          - notional: float，估算名目金額

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 市價單實際成交可能高於 mid；若風控要嚴格，應加入 buffer 或用 order book 模擬成交均價。
        ```
        """
        max_notional = float(self.tradeConfig.get("risk", {}).get("max_notional_per_order", 0))
        used_price = float(price) if price is not None else self._estimate_mid_price(symbol)
        notional = float(amount) * used_price
        if max_notional > 0 and notional > max_notional:
            raise RuntimeError(f"單筆名目超限：symbol={symbol}, notional={notional:.4f}, cap={max_notional}")
        return used_price, notional

    def _check_market_slippage(self, symbol: str, side: str, used_price: float) -> None:
        """\
        ```
        Method 用途與注意事項
        - 市價單滑點風控（粗估）：比較 used_price 與 mid price 的偏離是否在 risk.max_slippage_bps_market 以內。
        - buy：used_price 不得高於 mid*(1+max_bps)
        - sell：used_price 不得低於 mid*(1-max_bps)

        Args: 參數說明
          - symbol: 交易對
          - side: buy / sell
          - used_price: 用於估算的成交價（通常是 mid 或你策略提供的估價）

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 此處是「基於 mid 的簡化滑點控管」，不是 order book impact 模擬。
        ```
        """
        max_bps = float(self.tradeConfig.get("risk", {}).get("max_slippage_bps_market", 0))
        if max_bps <= 0:
            return
        mid = self._estimate_mid_price(symbol)
        if side.lower() == "buy":
            worst = mid * (1.0 + max_bps / 10000.0)
            if used_price > worst:
                raise RuntimeError(f"市價單滑點風控未過（buy）：used={used_price}, mid={mid}, worst={worst}")
        else:
            worst = mid * (1.0 - max_bps / 10000.0)
            if used_price < worst:
                raise RuntimeError(f"市價單滑點風控未過（sell）：used={used_price}, mid={mid}, worst={worst}")

    # -------------------------
    # Futures sizing / margin helpers
    # -------------------------

    def _get_quote_asset(self) -> str:
        """\
        ```
        Method 用途與注意事項
        - 取得本 module 預設的 quote asset（通常為 USDT）。
        - 依序取值：trade.position_sizing.quote -> trade.risk.quote_asset（相容舊設定）-> "USDT"。

        Args: 參數說明
          - None

        Returns : 回傳內容說明
          - quote: quote asset 字串（例如 "USDT"）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 你的新 config.yaml 建議放在 trade.position_sizing.quote。
        ```
        """
        trade_cfg = self.tradeConfig or {}
        ps_cfg = trade_cfg.get("position_sizing", {}) or {}
        quote = ps_cfg.get("quote", None)
        if quote is None:
            quote = (trade_cfg.get("risk", {}) or {}).get("quote_asset", "USDT")
        return str(quote)

    def get_config_leverage(self) -> int:
        """\
        ```
        Method 用途與注意事項
        - 取得 config.futures.leverage。

        Args: 參數說明
          - None

        Returns : 回傳內容說明
          - leverage: int

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若 futures.leverage 未設定，預設 1。
        ```
        """
        lev = int(self.tradeConfig.get("futures", {}).get("leverage", 1) or 1)
        return max(1, lev)

    def estimate_notional_from_budget(self, budget_quote: float, effective_leverage: float) -> float:
        """\
        ```
        Method 用途與注意事項
        - 以「本金(quote) * 有效槓桿」估算要打的名目（notional）。

        Args: 參數說明
          - budget_quote: 你願意投入的「本金」(quote，例如 USDT)
          - effective_leverage: 你策略層想要的有效槓桿（例如 2.5 表示 notional=本金*2.5）

        Returns : 回傳內容說明
          - notional: float（USDT）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 交易所層面的 leverage（set_leverage）通常是「symbol/倉位層級」的保證金設定，不是每單單獨槓桿。
          - hint 2: 若 effective_leverage > config.futures.leverage，代表你用同一筆本金想開更大倉位，通常不可能；此時應提高 config leverage 或降低 notional。
        ```
        """
        return float(budget_quote) * float(effective_leverage)

    def calc_amount_from_notional(self, symbol: str, notional: float, price: Optional[float] = None) -> float:
        """\
        ```
        Method 用途與注意事項
        - 以名目（USDT）反推下單 amount。

        Args: 參數說明
          - symbol: 交易對
          - notional: 名目金額（USDT）
          - price: 若提供，用此價格換算；否則以 mid price 粗估

        Returns : 回傳內容說明
          - amount: float（base amount；對多數 U 本位永續可視為幣數）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 對 U 本位永續多數 contractSize=1，因此 amount ≈ 合約張數；若你的市場 contractSize != 1，請留意。
        ```
        """
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        used_price = float(price) if price is not None else self._estimate_mid_price(symbol)
        if used_price <= 0:
            raise ValueError("price 必須為正")
        amount = float(notional) / used_price
        return float(amount)

    def _check_futures_initial_margin(self, notional_est: float) -> None:
        """\
        ```
        Method 用途與注意事項
        - futures 下單前，做「初始保證金」粗估檢查：required_margin ≈ notional / leverage + buffer。

        Args: 參數說明
          - notional_est: 估算名目（USDT）

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 這不是交易所最終風控計算（會含維持保證金、手續費、資金費等），但足以阻擋「明顯資金不足」。
          - hint 2: buffer 由 risk.futures_initial_margin_buffer_ratio 控制（預設 0.05 = +5% buffer）。
        ```
        """
        quote = self._get_quote_asset()
        free = self.get_free_balance(quote)
        lev = float(self.get_config_leverage())
        buf = float(self.tradeConfig.get("risk", {}).get("futures_initial_margin_buffer_ratio", 0.05))

        required = float(notional_est) / lev
        required *= (1.0 + max(0.0, buf))

        if free < required:
            raise RuntimeError(
                f"Futures 保證金不足（粗估）：free_{quote}={free:.4f} < required~={required:.4f} (notional={notional_est:.4f}, lev={lev}, buf={buf})"
            )

    # -------------------------
    # Order placement
    # -------------------------
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 統一的下單入口：支援 market/limit，並套用常用風控（enabled/allowlist/open cap/balance/max notional/slippage/precision/min limits）。
        - 若 trade.dry_run=true，會回傳模擬結果且不送單。

        Args: 參數說明
          - symbol: 交易對（BTC/USDT 或 BTCUSDT）
          - side: "buy" / "sell"
          - order_type: "market" / "limit"（若要 stop/TP/SL 建議用下方 futures helper）
          - amount: 下單數量（base amount）
          - price: limit 單必填；market 單可為 None
          - params: ccxt 的額外參數（例如 {"postOnly": True} / {"reduceOnly": True}）

        Returns : 回傳內容說明
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: limit 單價格/數量會做 precision 正規化（price_to_precision/amount_to_precision）。
          - hint 2: 若 risk.require_post_only_for_limit=true，會自動加上 params["postOnly"]=True。
          - hint 3: futures 下單會做初始保證金粗估檢查（可由 risk.futures_initial_margin_buffer_ratio 調整）。
        ```
        """
        self._check_trade_enabled()

        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        self._check_open_orders_cap(symbol)

        params = params or {}
        side = side.lower()
        order_type = order_type.lower()

        reduce_only = bool(params.get("reduceOnly", False) or params.get("reduce_only", False))

        # 先做名目與最小限制檢查（需價格粗估）
        used_price, notional = self._check_max_notional(symbol, amount, price)
        self._check_min_limits(symbol, amount, used_price)

        if order_type == "market":
            self._check_market_slippage(symbol, side, used_price)

        # 資金/保證金門檻（spot/futures 分流）
        self._check_balance_for_side(symbol, side, notional, reduce_only=reduce_only)

        require_post_only = bool(self.tradeConfig.get("risk", {}).get("require_post_only_for_limit", False))
        if order_type == "limit" and require_post_only:
            params = dict(params)
            params["postOnly"] = True

        amt_str = self._to_amount(symbol, float(amount))
        px_str = None
        if order_type == "limit":
            if price is None:
                raise ValueError("limit order 必須提供 price")
            px_str = self._to_price(symbol, float(price))

        trade_cfg = self.tradeConfig or {}
        if bool(trade_cfg.get("dry_run", False)):
            self.logger.info(
                f"[DRY_RUN] create_order symbol={symbol}, side={side}, type={order_type}, amount={amt_str}, price={px_str}, params={params}, notional~={notional:.4f}"
            )
            return {
                "id": None,
                "info": {"dry_run": True},
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "amount": float(amt_str),
                "price": float(px_str) if px_str is not None else None,
                "notional_est": notional,
                "params": params,
            }

        self.logger.info(
            f"Sending order symbol={symbol}, side={side}, type={order_type}, amount={amt_str}, price={px_str}, notional~={notional:.4f}"
        )

        if order_type == "market":
            return self._safe_call(self.exchange.create_order, symbol, "market", side, float(amt_str), None, params)
        if order_type == "limit":
            return self._safe_call(
                self.exchange.create_order,
                symbol,
                "limit",
                side,
                float(amt_str),
                float(px_str),
                params,
            )
        raise ValueError(f"Unsupported order_type: {order_type}")

    def create_order_by_notional(
        self,
        symbol: str,
        side: str,
        order_type: str,
        notional: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 以名目（USDT）為主的下單入口：你提供 notional，系統換算 amount 後呼叫 create_order。

        Args: 參數說明
          - symbol: 交易對
          - side: buy / sell
          - order_type: market / limit
          - notional: 名目金額（USDT）
          - price: limit 單必填；market 可為 None（用 mid 粗估換算 amount）
          - params: ccxt 額外參數

        Returns : 回傳內容說明
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: market 下單若未提供 price，amount 以 mid 粗估換算；實際成交價格偏離 mid 會導致實際 notional 偏移。
        ```
        """
        amt = self.calc_amount_from_notional(symbol, notional, price=price)
        return self.create_order(symbol, side, order_type, amt, price=price, params=params)

    def create_order_by_budget(
        self,
        symbol: str,
        side: str,
        order_type: str,
        budget_quote: float,
        effective_leverage: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 以「本金(USDT) * 有效槓桿」的思維下單：
          notional = budget_quote * effective_leverage
          amount = notional / price
        - 適用於你習慣用本金概念控倉位大小的策略。

        Args: 參數說明
          - symbol: 交易對
          - side: buy / sell
          - order_type: market / limit
          - budget_quote: 本金（USDT）
          - effective_leverage: 你想要的有效槓桿（策略層）
          - price: limit 單必填；market 可為 None（用 mid 粗估）
          - params: ccxt 額外參數（例如 reduceOnly）

        Returns : 回傳內容說明
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: effective_leverage 不是交易所的 leverage 設定；交易所 leverage（set_leverage）建議固定，策略用 notional 控制倉位。
          - hint 2: 若 effective_leverage > config.futures.leverage，代表用同一筆本金想放大到超過交易所 leverage 允許的保證金比；會在保證金檢查階段被擋下。
        ```
        """
        notional = self.estimate_notional_from_budget(budget_quote, effective_leverage)
        return self.create_order_by_notional(symbol, side, order_type, notional, price=price, params=params)

    # ---- shortcuts ----
    def create_market_buy(self, symbol: str, amount: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 市價買入快捷方法（等價於 create_order(..., side="buy", order_type="market")）。

        Args: 參數說明
          - symbol: 交易對
          - amount: base asset amount
          - params: ccxt 額外參數

        Returns : 回傳內容說明
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 市價單會套用滑點風控（risk.max_slippage_bps_market）。
        ```
        """
        return self.create_order(symbol, "buy", "market", amount, price=None, params=params)

    def create_market_sell(self, symbol: str, amount: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 市價賣出快捷方法（等價於 create_order(..., side="sell", order_type="market")）。

        Args: 參數說明
          - symbol: 交易對
          - amount: base asset amount
          - params: ccxt 額外參數

        Returns : 回傳內容說明
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 市價單會套用滑點風控（risk.max_slippage_bps_market）。
        ```
        """
        return self.create_order(symbol, "sell", "market", amount, price=None, params=params)

    def create_limit_buy(self, symbol: str, amount: float, price: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 限價買入快捷方法（等價於 create_order(..., side="buy", order_type="limit")）。

        Args: 參數說明
          - symbol: 交易對
          - amount: base asset amount
          - price: limit price
          - params: ccxt 額外參數（例如 postOnly）

        Returns : 回傳內容說明
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若 risk.require_post_only_for_limit=true 會自動加 postOnly。
        ```
        """
        return self.create_order(symbol, "buy", "limit", amount, price=price, params=params)

    def create_limit_sell(self, symbol: str, amount: float, price: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 限價賣出快捷方法（等價於 create_order(..., side="sell", order_type="limit")）。

        Args: 參數說明
          - symbol: 交易對
          - amount: base asset amount
          - price: limit price
          - params: ccxt 額外參數（例如 postOnly）

        Returns : 回傳內容說明
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若 risk.require_post_only_for_limit=true 會自動加 postOnly。
        ```
        """
        return self.create_order(symbol, "sell", "limit", amount, price=price, params=params)

    # -------------------------
    # Cancel helpers
    # -------------------------
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 撤銷單筆訂單。
        - 部分交易所撤單需要 symbol；建議盡量提供 symbol（並在系統內保存 order_id->symbol）。

        Args: 參數說明
          - order_id: 訂單 ID
          - symbol: 可選；提供則做 allowlist 檢查並提高撤單成功率

        Returns : 回傳內容說明
          - result: ccxt 回傳的撤單結果 dict

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: trade.enabled=false 時會阻擋撤單（如需允許撤單，請自行調整 _check_trade_enabled 的策略）。
        ```
        """
        self._check_trade_enabled()
        if symbol:
            symbol = self._normalize_symbol(symbol)
            self._ensure_symbol_allowed(symbol)
            return self._safe_call(self.exchange.cancel_order, order_id, symbol)
        return self._safe_call(self.exchange.cancel_order, order_id)

    def cancel_all_orders(self, symbol: str) -> Any:
        """\
        ```
        Method 用途與注意事項
        - 撤銷指定 symbol 的所有 open orders。
        - 若交易所不支援 cancel_all_orders，會 fallback：先 fetch_open_orders 再逐筆 cancel。

        Args: 參數說明
          - symbol: 交易對

        Returns : 回傳內容說明
          - result: 若支援 cancel_all_orders，回傳交易所結果；否則回傳逐筆撤單結果 list

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若 open orders 很多，逐筆撤單可能較慢，且較容易觸發 rate limit。
        ```
        """
        self._check_trade_enabled()
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)

        if hasattr(self.exchange, "cancel_all_orders"):
            return self._safe_call(self.exchange.cancel_all_orders, symbol)

        opens = self.fetch_open_orders(symbol)
        results = []
        for o in opens:
            oid = o.get("id")
            if oid:
                results.append(self.cancel_order(oid, symbol))
        return results

    # -------------------------
    # Futures helpers (optional)
    # -------------------------
    def set_leverage(self, symbol: str, leverage: int) -> Any:
        """\
        ```
        Method 用途與注意事項
        - 設定期貨槓桿（futures 才有意義；spot 通常不支援）。
        - leverage 通常是「symbol/倉位層級」設定，而不是每一筆 order 單獨設定。

        Args: 參數說明
          - symbol: 交易對
          - leverage: 槓桿倍數（整數）

        Returns : 回傳內容說明
          - result: ccxt 回傳結果（交易所原始 response 或 unified result）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若你用的是 ccxt.binance 且 defaultType=future，不一定代表支援 set_leverage；最好改用 binanceusdm。
        ```
        """
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        if not hasattr(self.exchange, "set_leverage"):
            raise NotImplementedError("此 exchange class 不支援 set_leverage（請確認你用的是期貨類型，例如 binanceusdm）。")
        return self._safe_call(self.exchange.set_leverage, int(leverage), symbol)

    def set_margin_mode(self, margin_mode: str, symbol: Optional[str] = None) -> Any:
        """\
        ```
        Method 用途與注意事項
        - 設定期貨保證金模式（isolated/cross）。

        Args: 參數說明
          - margin_mode: "isolated" 或 "cross"
          - symbol: 可選；部分交易所/市場要求指定 symbol

        Returns : 回傳內容說明
          - result: ccxt 回傳結果

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 有些市場/帳戶型態不支援切換，會回傳交易所錯誤；建議捕捉例外並 logger.warning。
        ```
        """
        if not hasattr(self.exchange, "set_margin_mode"):
            raise NotImplementedError("此 exchange class 不支援 set_margin_mode。")
        if symbol:
            symbol = self._normalize_symbol(symbol)
            self._ensure_symbol_allowed(symbol)
            return self._safe_call(self.exchange.set_margin_mode, margin_mode, symbol)
        return self._safe_call(self.exchange.set_margin_mode, margin_mode)

    def set_position_mode(self, hedged: bool) -> Any:
        """\
        ```
        Method 用途與注意事項
        - 設定持倉模式（one-way / hedge）。
        - hedged=True 表示雙向持倉（hedge mode）；False 表示單向（one-way）。

        Args: 參數說明
          - hedged: bool

        Returns : 回傳內容說明
          - result: ccxt 回傳結果

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 不是所有 binance futures 類型/帳戶都支援；若 exchange class 不支援，會 raise NotImplementedError。
        ```
        """
        if not hasattr(self.exchange, "set_position_mode"):
            raise NotImplementedError("此 exchange class 不支援 set_position_mode。")
        return self._safe_call(self.exchange.set_position_mode, hedged)

    def bootstrap_futures_settings(self, symbol: str) -> None:
        """\
        ```
        Method 用途與注意事項
        - 期貨啟動時的初始化設定：套用 config 裡的 leverage / margin_mode / position_mode。
        - 只有 market_type == "future" 才會執行；spot 直接 return。

        Args: 參數說明
          - symbol: 交易對（用於 set_leverage / set_margin_mode）

        Returns : 回傳內容說明
          - None

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若交易所不支援對應方法，會 warning 而不中斷（避免初始化直接爆掉）。
        ```
        """
        if not self.is_future():
            return

        fut_cfg = self.tradeConfig.get("futures", {})
        lev = int(fut_cfg.get("leverage", 0) or 0)
        mm = str(fut_cfg.get("margin_mode", "")).lower()
        pm = str(fut_cfg.get("position_mode", "")).lower()

        if lev > 0:
            try:
                self.set_leverage(symbol, lev)
                self.logger.info(f"Set leverage={lev} for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to set leverage: {repr(e)}")

        if mm in ("isolated", "cross"):
            try:
                self.set_margin_mode(mm, symbol)
                self.logger.info(f"Set margin_mode={mm} for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to set margin mode: {repr(e)}")

        if pm in ("one-way", "oneway", "one_way"):
            try:
                self.set_position_mode(False)
                self.logger.info("Set position_mode=one-way")
            except Exception as e:
                self.logger.warning(f"Failed to set position mode (one-way): {repr(e)}")
        elif pm in ("hedge", "hedged"):
            try:
                self.set_position_mode(True)
                self.logger.info("Set position_mode=hedge")
            except Exception as e:
                self.logger.warning(f"Failed to set position mode (hedge): {repr(e)}")

    # ---- Stop/TP helpers (best-effort) ----
    def create_stop_loss_market(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        reduce_only: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 建立 Stop Loss 市價單（通常是 futures 用來保護倉位）。

        Args: 參數說明
          - symbol: 交易對
          - side: "buy" / "sell"（停損單方向通常是「平倉方向」，例如持有多單要停損 => side="sell"）
          - amount: 下單數量
          - stop_price: 觸發價
          - reduce_only: 是否只減倉（建議 True）
          - params: ccxt 額外參數

        Returns : 回傳內容說明
          - order: ccxt order dict

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 不同 exchange class 對 stop 類型命名可能不同；此處用 ccxt 常見型別 "stop_market"，並在 params 傳 stopPrice。
          - hint 2: 若你用的是 binance spot，通常不支援；請用 futures class。
        ```
        """
        if not self.is_future():
            raise RuntimeError("create_stop_loss_market 主要給 futures 使用。")
        params = dict(params or {})
        params["stopPrice"] = float(stop_price)
        if reduce_only:
            params["reduceOnly"] = True
        return self.create_order(symbol, side, "market", amount, price=None, params={"type": "stop_market", **params})

    def create_take_profit_market(
        self,
        symbol: str,
        side: str,
        amount: float,
        take_profit_price: float,
        reduce_only: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """\
        ```
        Method 用途與注意事項
        - 建立 Take Profit 市價單（通常是 futures 用來保護倉位）。

        Args: 參數說明
          - symbol: 交易對
          - side: "buy" / "sell"（持有空單要止盈 => side="buy"；持有多單要止盈 => side="sell"）
          - amount: 下單數量
          - take_profit_price: 觸發價
          - reduce_only: 是否只減倉（建議 True）
          - params: ccxt 額外參數

        Returns : 回傳內容說明
          - order: ccxt order dict

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 此處用 ccxt 常見型別 "take_profit_market"，並在 params 傳 stopPrice。
        ```
        """
        if not self.is_future():
            raise RuntimeError("create_take_profit_market 主要給 futures 使用。")
        params = dict(params or {})
        params["stopPrice"] = float(take_profit_price)
        if reduce_only:
            params["reduceOnly"] = True
        return self.create_order(symbol, side, "market", amount, price=None, params={"type": "take_profit_market", **params})
