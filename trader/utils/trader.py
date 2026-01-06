import ccxt
import os
import time
import yaml
import logging
from typing import Any, Dict, Optional, Tuple, List


class Trader(object):
    def __init__(self, tradeConfig: Dict[str, Any], apiKey: Optional[str] = None, apiSecret: Optional[str] = None):
        """
        - 初始化 Trader，完成 logger/auth/exchange 初始化，並預先 load_markets() 以便後續做 precision/limits 正規化。
        - 若 config 指定 sandbox=false（live），會要求 trade.allow_live=true 才允許下單（安全防呆）。

        Args:
          - tradeConfig: YAML 讀進來的設定 dict（需包含 exchange/trade/risk/logging/auth 等區塊）。
          - apiKey: 強制檢查，必須傳入(由 .env 讀取)。
          - apiSecret: 強制檢查，必須傳入(由 .env 讀取)。

        Hint : 
          - 若在 Binance testnet 遇到 timestamp/recvWindow 問題，建議開啟 adjustForTimeDifference 並調整 recvWindow。
        """
        self.tradeConfig = tradeConfig

        self._init_logger()
        if apiKey is None or apiSecret is None:
            raise ValueError("必須提供 apiKey 與 apiSecret 參數（請從 .env 讀取）")

        self.exchange = self._init_exchange()
        self._post_init_safety_checks()

        self._markets_loaded = False
        self.load_markets()

    # -------------------------
    # Config / Logger / Auth
    # -------------------------
    @staticmethod
    def load_yaml_config(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_logger(self) -> None:
        """
        - 初始化 logger（console + optional file handler）。
        - 避免重複加 handler（多次初始化 Trader 時常見問題）。

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: logging.log_file 若提供路徑，會寫入檔案；未提供則僅 console。
          - hint 2: level 需為 INFO/WARNING/ERROR/DEBUG 等字串。
        """
        log_cfg = self.tradeConfig.get("logging", {})
        level_name = str(log_cfg.get("level", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")

            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

            log_file = log_cfg.get("log_file", None)
            if log_file:
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setLevel(level)
                fh.setFormatter(fmt)
                self.logger.addHandler(fh)

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
        ex_cfg = self.tradeConfig.get("exchange", {})
        exchange_id = ex_cfg.get("id", "binance")
        market_type = ex_cfg.get("market_type", "spot")
        sandbox = bool(ex_cfg.get("sandbox", True))

        enable_rate_limit = bool(ex_cfg.get("enable_rate_limit", True))
        timeout_ms = int(ex_cfg.get("timeout_ms", 30000))
        recv_window = int(ex_cfg.get("recv_window", 10000))
        adjust_time_diff = bool(ex_cfg.get("adjust_for_time_difference", True))

        options = {
            "defaultType": "spot" if market_type == "spot" else "future",
            "adjustForTimeDifference": adjust_time_diff,
        }

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "apiKey": self.apiKey,
            "secret": self.apiSecret,
            "enableRateLimit": enable_rate_limit,
            "timeout": timeout_ms,
            "recvWindow": recv_window,
            "options": options,
        })

        if sandbox:
            exchange.set_sandbox_mode(True)

            urls_override = ex_cfg.get("urls_override", {}) or {}
            if "api" in urls_override:
                exchange.urls = exchange.urls or {}
                exchange.urls["api"] = urls_override["api"]
                self.logger.info(f"Override exchange.urls['api'] => {urls_override['api']}")

        self.logger.info(
            f"Exchange initialized: id={exchange_id}, market_type={market_type}, sandbox={sandbox}"
        )
        return exchange

    def _post_init_safety_checks(self) -> None:
        """
        - 安全防呆：當 sandbox=false（live）時，要求 trade.allow_live=true 才允許系統啟動（避免誤打真倉）。

        Hint : 
          - hint 1: 若你要在真倉跑，請同時確保 symbol_allowlist / max_notional / min_free 等風控都設定好。
        ```
        """
        trade_cfg = self.tradeConfig.get("trade", {})
        ex_cfg = self.tradeConfig.get("exchange", {})
        sandbox = bool(ex_cfg.get("sandbox", True))

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
        """
        - 回傳當前 epoch milliseconds（常用於 debug、紀錄、以及部分需要 timestamp 的流程）。

        Returns : 
          - now_ms: int，當前時間（毫秒）

        Hint : 
          - hint 1: 交易所簽名 timestamp 通常由 ccxt 內部處理；此方法偏向輔助用途。
        """
        return int(time.time() * 1000)

    def _safe_call(self, fn, *args, **kwargs):
        """
        - 封裝 ccxt API 呼叫，遇到常見網路/交易所暫不可用/timeout/防護時做重試（指數退避）。
        - AuthenticationError 不重試（通常是 key/secret/權限問題）。

        Args: 
          - fn: 要呼叫的 function（例如 self.exchange.fetch_balance）
          - *args: fn 的 positional args
          - **kwargs: fn 的 keyword args

        Returns : 
          - result: fn 的回傳值（依 ccxt method 而定）

        Hint : 
          - hint 1: 重試次數與基礎 sleep 可由 tradeConfig.network.max_retry/base_sleep_sec 控制（若你有設）。
          - hint 2: 未涵蓋的 Exception 會被 logger.exception 記錄後直接拋出，方便定位 bug。
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
        """
        - 將輸入 symbol 正規化成 ccxt 常用格式（例如 BTC/USDT）。
        - 若輸入是 BTCUSDT 且以 USDT 結尾，會嘗試轉成 BTC/USDT。

        Args:
          - symbol: 原始輸入交易對字串。

        Returns :
          - normalized_symbol: 正規化後的 symbol

        Hint : 
          - hint 1: 若你要支援更多格式（例如 ETHUSD_PERP 或 1000SHIB/USDT），建議改為依 markets mapping 做嚴格轉換。
        """
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        return symbol

    def _ensure_symbol_allowed(self, symbol: str) -> None:
        """
        - allowlist 風控：若 trade.symbol_allowlist 有設定，則 symbol 必須在清單內才允許操作。

        Args: 參數說明
          - symbol: 已正規化的交易對（例如 BTC/USDT）

        Hint :
          - hint 1: 建議在真倉務必設定 allowlist，避免策略/資料錯誤導致下到不該交易的標的。
        """
        allowlist = self.tradeConfig.get("trade", {}).get("symbol_allowlist", None)
        if allowlist:
            if symbol not in allowlist:
                raise PermissionError(f"Symbol 不在 allowlist：{symbol}")

    def _to_amount(self, symbol: str, amount: float) -> str:
        """
        - 使用 ccxt 的 amount_to_precision 依交易對精度將 amount 格式化（避免下單因精度不符被拒）。

        Args: 
          - symbol: 交易對（需已 load_markets）
          - amount: 原始下單數量（base asset）

        Returns : 
          - amount_str: precision 後的 amount 字串

        Hint : 
          - hint 1: 需先 load_markets()，否則 exchange 可能缺少 precision 資訊。
        """
        return self.exchange.amount_to_precision(symbol, amount)

    def _to_price(self, symbol: str, price: float) -> str:
        """

        - 使用 ccxt 的 price_to_precision 依交易對精度將 price 格式化（避免下單因 tick size 不符被拒）。

        Args: 
          - symbol: 交易對（需已 load_markets）
          - price: 原始限價

        Returns :
          - price_str: precision 後的 price 字串

        Hint : 
          - hint 1: 限價單價格精度不合規，是 Binance 常見的下單失敗原因之一。

        """
        return self.exchange.price_to_precision(symbol, price)

    # -------------------------
    # Market data / Account
    # -------------------------
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """

        - 取得指定交易對 ticker（last/bid/ask/volume 等）。
        - 可用於估算 mid price、或作為策略輸入。

        Args: 
          - symbol: 交易對（BTC/USDT 或 BTCUSDT）

        Returns :
          - ticker: ccxt unified ticker dict

        Hint : 
          - hint 1: 此方法會套用 symbol_allowlist 檢查。

        """
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        return self._safe_call(self.exchange.fetch_ticker, symbol)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        - 取得 order book（bids/asks），常用於估算 mid、滑點、impact。
        - limit 越大資料越多，但也可能更慢、更容易觸發 rate limit。

        Args: 
          - symbol: 交易對
          - limit: 深度（依交易所支援）

        Returns :
          - order_book: dict，包含 bids/asks

        Hint : 
          - hint 1: 若 bids/asks 為空，通常代表市場資料異常或該對不支援/不存在。

        """
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        return self._safe_call(self.exchange.fetch_order_book, symbol, limit)

    def fetch_balance(self) -> Dict[str, Any]:
        """
        - 取得帳戶資產餘額（free/used/total），用於下單前資金風控。

        Returns : 
          - balance: ccxt unified balance dict

        Hint : 
          - hint 1: Spot testnet 有功能限制，部分資產/子帳戶等介面可能不可用。
        """
        return self._safe_call(self.exchange.fetch_balance)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """

        - 取得目前未成交/部分成交的 open orders。
        - 常用於：下單上限控制、策略去重、撤單管理。

        Args: 
          - symbol: 可選；若提供則只查該交易對 open orders。

        Returns : 
          - open_orders: list[dict]，ccxt unified orders

        Hint : 
          - hint 1: 若策略每輪都會下單，建議先查 open orders 做幂等控制（避免重複下單）。
        """
        if symbol:
            symbol = self._normalize_symbol(symbol)
            self._ensure_symbol_allowed(symbol)
            return self._safe_call(self.exchange.fetch_open_orders, symbol)
        return self._safe_call(self.exchange.fetch_open_orders)

    def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        - 依 order_id 查詢訂單狀態（open/closed/canceled 等）。
        - 部分交易所要求 symbol；此處提供 symbol 可提高成功率。

        Args: 
          - order_id: 訂單 ID
          - symbol: 可選；若提供會先做 allowlist 檢查

        Returns : 
          - order: ccxt unified order dict

        Hint : 
          - hint 1: 若交易所端要求 symbol 而你未提供，可能會拋出錯誤；建議在你系統內保留 order_id->symbol 的 mapping。
        """
        if symbol:
            symbol = self._normalize_symbol(symbol)
            self._ensure_symbol_allowed(symbol)
            return self._safe_call(self.exchange.fetch_order, order_id, symbol)
        return self._safe_call(self.exchange.fetch_order, order_id)

    # -------------------------
    # Risk checks (common)
    # -------------------------
    def _estimate_mid_price(self, symbol: str) -> float:
        """
        - 估算 mid price：優先用 order book 的 best bid/ask 平均；若 order book 不可用則 fallback 用 ticker.last。
        - 用於名目金額估算、滑點風控等。

        Args: 
          - symbol: 交易對（需已 allowlist 通過）

        Returns : 
          - mid_price: float

        Hint : 
          - hint 1: 這是「粗估 mid」，不等於實際成交價；若你要更嚴謹滑點控制，需模擬 order book impact。
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
        """
        Method 用途與注意事項
        - 檢查交易總開關 trade.enabled。
        - 若為 false，所有下單/撤單會被阻擋。

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 部署時可用 trade.enabled=false 做「只監控不交易」模式。
        ```
        """
        trade_cfg = self.tradeConfig.get("trade", {})
        if not bool(trade_cfg.get("enabled", True)):
            raise RuntimeError("trade.enabled=false：已禁止送單。")

    def _check_open_orders_cap(self, symbol: str) -> None:
        """
        - 限制單一 symbol 的 open orders 數量，避免策略異常時爆量掛單。
        Args: 
          - symbol: 交易對

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 上限由 trade.max_open_orders_per_symbol 控制；<=0 表示不限制。
        """
        cap = int(self.tradeConfig.get("trade", {}).get("max_open_orders_per_symbol", 0))
        if cap <= 0:
            return
        opens = self.fetch_open_orders(symbol)
        if len(opens) >= cap:
            raise RuntimeError(f"Open orders 達上限：symbol={symbol}, open={len(opens)}, cap={cap}")

    def _check_quote_balance(self, quote: str = "USDT") -> None:
        """
        - 檢查 quote（例如 USDT）可用餘額是否高於 risk.min_free_quote_balance。
        - 主要用於買入/掛買單前的最低資金門檻（粗略風控）。

        Args: 
          - quote: quote asset symbol（預設 USDT）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 這是「最低餘額門檻」，不是精準的下單可用額度檢查；更嚴謹做法是比對 notional + fees。
        """
        min_free = float(self.tradeConfig.get("risk", {}).get("min_free_quote_balance", 0))
        if min_free <= 0:
            return
        bal = self.fetch_balance()
        free = 0.0
        try:
            free = float(bal.get(quote, {}).get("free", 0.0))
        except Exception:
            free = 0.0
        if free < min_free:
            raise RuntimeError(f"可用 {quote} 餘額不足：free={free}, min_required={min_free}")

    def _check_max_notional(self, symbol: str, amount: float, price: Optional[float]) -> Tuple[float, float]:
        """
        - 檢查單筆下單名目金額（amount * price_est）不超過 risk.max_notional_per_order。
        - 若 price=None（市價單或未提供），使用 mid price 粗估。

        Args: 
          - symbol: 交易對
          - amount: base asset amount
          - price: 限價單 price；市價單可傳 None

        Returns : 
          - used_price: 用於估算 notional 的價格（price 或 mid）
          - notional: float，估算名目金額

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 市價單實際成交可能高於 mid；若風控要嚴格，應加入 buffer 或用 order book 模擬成交均價。
        """
        max_notional = float(self.tradeConfig.get("risk", {}).get("max_notional_per_order", 0))
        used_price = float(price) if price is not None else self._estimate_mid_price(symbol)
        notional = float(amount) * used_price
        if max_notional > 0 and notional > max_notional:
            raise RuntimeError(f"單筆名目超限：symbol={symbol}, notional={notional:.4f}, cap={max_notional}")
        return used_price, notional

    def _check_market_slippage(self, symbol: str, side: str, used_price: float) -> None:
        """
        - 市價單滑點風控（粗估）：比較 used_price 與 mid price 的偏離是否在 risk.max_slippage_bps_market 以內。
        - buy：used_price 不得高於 mid*(1+max_bps)
        - sell：used_price 不得低於 mid*(1-max_bps)

        Args: 
          - symbol: 交易對
          - side: buy / sell
          - used_price: 用於估算的成交價（通常是 mid 或你策略提供的估價）

        Returns : 
          - None

        Hint : 
          - hint 1: 此處是「基於 mid 的簡化滑點控管」，不是 order book impact 模擬。
          - hint 2: 若你希望更可信，請改為：用 order book 按數量逐檔吃單計算 VWAP，再與 mid 比較。
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
        """
        - 統一的下單入口：支援 market/limit，並套用常用風控（enabled/allowlist/open cap/min balance/max notional/slippage/precision）。
        - 若 trade.dry_run=true，會回傳模擬結果且不送單。

        Args: 
          - symbol: 交易對（BTC/USDT 或 BTCUSDT）
          - side: "buy" / "sell"
          - order_type: "market" / "limit"
          - amount: 下單數量（base asset amount）
          - price: limit 單必填；market 單可為 None
          - params: ccxt 的額外參數（例如 {"postOnly": True}）

        Returns : 
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 
          - hint 1: limit 單價格/數量會做 precision 正規化（price_to_precision/amount_to_precision）。
          - hint 2: 若 risk.require_post_only_for_limit=true，會自動加上 params["postOnly"]=True。
          - hint 3: 若你要支援 OCO/stop/TP/SL 等進階單，通常是透過 params（或 exchange 特定方法）擴充。
        """
        self._check_trade_enabled()

        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        self._check_open_orders_cap(symbol)

        params = params or {}
        side = side.lower()
        order_type = order_type.lower()

        self._check_quote_balance("USDT")

        used_price, notional = self._check_max_notional(symbol, amount, price)

        if order_type == "market":
            self._check_market_slippage(symbol, side, used_price)

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

        trade_cfg = self.tradeConfig.get("trade", {})
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
        elif order_type == "limit":
            return self._safe_call(self.exchange.create_order, symbol, "limit", side, float(amt_str), float(px_str), params)
        else:
            raise ValueError(f"Unsupported order_type: {order_type}")

    def create_market_buy(self, symbol: str, amount: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        - 市價買入快捷方法（等價於 create_order(..., side="buy", order_type="market")）。

        Args: 
          - symbol: 交易對
          - amount: base asset amount
          - params: ccxt 額外參數

        Returns : 
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 
          - hint 1: 市價單會套用滑點風控（risk.max_slippage_bps_market）。
        """
        return self.create_order(symbol, "buy", "market", amount, price=None, params=params)

    def create_market_sell(self, symbol: str, amount: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        - 市價賣出快捷方法（等價於 create_order(..., side="sell", order_type="market")）。

        Args: 
          - symbol: 交易對
          - amount: base asset amount
          - params: ccxt 額外參數

        Returns :
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 
          - hint 1: 市價單會套用滑點風控（risk.max_slippage_bps_market）。
        """
        return self.create_order(symbol, "sell", "market", amount, price=None, params=params)

    def create_limit_buy(self, symbol: str, amount: float, price: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        - 限價買入快捷方法（等價於 create_order(..., side="buy", order_type="limit")）。

        Args: 
          - symbol: 交易對
          - amount: base asset amount
          - price: limit price
          - params: ccxt 額外參數（例如 postOnly）

        Returns : 
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 
          - hint 1: 若 risk.require_post_only_for_limit=true 會自動加 postOnly。
        """
        return self.create_order(symbol, "buy", "limit", amount, price=price, params=params)

    def create_limit_sell(self, symbol: str, amount: float, price: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        - 限價賣出快捷方法（等價於 create_order(..., side="sell", order_type="limit")）。

        Args: 
          - symbol: 交易對
          - amount: base asset amount
          - price: limit price
          - params: ccxt 額外參數（例如 postOnly）

        Returns : 
          - order: ccxt unified order dict（或 dry_run mock dict）

        Hint : 
          - hint 1: 若 risk.require_post_only_for_limit=true 會自動加 postOnly。
        """
        return self.create_order(symbol, "sell", "limit", amount, price=price, params=params)

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        - 撤銷單筆訂單。
        - 部分交易所撤單需要 symbol；建議盡量提供 symbol（並在系統內保存 order_id->symbol）。

        Args: 
          - order_id: 訂單 ID
          - symbol: 可選；提供則做 allowlist 檢查並提高撤單成功率

        Returns :
          - result: ccxt 回傳的撤單結果 dict

        Hint : 
          - hint 1: trade.enabled=false 時會阻擋撤單（你也可自行調整策略：撤單仍允許）。
        """
        self._check_trade_enabled()
        if symbol:
            symbol = self._normalize_symbol(symbol)
            self._ensure_symbol_allowed(symbol)
            return self._safe_call(self.exchange.cancel_order, order_id, symbol)
        return self._safe_call(self.exchange.cancel_order, order_id)

    def cancel_all_orders(self, symbol: str) -> Any:
        """

        - 撤銷指定 symbol 的所有 open orders。
        - 若交易所不支援 cancel_all_orders，會 fallback：先 fetch_open_orders 再逐筆 cancel。

        Args:
          - symbol: 交易對

        Returns :
          - result: 若支援 cancel_all_orders，回傳交易所結果；否則回傳逐筆撤單結果 list

        Hint : 
          - hint 1: 若 open orders 很多，逐筆撤單可能較慢，且較容易觸發 rate limit。
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
        """
        - 設定期貨槓桿（futures 才有意義；spot 通常不支援）。
        - 需要你的 exchange class 本身支援 set_leverage（例如 binanceusdm 等）。

        Args: 
          - symbol: 交易對
          - leverage: 槓桿倍數（整數）

        Returns : 回傳內容說明
          - result: ccxt 回傳結果（交易所原始 response 或 unified result）

        Hint : 任何注意事項，設定等需要提醒或特別注意的內容
          - hint 1: 若你用的是 ccxt.binance 且 defaultType=future，不一定代表支援 set_leverage；最好改用對應期貨 exchange class。
        """
        symbol = self._normalize_symbol(symbol)
        self._ensure_symbol_allowed(symbol)
        if not hasattr(self.exchange, "set_leverage"):
            raise NotImplementedError("此 exchange class 不支援 set_leverage（請確認你用的是期貨類型，例如 binanceusdm）。")
        return self._safe_call(self.exchange.set_leverage, leverage, symbol)

    def set_margin_mode(self, margin_mode: str, symbol: Optional[str] = None) -> Any:
        """
        - 設定期貨保證金模式（isolated/cross）。
        - 需要 exchange class 支援 set_margin_mode。

        Args: 
          - margin_mode: "isolated" 或 "cross"
          - symbol: 可選；部分交易所/市場要求指定 symbol

        Returns : 回傳內容說明
          - result: ccxt 回傳結果

        Hint :
          - hint 1: 有些市場/帳戶型態不支援切換，會回傳交易所錯誤；建議捕捉例外並 logger.warning。
        """
        if not hasattr(self.exchange, "set_margin_mode"):
            raise NotImplementedError("此 exchange class 不支援 set_margin_mode。")
        if symbol:
            symbol = self._normalize_symbol(symbol)
            self._ensure_symbol_allowed(symbol)
            return self._safe_call(self.exchange.set_margin_mode, margin_mode, symbol)
        return self._safe_call(self.exchange.set_margin_mode, margin_mode)

    def bootstrap_futures_settings(self, symbol: str) -> None:
        """
        - 期貨啟動時的初始化設定：套用 config 裡的 leverage / margin_mode。
        - 只有 exchange.market_type == "future" 才會執行；spot 直接 return。

        Args: 
          - symbol: 交易對（用於 set_leverage / set_margin_mode）

        Hint : 
          - hint 1: 若交易所不支援對應方法，會 warning 而不中斷（避免初始化直接爆掉）。
        """
        ex_cfg = self.tradeConfig.get("exchange", {})
        if ex_cfg.get("market_type") != "future":
            return

        fut_cfg = self.tradeConfig.get("futures", {})
        lev = int(fut_cfg.get("leverage", 0))
        mm = str(fut_cfg.get("margin_mode", "")).lower()

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
