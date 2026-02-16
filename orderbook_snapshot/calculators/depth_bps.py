from __future__ import annotations

"""Depth-within-bps 特徵計算。"""

from orderbook_snapshot.domain_types import SnapshotRecord


class DepthWithinBpsCalculator:
    """計算不同 bps 範圍內的 bid/ask 累積量。"""

    name = "depth_within_bps"
    version = "1.0.0"

    def __init__(self, bps_levels: tuple[int, ...] = (5, 10, 25, 50)) -> None:
        """初始化 depth-within-bps 計算器。

        Args:
            bps_levels: 要計算的 bps 清單，例如 `(5, 10, 25, 50)`。
        """
        self.bps_levels = bps_levels

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
        """計算各 bps 範圍內的深度累積量。

        Args:
            snap: 單筆 snapshot。

        Returns:
            `depth_bid_*bps` 與 `depth_ask_*bps` 欄位字典。
        """
        best_bid_p, _ = snap.bids[0]
        best_ask_p, _ = snap.asks[0]
        mid = (best_bid_p + best_ask_p) / 2.0

        if mid <= 0:
            return {f"depth_{side}_{bps}bps": 0.0 for side in ("bid", "ask") for bps in self.bps_levels}

        out: dict[str, float | int | bool] = {}
        for bps in self.bps_levels:
            pct = bps / 10000.0
            bid_cutoff = mid * (1.0 - pct)
            ask_cutoff = mid * (1.0 + pct)

            bid_qty = sum(float(qty) for price, qty in snap.bids if price >= bid_cutoff)
            ask_qty = sum(float(qty) for price, qty in snap.asks if price <= ask_cutoff)

            out[f"depth_bid_{bps}bps"] = bid_qty
            out[f"depth_ask_{bps}bps"] = ask_qty

        return out
