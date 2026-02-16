from __future__ import annotations

"""Top-of-Book 與 micro-structure 基礎特徵。"""

from orderbook_snapshot.domain_types import SnapshotRecord


class TopOfBookCalculator:
    """計算 TOB 與一階市場微觀狀態特徵。"""

    name = "tob"
    version = "1.0.0"

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
        """從最佳買一/賣一產生 TOB 特徵。

        Args:
            snap: 單筆 snapshot。

        Returns:
            包含 best bid/ask、spread、mid、imbalance、microprice 的特徵字典。
        """
        best_bid_p, best_bid_q = snap.bids[0]
        best_ask_p, best_ask_q = snap.asks[0]

        spread = float(best_ask_p - best_bid_p)
        mid = float((best_ask_p + best_bid_p) / 2.0)

        qty_sum = best_bid_q + best_ask_q
        imbalance_l1 = 0.0 if qty_sum == 0 else float((best_bid_q - best_ask_q) / qty_sum)

        microprice = None
        if qty_sum > 0:
            microprice = float((best_ask_p * best_bid_q + best_bid_p * best_ask_q) / qty_sum)

        return {
            "best_bid_p": float(best_bid_p),
            "best_bid_q": float(best_bid_q),
            "best_ask_p": float(best_ask_p),
            "best_ask_q": float(best_ask_q),
            "spread": spread,
            "mid": mid,
            "imbalance_l1": imbalance_l1,
            "microprice": microprice,
        }
