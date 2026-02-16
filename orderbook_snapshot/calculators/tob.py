from __future__ import annotations

from orderbook_snapshot.domain_types import SnapshotRecord


class TopOfBookCalculator:
    name = "tob"
    version = "1.0.0"

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
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
