from __future__ import annotations

from orderbook_snapshot.domain_types import SnapshotRecord


class BpsBinsCalculator:
    name = "bps_bins"
    version = "1.0.0"

    def __init__(self, bins: tuple[tuple[int, int], ...] = ((0, 5), (5, 10), (10, 25), (25, 50))) -> None:
        self.bins = bins

    @staticmethod
    def _in_bin(value: float, left: int, right: int, include_right: bool) -> bool:
        if include_right:
            return left <= value <= right
        return left <= value < right

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
        best_bid_p, _ = snap.bids[0]
        best_ask_p, _ = snap.asks[0]
        mid = (best_bid_p + best_ask_p) / 2.0

        out: dict[str, float | int | bool] = {}
        if mid <= 0:
            for left, right in self.bins:
                out[f"bin_{left}_{right}bps_bid_qty"] = 0.0
                out[f"bin_{left}_{right}bps_ask_qty"] = 0.0
            return out

        bid_bps_with_qty = [(((mid - price) / mid) * 10000.0, float(qty)) for price, qty in snap.bids]
        ask_bps_with_qty = [(((price - mid) / mid) * 10000.0, float(qty)) for price, qty in snap.asks]

        for idx, (left, right) in enumerate(self.bins):
            include_right = idx == len(self.bins) - 1
            bid_qty = sum(qty for bps, qty in bid_bps_with_qty if self._in_bin(bps, left, right, include_right))
            ask_qty = sum(qty for bps, qty in ask_bps_with_qty if self._in_bin(bps, left, right, include_right))
            out[f"bin_{left}_{right}bps_bid_qty"] = bid_qty
            out[f"bin_{left}_{right}bps_ask_qty"] = ask_qty

        return out
