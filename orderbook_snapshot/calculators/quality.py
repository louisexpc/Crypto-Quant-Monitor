from __future__ import annotations

from orderbook_snapshot.domain_types import SnapshotRecord


class QualityFlagsCalculator:
    name = "quality_flags"
    version = "1.0.0"

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
        best_bid_p, _ = snap.bids[0]
        best_ask_p, _ = snap.asks[0]

        flag_crossed_book = best_bid_p >= best_ask_p

        flag_bad_sorting = False
        for (p0, _), (p1, _) in zip(snap.bids, snap.bids[1:]):
            if p0 < p1:
                flag_bad_sorting = True
                break
        if not flag_bad_sorting:
            for (p0, _), (p1, _) in zip(snap.asks, snap.asks[1:]):
                if p0 > p1:
                    flag_bad_sorting = True
                    break

        flag_non_positive_qty = any(qty <= 0 for _, qty in snap.bids) or any(qty <= 0 for _, qty in snap.asks)
        flag_depth_insufficient = snap.depth < 1000

        return {
            "flag_crossed_book": flag_crossed_book,
            "flag_bad_sorting": flag_bad_sorting,
            "flag_depth_insufficient": flag_depth_insufficient,
            "flag_non_positive_qty": flag_non_positive_qty,
        }
