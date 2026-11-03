"""Capital Manager — cấp ngân sách theo tầng, khống chế bởi tổng vốn.

Luồng vốn (theo chỉ đạo Sếp, 2026-11-03):
    AI xác định cơ hội + chế độ thị trường
        -> CapitalManager cấp ngân sách (margin)
        -> Grid Engine chia tiền theo tầng
        -> Risk Engine có quyền từ chối lệnh

Ví dụ mặc định: equity 1000 USDT -> tổng ngân sách grid 30% = 300 USDT;
một symbol được cấp tối đa 100 USDT margin; 5 tầng -> mỗi tầng 20 USDT margin;
đòn bẩy 2x -> notional mỗi tầng 40 USDT, notional vị thế tối đa 200 USDT.
Vào lệnh theo tầng khi giá chạm — KHÔNG dùng hết ngân sách ngay khi AI báo tốt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapitalSettings:
    total_budget_pct: float = 0.30  # 30% equity cho TOÀN BỘ grid positions
    per_symbol_margin_cap: float = 100.0  # margin tối đa mỗi symbol
    default_leverage: float = 2.0
    min_levels: int = 3
    max_levels: int = 8
    # margin tối thiểu mỗi tầng để vượt minNotional Binance (20 USDT notional / leverage)
    min_level_margin_usdt: float = 10.0

    def validate(self) -> None:
        checks = [
            0.01 <= self.total_budget_pct <= 1.0,
            5.0 <= self.per_symbol_margin_cap <= 10_000.0,
            1.0 <= self.default_leverage <= 10.0,
            2 <= self.min_levels <= self.max_levels <= 20,
            0.5 <= self.min_level_margin_usdt <= 1000.0,
        ]
        if not all(checks):
            raise ValueError("CapitalSettings outside safe bounds")


@dataclass(frozen=True)
class CapitalAllocation:
    symbol: str
    margin_usdt: float  # margin cấp cho symbol này
    leverage: float
    notional_usdt: float  # margin × leverage
    levels: int
    level_margin_usdt: float  # margin mỗi tầng
    level_notional_usdt: float  # margin mỗi tầng × leverage
    budget_remaining_usdt: float  # ngân sách còn lại sau khi cấp

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class CapitalManager:
    """Cấp vốn theo symbol; tổng margin cấp ra không vượt ngân sách tổng."""

    def __init__(self, settings: CapitalSettings, max_leverage: float = 3.0):
        settings.validate()
        self.settings = settings
        self.max_leverage = max_leverage

    def total_budget(self, equity_usdt: float) -> float:
        return equity_usdt * self.settings.total_budget_pct

    def allocate(
        self,
        symbol: str,
        equity_usdt: float,
        allocated_margins: dict[str, float] | None = None,
        levels: int | None = None,
        leverage: float | None = None,
        min_notional: float = 20.0,
    ) -> CapitalAllocation | None:
        """Cấp vốn cho symbol. None = KHÔNG TRADE (không đủ ngân sách tối thiểu).

        allocated_margins: {symbol: margin đã cấp} — dùng để trừ ngân sách còn lại.
        """
        allocated_margins = allocated_margins or {}
        lev = min(leverage or self.settings.default_leverage, self.max_leverage)
        n_levels = max(self.settings.min_levels, min(self.settings.max_levels, levels or 5))
        budget = self.total_budget(equity_usdt)
        used = sum(allocated_margins.values())
        available = budget - used
        margin = min(self.settings.per_symbol_margin_cap, available)
        # Số tầng khả thi = min(tầng yêu cầu, margin/min_margin_mỗi_tầng,
        # margin*lev/minNotional*1.05). Nếu không đủ min_levels -> NO TRADE.
        feasible = min(
            n_levels,
            int(margin / self.settings.min_level_margin_usdt),
            int(margin * lev / (min_notional * 1.05)),
        )
        if feasible < self.settings.min_levels:
            return None  # không đủ cấp tối thiểu -> NO TRADE
        n_levels = feasible
        level_margin = margin / n_levels
        return CapitalAllocation(
            symbol=symbol,
            margin_usdt=round(margin, 2),
            leverage=lev,
            notional_usdt=round(margin * lev, 2),
            levels=n_levels,
            level_margin_usdt=round(level_margin, 2),
            level_notional_usdt=round(level_margin * lev, 2),
            budget_remaining_usdt=round(available - margin, 2),
        )
