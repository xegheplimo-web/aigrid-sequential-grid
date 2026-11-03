"""Backtest harness — runs the adaptive grid over historical candles.

Configs (spec section XIV):
  C. Adaptive Grid WITHOUT AI  — regime engine + deterministic plans (no LLM)
  D. TradingAgents + Adaptive Grid — LLM assessments drive plans (needs API key;
     without a key it degrades to C behavior + records AI_UNAVAILABLE)

Accounting (single source of truth = VirtualGridEngine):
  equity(t) = initial + engine.realized_pnl + engine.unrealized_pnl(price)
              - funding_paid
  engine.realized_pnl already includes entry+exit fees and slippage.

No-lookahead: every decision sees only candles closed at or before its time.
Funding: applied every funding row (8h) on open inventory, from the existing
funding feather. Liquidation approximated by plan-SL + leg-SL (documented).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from .capital import CapitalManager
from .data.binance_adapter import TF_MS, BinanceFuturesDataAdapter
from .grid.engine import VirtualGridEngine
from .plans.schema import AIGridPlan, GridMode, draft_plan_from_assessment
from .regime.controller import RegimeController
from .regime.engine import (
    DOWNTREND,
    RANGE,
    UPTREND,
    RegimeThresholds,
    compute_features,
    decide_from_features,
)
from .settings import AIGridSettings


@dataclass
class BacktestTrade:
    open_ms: int
    close_ms: int
    side: str
    level: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fees: float
    reason: str


@dataclass
class BacktestResult:
    pair: str
    timeframe: str
    start: datetime
    end: datetime
    config: str
    initial_usdt: float
    final_equity_usdt: float
    net_profit_usdt: float
    net_profit_pct: float
    max_drawdown_pct: float
    sharpe: float
    win_rate: float
    trade_count: int
    profit_factor: float
    total_fees_usdt: float
    total_funding_usdt: float
    regime_transitions: int
    plans_adopted: int = 0
    ai_calls: int = 0
    ai_cost_usdt: float = 0.0
    ai_unavailable_cycles: int = 0
    runtime_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[tuple[int, float]] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "config": self.config,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "initial_usdt": self.initial_usdt,
            "final_equity_usdt": round(self.final_equity_usdt, 2),
            "net_profit_usdt": round(self.net_profit_usdt, 2),
            "net_profit_pct": round(self.net_profit_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe": round(self.sharpe, 3),
            "win_rate": round(self.win_rate, 4),
            "trade_count": self.trade_count,
            "profit_factor": round(self.profit_factor, 3)
            if math.isfinite(self.profit_factor)
            else None,
            "total_fees_usdt": round(self.total_fees_usdt, 2),
            "total_funding_usdt": round(self.total_funding_usdt, 2),
            "regime_transitions": self.regime_transitions,
            "plans_adopted": self.plans_adopted,
            "ai_calls": self.ai_calls,
            "ai_cost_usdt": round(self.ai_cost_usdt, 4),
            "ai_unavailable_cycles": self.ai_unavailable_cycles,
            "runtime_s": round(self.runtime_s, 1),
            "notes": self.notes,
        }


class RecordingGridEngine(VirtualGridEngine):
    """VirtualGridEngine + trade recording via fill hooks (exact pnl deltas)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trades: list[BacktestTrade] = []
        self._open_meta: dict[int, dict] = {}
        self._prev_realized: float = 0.0

    def _fill_entry(self, level: int, price: float, qty: float, candle_ms: int) -> None:
        super()._fill_entry(level, price, qty, candle_ms)
        self._open_meta[level] = {
            "open_ms": candle_ms,
            "side": "short" if self.is_short else "long",
            "entry_price": self.state.lots[level].entry_price,
            "entry_fee": self.state.lots[level].entry_fee,
        }

    def _fill_exit(self, level: int, price: float, lot) -> None:
        before = self.state.realized_pnl
        meta = self._open_meta.pop(level, None)
        super()._fill_exit(level, price, lot)
        pnl_delta = self.state.realized_pnl - before
        if meta is not None:
            self.trades.append(
                BacktestTrade(
                    open_ms=meta["open_ms"],
                    close_ms=0,
                    side=meta["side"],
                    level=level,
                    entry_price=meta["entry_price"],
                    exit_price=price,
                    qty=lot.qty,
                    pnl=pnl_delta,
                    fees=meta["entry_fee"] + lot.qty * price * self.fee_rate,
                    reason="GRID_EXIT",
                )
            )


class GridBacktester:
    """Candle-loop backtest of the adaptive grid (config C or D)."""

    def __init__(
        self,
        settings: AIGridSettings,
        config: str = "C",
        leg_stop_loss_pct: float = 0.03,
        levels: int = 6,
        assess_fn=None,
    ):
        """assess_fn(pair, ctx:dict, as_of_ms) -> MarketAssessment | None.

        Config D only: called at plan-adoption points. None/exception/
        not-ok -> degrade to deterministic (config C) behavior for that
        cycle and count ai_unavailable_cycles. NEVER blocks the grid loop.
        """
        if config not in ("C", "D"):
            raise ValueError("config must be 'C' or 'D'")
        self.settings = settings
        self.config = config
        self.adapter = BinanceFuturesDataAdapter(settings.data_dir)
        self.limits = settings.risk
        self.leg_sl = leg_stop_loss_pct
        self.levels = levels
        self.capital = CapitalManager(settings.capital, max_leverage=settings.risk.max_leverage)
        self.assess_fn = assess_fn

    # ------------------------------------------------------------------ run
    def run(
        self, pair: str, start: datetime, end: datetime, initial_usdt: float = 1000.0
    ) -> BacktestResult:
        t0 = time.monotonic()
        tf = self.settings.timeframe
        tf_ms = TF_MS[tf]
        ohlcv = self.adapter.get_ohlcv(pair, tf, as_of=end, lookback=10**9)
        ohlcv = ohlcv[ohlcv.index >= pd.Timestamp(start)]
        if len(ohlcv) < 200:
            raise ValueError(f"Insufficient candles for {pair} {tf}: {len(ohlcv)}")
        funding = self.adapter.get_funding(pair, as_of=end)

        # Precompute causal features ONCE (no lookahead: ewm/rolling are causal).
        feats = compute_features(ohlcv)
        th = RegimeThresholds()

        peak = initial_usdt
        max_dd = 0.0
        transitions = 0
        plans_adopted = 0
        ai_calls = 0
        ai_unavailable_cycles = 0
        last_ai_ms = 0
        last_assessment = None  # latest OK AI assessment (persists)
        ai_period_ms = self.settings.scheduler.periodic_minutes * 60_000
        funding_paid = 0.0
        last_funding_ms = 0
        equity_curve: list[tuple[int, float]] = []

        self.controller = RegimeController(confirm_candles=3, switch_cooldown_candles=6)

        engine: RecordingGridEngine | None = None
        plan: AIGridPlan | None = None
        plan_expiry_ms = 0
        pending_plan: AIGridPlan | None = None
        total_realized = 0.0  # accumulated across engine swaps (never lost)
        all_trades: list[BacktestTrade] = []

        for i in range(len(ohlcv)):
            idx = ohlcv.index[i]
            row = ohlcv.iloc[i]
            open_ms = int(idx.value // 1_000_000)
            close_ms = open_ms + tf_ms
            o, h, l, c = (
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
            )

            # ---- regime from precomputed features at index i (causal, no lookahead)
            frow = feats.iloc[i]
            v = {
                k: float(frow[k])
                for k in (
                    "adx",
                    "plus_di",
                    "minus_di",
                    "atr_pct",
                    "bbw",
                    "slope",
                    "qv",
                    "ema20_gt_ema50",
                )
            }
            rr = decide_from_features(v, th, timeframe_seconds=tf_ms // 1000)
            prev = self.controller.confirmed
            self.controller.update(rr.regime, i)
            if self.controller.confirmed != prev:
                transitions += 1

            # ---- plan expiry: RENEW when the regime is unchanged (keeps the
            # engine and its lots alive — no churn); drop when regime moved.
            if plan is not None and close_ms >= plan_expiry_ms:
                if self.controller.confirmed == plan.market_regime and not self.controller.paused:
                    plan_expiry_ms = close_ms + int(plan.ttl_seconds() * 1000)
                else:
                    plan = None
                    plan_expiry_ms = 0

            # ---- pending swap: engine drained -> adopt the queued plan
            if pending_plan is not None and engine is not None and not engine.state.lots:
                plan = pending_plan
                plan_expiry_ms = close_ms + int(plan.ttl_seconds() * 1000)
                total_realized += engine.state.realized_pnl
                all_trades.extend(engine.trades)
                engine = RecordingGridEngine(
                    plan,
                    fee_rate=self.limits.fee_rate_maker,
                    slippage_rate=self.limits.slippage_rate,
                    leg_stop_loss_pct=self.leg_sl,
                    max_inventory_notional=plan.max_position_size,
                )
                plans_adopted += 1
                pending_plan = None

            # ---- adopt a plan when none is active and the regime wants one
            if plan is None and not self.controller.paused:
                regime = self.controller.confirmed
                mode = {RANGE: "RANGE_GRID", UPTREND: "LONG_GRID", DOWNTREND: "SHORT_GRID"}.get(
                    regime, "SAFE_HOLD"
                )
                proposal = None
                if mode != "SAFE_HOLD" and self.config == "D":
                    # ---- AI cycle (config D): refreshed on the periodic gate
                    # (never per-candle); the LATEST assessment persists and
                    # applies to every adoption point until the next refresh —
                    # an AI veto (SAFE_HOLD) therefore sticks, and an AI outage
                    # degrades to deterministic C behavior for that stretch.
                    if close_ms - last_ai_ms >= ai_period_ms:
                        ai_calls += 1
                        last_ai_ms = close_ms
                        try:
                            ctx = {
                                "pair": pair,
                                "timeframe": tf,
                                "as_of_utc": datetime.fromtimestamp(
                                    close_ms / 1000, tz=UTC
                                ).isoformat(),
                                "last_close": round(c, 8),
                                "indicators": {
                                    k: round(float(frow[k]), 6)
                                    for k in ("adx", "atr_pct", "bbw", "slope")
                                },
                            }
                            assessment = (
                                self.assess_fn(pair, ctx, close_ms)
                                if self.assess_fn is not None
                                else None
                            )
                        except Exception:
                            assessment = None
                        if assessment is not None and assessment.ok:
                            last_assessment = assessment
                        else:
                            ai_unavailable_cycles += 1  # AI down -> C behavior
                            last_assessment = None
                    if last_assessment is not None and last_assessment.ok:
                        proposal = last_assessment.proposal
                        regime = last_assessment.regime
                        mode = {
                            RANGE: "RANGE_GRID",
                            UPTREND: "LONG_GRID",
                            DOWNTREND: "SHORT_GRID",
                        }.get(regime, "SAFE_HOLD")
                if mode != "SAFE_HOLD":
                    # Capital Manager cấp vốn theo equity hiện tại (30% budget,
                    # cap mỗi symbol) — KHÔNG phải theo ý muốn của AI
                    equity_now = (
                        initial_usdt
                        + total_realized
                        + (engine.unrealized_pnl(c) if engine is not None else 0.0)
                        - funding_paid
                    )
                    allocated = {}
                    if engine is not None and engine.plan.grid_mode != GridMode.SAFE_HOLD:
                        allocated[pair] = engine.plan.max_capital
                    allocation = self.capital.allocate(
                        pair, equity_now, allocated, levels=self.levels
                    )
                    if allocation is not None:
                        new_plan = draft_plan_from_assessment(
                            pair,
                            regime,
                            rr.confidence,
                            c,
                            rr.atr_pct * c,
                            grid_mode=mode,
                            ttl_minutes=self.limits.plan_ttl_minutes,
                            margin_usdt=allocation.margin_usdt,
                            leverage=allocation.leverage,
                            levels=allocation.levels,
                            level_margin_usdt=allocation.level_margin_usdt,
                            grid_lower=(proposal.grid_lower if proposal else None),
                            grid_upper=(proposal.grid_upper if proposal else None),
                            reasoning=f"backtest config {self.config} regime={regime}",
                        )
                        if engine is None:
                            plan = new_plan
                            plan_expiry_ms = close_ms + int(new_plan.ttl_seconds() * 1000)
                            engine = RecordingGridEngine(
                                plan,
                                fee_rate=self.limits.fee_rate_maker,
                                slippage_rate=self.limits.slippage_rate,
                                leg_stop_loss_pct=self.leg_sl,
                                max_inventory_notional=plan.max_position_size,
                            )
                            plans_adopted += 1
                        elif engine.plan.grid_mode == new_plan.grid_mode:
                            # same mode: keep the engine (lots + levels), renew plan
                            plan = new_plan
                            plan_expiry_ms = close_ms + int(new_plan.ttl_seconds() * 1000)
                        elif not engine.state.lots:
                            plan = new_plan
                            plan_expiry_ms = close_ms + int(new_plan.ttl_seconds() * 1000)
                            total_realized += engine.state.realized_pnl
                            all_trades.extend(engine.trades)
                            engine = RecordingGridEngine(
                                plan,
                                fee_rate=self.limits.fee_rate_maker,
                                slippage_rate=self.limits.slippage_rate,
                                leg_stop_loss_pct=self.leg_sl,
                                max_inventory_notional=plan.max_position_size,
                            )
                            plans_adopted += 1
                        else:
                            pending_plan = new_plan  # drain-then-switch

            # ---- engine step (exits always managed; entries only with an active plan)
            if engine is not None:
                permit = plan is not None and not self.controller.paused
                engine.on_candle(open_ms, o, h, l, c, permit_entry=permit)

            # ---- funding every funding row on open inventory
            if len(funding):
                cut_lo = (
                    pd.Timestamp(last_funding_ms, unit="ms", tz=UTC) if last_funding_ms else None
                )
                cut_hi = pd.Timestamp(close_ms, unit="ms", tz=UTC)
                mask = (
                    (funding.index <= cut_hi)
                    if cut_lo is None
                    else ((funding.index > cut_lo) & (funding.index <= cut_hi))
                )
                for fts, frow in funding[mask].iterrows():
                    if engine is not None and engine.inventory_notional(c) > 0:
                        rate = float(frow["funding_rate"])
                        sign = -1.0 if engine.is_short else 1.0
                        funding_paid += sign * rate * engine.inventory_notional(c)
                    last_funding_ms = int(fts.value // 1_000_000)

            # ---- equity + drawdown (realized accumulated across engine swaps)
            unreal = engine.unrealized_pnl(c) if engine is not None else 0.0
            realized = total_realized + (engine.state.realized_pnl if engine is not None else 0.0)
            equity = initial_usdt + realized + unreal - funding_paid
            peak = max(peak, equity)
            dd = 100.0 * (peak - equity) / max(peak, 1e-9)
            max_dd = max(max_dd, dd)
            equity_curve.append((close_ms, equity))

        runtime = time.monotonic() - t0
        final_equity = equity_curve[-1][1] if equity_curve else initial_usdt
        net = final_equity - initial_usdt
        trades = list(all_trades)
        if engine is not None:
            trades.extend(engine.trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        eq = pd.Series([e[1] for e in equity_curve])
        returns = eq.pct_change().dropna()
        # annualization by timeframe
        periods_per_year = 365 * 24 * 3600_000 / tf_ms
        sharpe = (
            float(returns.mean() / returns.std() * math.sqrt(periods_per_year))
            if len(returns) > 2 and returns.std() > 0
            else 0.0
        )
        total_fees = sum(t.fees for t in trades)
        return BacktestResult(
            pair=pair,
            timeframe=tf,
            start=start,
            end=end,
            config=self.config,
            initial_usdt=initial_usdt,
            final_equity_usdt=final_equity,
            net_profit_usdt=net,
            net_profit_pct=100 * net / initial_usdt,
            max_drawdown_pct=max_dd,
            sharpe=sharpe,
            win_rate=len(wins) / len(trades) if trades else 0.0,
            trade_count=len(trades),
            profit_factor=gross_win / gross_loss if gross_loss > 0 else float("inf"),
            total_fees_usdt=total_fees,
            total_funding_usdt=funding_paid,
            regime_transitions=transitions,
            plans_adopted=plans_adopted,
            ai_calls=ai_calls,
            ai_unavailable_cycles=ai_unavailable_cycles,
            runtime_s=runtime,
            trades=trades,
            equity_curve=equity_curve,
        )
