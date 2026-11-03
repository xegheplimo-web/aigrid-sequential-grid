"""Central settings for AIGrid. Everything defaults to OFF / fail-closed.

Loaded from a TOML file (configs_aigrid/aigrid.toml). Missing file -> defaults.
Unknown keys are ignored; invalid values raise at load time (fail-closed).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .capital import CapitalSettings

RunMode = Literal["off", "backtest", "shadow", "dryrun"]


@dataclass(frozen=True)
class RiskLimits:
    max_total_exposure_usdt: float = 500.0
    max_position_notional_usdt: float = 250.0
    max_leverage: float = 3.0
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 8.0
    min_liquidation_buffer_pct: float = 30.0
    max_funding_cost_daily_usdt: float = 5.0
    min_free_margin_usdt: float = 200.0
    data_max_staleness_factor: float = 3.0  # x timeframe seconds
    plan_ttl_minutes: int = 45
    fee_rate_maker: float = 0.0002
    fee_rate_taker: float = 0.0005
    fee_rate_exit: float = 0.0002  # maker — TP exits
    fee_rate_emergency: float = 0.0005  # taker — leg-SL / plan-SL / close-all
    slippage_rate: float = 0.0005

    def validate(self) -> None:
        checks = [
            self.max_total_exposure_usdt > 0,
            0 < self.max_position_notional_usdt <= self.max_total_exposure_usdt,
            1.0 <= self.max_leverage <= 10.0,
            0 < self.max_daily_loss_pct <= 20.0,
            0 < self.max_drawdown_pct <= 50.0,
            0 < self.min_liquidation_buffer_pct <= 90.0,
            self.max_funding_cost_daily_usdt >= 0,
            self.min_free_margin_usdt >= 0,
            self.data_max_staleness_factor >= 1.0,
            1 <= self.plan_ttl_minutes <= 24 * 60,
            0 <= self.fee_rate_maker <= 0.01,
            0 <= self.fee_rate_taker <= 0.02,
            0 <= self.fee_rate_exit <= 0.01,
            0 <= self.fee_rate_emergency <= 0.02,
            0 <= self.slippage_rate <= 0.02,
        ]
        if not all(checks):
            raise ValueError("RiskLimits outside safe bounds")


@dataclass(frozen=True)
class SchedulerSettings:
    periodic_minutes: int = 30  # periodic AI analysis cycle (default 30m, configurable)
    event_cooldown_minutes: int = 10  # min gap between event-driven reanalyses
    max_llm_calls_per_hour: int = 6
    llm_timeout_seconds: float = 40.0  # hard timeout per LLM call
    llm_total_deadline_seconds: float = 120.0  # hard deadline for the whole pipeline
    llm_max_retries: int = 1  # retries only on transient errors
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_minutes: int = 60

    def validate(self) -> None:
        checks = [
            5 <= self.periodic_minutes <= 24 * 60,
            1 <= self.event_cooldown_minutes <= 24 * 60,
            1 <= self.max_llm_calls_per_hour <= 60,
            1 <= self.llm_timeout_seconds <= 900,
            10 <= self.llm_total_deadline_seconds <= 900,
            0 <= self.llm_max_retries <= 3,
            1 <= self.circuit_breaker_failures <= 20,
            5 <= self.circuit_breaker_cooldown_minutes <= 24 * 60,
        ]
        if not all(checks):
            raise ValueError("SchedulerSettings outside safe bounds")


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "openrouter"  # key: OPENROUTER_API_KEY (from .env)
    model: str = "z-ai/glm-5.3-flash"  # GLM 5.3 Flash via OpenRouter
    quick_model: str = "z-ai/glm-5.3-flash"
    backend_url: str | None = None


DEFAULT_WATCHLIST: tuple[str, ...] = (
    "SOL/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "SUI/USDT:USDT",
    "LINK/USDT:USDT",
    "ADA/USDT:USDT",
    "NEAR/USDT:USDT",
)


@dataclass(frozen=True)
class SelectionSettings:
    """Deterministic pair selection + reference guard (không dùng LLM)."""

    execution_timeframe: str = "5m"  # grid candles run on 5m
    trend_timeframes: tuple[str, ...] = ("15m", "1h")  # regime context
    top_candidates: int = 3  # technical scan picks best-N
    max_active_pairs: int = 3  # at most N pairs trade concurrently
    max_ai_analyses_per_cycle: int = 3  # TradingAgents depth per cycle
    min_quote_volume_5m: float = 300_000.0  # USDT quote vol / 5m candle
    min_fee_coverage: float = 1.5  # atr_pct must cover 1.5x round-trip cost
    reference_pairs: tuple[str, ...] = ("BTC/USDT:USDT", "ETH/USDT:USDT")
    ref_dump_threshold_pct: float = 0.015  # -1.5%/1h on BTC/ETH = dump veto

    def validate(self) -> None:
        checks = [
            self.execution_timeframe in ("1m", "3m", "5m", "15m", "1h"),
            all(t in ("15m", "1h", "4h", "1d") for t in self.trend_timeframes),
            1 <= self.top_candidates <= 10,
            1 <= self.max_active_pairs <= 10,
            1 <= self.max_ai_analyses_per_cycle <= 10,
            10_000.0 <= self.min_quote_volume_5m <= 1e9,
            0.5 <= self.min_fee_coverage <= 20.0,
            0.002 <= self.ref_dump_threshold_pct <= 0.2,
        ]
        if not all(checks):
            raise ValueError("SelectionSettings outside safe bounds")


@dataclass(frozen=True)
class AIGridSettings:
    enabled: bool = False  # MASTER SWITCH: everything OFF by default
    mode: RunMode = "off"
    symbols: tuple[str, ...] = ("ETH/USDT:USDT",)
    watchlist: tuple[str, ...] = DEFAULT_WATCHLIST  # pairs the scanner watches
    timeframe: str = "5m"  # EXECUTION timeframe (grid candles)
    data_dir: Path = Path("user_data/data/binance/futures")
    state_dir: Path = Path("user_data/aigrid_state")
    risk: RiskLimits = field(default_factory=RiskLimits)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    capital: CapitalSettings = field(default_factory=CapitalSettings)
    selection: SelectionSettings = field(default_factory=SelectionSettings)

    def validate(self) -> None:
        if self.mode != "off" and not self.enabled:
            raise ValueError("mode != 'off' requires enabled=true")
        if self.mode not in ("off", "backtest", "shadow", "dryrun"):
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.timeframe not in ("1m", "3m", "5m", "15m", "1h", "4h", "1d"):
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        self.risk.validate()
        self.scheduler.validate()
        self.capital.validate()
        self.selection.validate()


def load_settings(path: str | Path | None = None) -> AIGridSettings:
    """Load settings from TOML; absent file or absent [aigrid] table -> all-OFF defaults."""
    if path is None:
        return AIGridSettings()
    config_path = Path(path)
    if not config_path.exists():
        return AIGridSettings()
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)
    root = data.get("aigrid", {})
    if not root.get("enabled", False):
        return AIGridSettings()  # master switch OFF wins over any other keys
    risk = RiskLimits(**{k: v for k, v in (data.get("risk", {}).items())})
    sched = SchedulerSettings(**{k: v for k, v in (data.get("scheduler", {}).items())})
    llm = LLMSettings(**{k: v for k, v in (data.get("llm", {}).items())})
    capital = CapitalSettings(**{k: v for k, v in (data.get("capital", {}).items())})
    sel_raw = dict(data.get("selection", {}))
    if "trend_timeframes" in sel_raw:
        sel_raw["trend_timeframes"] = tuple(sel_raw["trend_timeframes"])
    if "reference_pairs" in sel_raw:
        sel_raw["reference_pairs"] = tuple(sel_raw["reference_pairs"])
    selection = SelectionSettings(**sel_raw)
    settings = AIGridSettings(
        enabled=bool(root.get("enabled", False)),
        mode=root.get("mode", "off"),
        symbols=tuple(root.get("symbols", ["ETH/USDT:USDT"])),
        watchlist=tuple(root.get("watchlist", DEFAULT_WATCHLIST)),
        timeframe=root.get("timeframe", "5m"),
        data_dir=Path(root.get("data_dir", "user_data/data/binance/futures")),
        state_dir=Path(root.get("state_dir", "user_data/aigrid_state")),
        risk=risk,
        scheduler=sched,
        llm=llm,
        capital=capital,
        selection=selection,
    )
    settings.validate()
    return settings
