"""R14 tests — DeterministicCycle + service dry-run wiring + hermetic chain.

The cycle is the card's ``closed-candle -> REGIME check -> deterministic
plan -> guarded atomic publish`` driver. Unit slices use a duck-typed
orchestrator stand-in; the hermetic test runs the REAL
``AIGridOrchestrator.process_candle`` / controller / validator / atomic
store and finishes inside the REAL ``AIGridStrategy`` (imported with the
same stub pattern as test_r11) — no file in ``user_data/`` is touched.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from aigrid.exchange_info import ExchangeInfo
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.plans.validator import PlanValidator
from aigrid.publish import DeterministicCycle, PlanPublisher, RuntimeFacts
from aigrid.risk.ownership import OwnershipRegistry
from aigrid.service import AIGridService
from aigrid.settings import AIGridSettings, RiskLimits
from aigrid.state.store import EventLog, StateStore
from tests._attest import (
    facts,
    make_config,
    runtime_config,
    write_integration,
    write_secret,
)

ROOT = Path(__file__).resolve().parents[1]
EI_PATH = ROOT / ".orchestrator/fapi_exchangeinfo.json"
DATA_DIR = ROOT / "user_data/data/binance/futures"
STRATEGY_PATH = ROOT / "user_data/strategies/AIGridStrategy.py"
PAIR = "ETH/USDT:USDT"
NOW_MS = int(datetime(2026, 11, 3, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)


class _Events:
    def __init__(self):
        self.rows: list[tuple[str, str, dict]] = []

    def log(self, kind: str, symbol: str = "", payload: dict | None = None):
        self.rows.append((kind, symbol, payload or {}))

    def kinds(self) -> list[str]:
        return [k for k, _, _ in self.rows]


def _df(
    n: int = 600,
    end_ms: int = NOW_MS - 300_000,
    step_ms: int = 300_000,
    base: float = 100.0,
    amp: float = 0.012,
) -> pd.DataFrame:
    """Ranging 5m candles; the last row is a CLOSED candle (open_time +
    300s <= the clock the publisher runs on)."""
    idx = [end_ms - step_ms * (n - 1 - i) for i in range(n)]
    rows = []
    for i, t in enumerate(idx):
        close = base * (1.0 + amp * math.sin(i * 0.35))
        open_ = base * (1.0 + amp * math.sin((i - 1) * 0.35))
        rows.append((open_, max(open_, close) * 1.001, min(open_, close) * 0.999, close, 5000.0))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(idx, unit="ms", utc=True)
    return df


def _facts(**kw) -> RuntimeFacts:
    args = {
        "dry_run": True,
        "trading_mode": "futures",
        "runmode": "dry_run",
        "pair_whitelist": (PAIR,),
        "max_open_trades": 3,
        "open_trades": [],
        "exchange_healthy": True,
        "server_time_ms": NOW_MS,
    }
    args.update(kw)
    args["runtime_config"] = runtime_config()
    return facts(**args)


class _FakeOrch:
    """Duck-typed orchestrator for the cycle: records process_candle calls
    and (optionally) clobbers the integration file like persist() does."""

    def __init__(
        self,
        store: StateStore,
        settings: AIGridSettings,
        events,
        *,
        recover_status: str = "FRESH_START",
        confirmed: str = "RANGE",
        paused: bool = False,
        clobber: bool = False,
        explode: bool = False,
    ):
        self.settings = settings
        self.store = store
        self.eventlog = events
        self.ownership = OwnershipRegistry()
        self.validator = PlanValidator(ExchangeInfo(EI_PATH), RiskLimits())
        self.preflight_ok = True
        self.recover_status = recover_status
        self.clobber = clobber
        self.explode = explode
        self.calls: list[dict] = []
        self.recover_args: list[tuple] = []
        self._ctrl = SimpleNamespace(confirmed=confirmed, paused=paused, pause_reason="")

    def process_candle(self, pair, candle, *, history_df=None, force_no_entry=False):
        if self.explode:
            raise RuntimeError("engine blew up")
        self.calls.append(
            {
                "pair": pair,
                "candle": dict(candle),
                "hist": len(history_df) if history_df is not None else 0,
            }
        )
        if self.clobber:
            # orchestrator.persist() writes ft_integration.json with the
            # engine view of the world — it must not strand our plan.
            self.store.save_state("ft_integration", {"enabled": False, "plans": {}})

    def recover(self, actual_open_pairs, actual_open_order_ids, actual_open_lots=None):
        self.recover_args.append(
            (
                set(actual_open_pairs),
                set(actual_open_order_ids),
                dict(actual_open_lots or {}),
            )
        )
        return {"status": self.recover_status, "reasons": []}

    def _controller_for(self, pair):
        return self._ctrl

    def _price_tick(self, pair):
        return 0.01


def _settings(tmp_path, **kw) -> AIGridSettings:
    base = {
        "enabled": True,
        "mode": "dryrun",
        "symbols": (PAIR,),
        "timeframe": "5m",
        "data_dir": DATA_DIR,
        "state_dir": tmp_path / "state",
    }
    base.update(kw)
    return AIGridSettings(**base)


def _cycle(
    tmp_path,
    orch=None,
    *,
    loader=None,
    facts=None,
    clock=None,
    events=None,
    whitelist=(PAIR,),
    orch_kw=None,
):
    events = events if events is not None else _Events()
    settings = _settings(tmp_path)
    store = StateStore(tmp_path / "state")
    if orch is None:
        orch = _FakeOrch(store, settings, events, **(orch_kw or {}))
    write_secret(settings.state_dir)
    publisher = PlanPublisher(
        store,
        events,
        validator=orch.validator,
        ownership=orch.ownership,
        settings=settings,
        throttle_s=0.0,
        price_tick_of=orch._price_tick,
        clock=clock or (lambda: NOW_MS / 1000.0),
    )
    holder = {"df": loader if isinstance(loader, pd.DataFrame) else _df()}
    cyc = DeterministicCycle(
        orch,
        publisher,
        candle_loader=loader if callable(loader) else (lambda pair: holder["df"]),
        facts_provider=(lambda: facts) if facts is not None else (lambda: _facts()),
        eventlog=events,
        whitelist=whitelist,
    )
    return cyc, orch, holder, events


def _integration(state_dir: Path) -> dict | None:
    path = state_dir / "ft_integration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ candle pump
def test_tick_processes_each_new_closed_candle_oldest_first(tmp_path):
    """A 3-candle backlog -> process_candle fires once per candle, oldest
    first, each with a causal history slice (no lookahead)."""
    cyc, orch, holder, _events = _cycle(tmp_path)
    cyc._reconciled = True
    cyc.tick()
    assert len(orch.calls) == 600  # backlog processed fully
    opens = [c["candle"]["open_ms"] for c in orch.calls]
    assert opens == sorted(opens)  # oldest first
    # causal slice: the k-th call sees exactly k+1 rows of history
    assert orch.calls[-1]["hist"] == 600
    holder["df"] = _df(n=603, end_ms=NOW_MS + 600_000)  # 3 newer candles
    orch.calls.clear()
    cyc.publisher._clock = lambda: (NOW_MS + 900_000) / 1000.0
    cyc.tick()
    assert [c["candle"]["open_ms"] for c in orch.calls] == [
        int(holder["df"].index[i].value // 1_000_000) for i in (-3, -2, -1)
    ]


def test_tick_no_new_candle_is_a_noop(tmp_path):
    cyc, orch, _holder, _ev = _cycle(tmp_path)
    cyc._reconciled = True
    cyc.tick()
    n = len(orch.calls)
    out = cyc.tick()
    assert len(orch.calls) == n and out == []


def test_tick_whitelist_drives_which_pairs_are_pumped(tmp_path):
    cyc, orch, _holder, _ev = _cycle(tmp_path, loader=_df(), whitelist=("SOL/USDT:USDT",))
    cyc._reconciled = True
    cyc.tick()
    assert {c["pair"] for c in orch.calls} == {"SOL/USDT:USDT"}
    assert "SOL/USDT:USDT" in cyc.seen_ts


# ------------------------------------------------------------------ reconciliation gate
def test_publish_blocked_until_startup_reconcile_pass(tmp_path):
    """D-102 (1): publish is refused before the startup reconciliation
    runs; a PASS arms the gate for the next new candle."""
    events = _Events()
    cyc, _orch, holder, ev = _cycle(tmp_path, events=events)
    cyc.tick()  # _reconciled False
    blocked = [p for k, _, p in ev.rows if k == "PUBLISH_BLOCKED"]
    assert blocked and "STARTUP_RECONCILIATION_REQUIRED" in blocked[0]["reasons"]
    assert _integration(tmp_path / "state") is None

    res = cyc.startup_reconcile()
    assert res["status"] == "FRESH_START"
    assert "STARTUP_RECONCILIATION_PASS" in ev.kinds()

    holder["df"] = _df(n=601, end_ms=NOW_MS)  # 1 newer candle
    cyc.publisher._clock = lambda: (NOW_MS + 300_000) / 1000.0
    cyc._facts_provider = lambda: _facts(server_time_ms=NOW_MS + 300_000)
    out = cyc.tick()
    assert out and out[0].status == "PUBLISHED", out
    assert _integration(tmp_path / "state")["plans"]


def test_startup_reconcile_safe_hold_keeps_gate_closed(tmp_path):
    events = _Events()
    cyc, _orch, _holder, ev = _cycle(
        tmp_path, events=events, orch_kw={"recover_status": "SAFE_HOLD"}
    )
    res = cyc.startup_reconcile(actual_open_pairs=set(), actual_open_order_ids=set())
    assert res["status"] == "SAFE_HOLD"
    assert cyc.reconciled is False
    assert "STARTUP_RECONCILIATION_FAILED" in ev.kinds()
    cyc.tick()
    payload = _integration(tmp_path / "state")
    assert payload is None or not payload.get("plans")


def test_seen_and_reconcile_survive_restart(tmp_path):
    """publish_cycle.json persists the last-seen candle + reconcile flag —
    a restarted cycle processes only candles newer than the marker."""
    cyc, _orch, _holder, _ev = _cycle(tmp_path)
    cyc._reconciled = True
    cyc.tick()
    # restart: fresh cycle over the same state dir
    cyc2, orch2, holder2, _ev2 = _cycle(tmp_path)
    assert cyc2.reconciled is True  # armed flag persisted
    assert cyc2.seen_ts[PAIR] == cyc.seen_ts[PAIR]
    holder2["df"] = _df(n=601, end_ms=NOW_MS)
    cyc2.publisher._clock = lambda: (NOW_MS + 300_000) / 1000.0
    cyc2.tick()
    assert len(orch2.calls) == 1  # only the new candle


# ------------------------------------------------------------------ process failure
def test_process_candle_failure_is_fail_closed(tmp_path):
    """A crashing process_candle: event logged, seen NOT advanced past the
    failure, and no plan publish is attempted on a half-processed
    backlog."""
    events = _Events()
    cyc, _orch, _holder, ev = _cycle(tmp_path, events=events, orch_kw={"explode": True})
    cyc._reconciled = True
    out = cyc.tick()
    assert out == []
    assert "CANDLE_PROCESS_FAILED" in ev.kinds()
    assert PAIR not in cyc.seen_ts
    assert _integration(tmp_path / "state") is None


def test_regime_gate_skips_plan_publish(tmp_path):
    """Controller confirmed != RANGE -> PLAN_SKIPPED_REGIME, no file."""
    events = _Events()
    cyc, _orch, _holder, ev = _cycle(
        tmp_path,
        events=events,
        orch_kw={"confirmed": "HIGH_VOLATILITY", "paused": True},
    )
    cyc._reconciled = True
    cyc.tick()
    assert "PLAN_SKIPPED_REGIME" in ev.kinds()
    assert _integration(tmp_path / "state") is None


# ------------------------------------------------------------------ clobber restore
def test_cycle_restores_integration_after_persist_clobber(tmp_path):
    """persist() inside process_candle rewrites ft_integration.json with an
    engine view (possibly plans={}). At end of tick the publisher re-syncs
    the ledger's active plans — the file can never stay stale."""
    events = _Events()
    cyc, orch, holder, _ev = _cycle(tmp_path, events=events, orch_kw={"clobber": True})
    cyc._reconciled = True
    cyc.tick()
    good = _integration(tmp_path / "state")
    assert good["plans"], "publish expected after tick 1"
    plan_payload = dict(good["plans"][PAIR])

    # next candle: process clobbers the file, regime then pauses -> no
    # publish this tick; sync must still restore the active plan.
    holder["df"] = _df(n=601, end_ms=NOW_MS)
    orch._ctrl.paused = True
    cyc.tick()
    restored = _integration(tmp_path / "state")
    assert restored["plans"][PAIR] == plan_payload


# ================================================================== service wiring
class _SvcOrch:
    def __init__(self, state_dir: Path, mode: str, enabled: bool = True):
        self.settings = SimpleNamespace(state_dir=state_dir, enabled=enabled, mode=mode)
        self.eventlog = EventLog(state_dir)
        self.preflight_ok = True


class _FakeCycle:
    def __init__(self):
        self.reconciles = 0
        self.ticks: list[int] = []
        self.seen_ts: dict[str, int] = {}
        self.raise_in_tick = False

    def startup_reconcile(self, **kw):
        self.reconciles += 1

    def tick(self, cycle_no: int = 0):
        if self.raise_in_tick:
            raise RuntimeError("tick blew up")
        self.ticks.append(cycle_no)
        return []


class _Clock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.t = start

    def __call__(self) -> float:
        self.t += 30.0
        return self.t


def test_service_drives_cycle_only_in_dryrun(tmp_path):
    """The deterministic chain is wired ONLY into the dryrun branch —
    shadow/off modes stay heartbeat-only even when a cycle is injected."""
    cyc = _FakeCycle()
    orch = _SvcOrch(tmp_path / "s1", mode="dryrun")
    svc = AIGridService(
        orch,
        interval_s=0,
        clock=_Clock(),
        cycle=cyc,
        on_cycle=lambda c: c >= 2 and svc.stop(),
    )
    assert svc.run() == 0
    assert cyc.reconciles == 1
    assert cyc.ticks == [1, 2]

    cyc2 = _FakeCycle()
    orch2 = _SvcOrch(tmp_path / "s2", mode="shadow")
    svc2 = AIGridService(
        orch2,
        interval_s=0,
        clock=_Clock(),
        cycle=cyc2,
        on_cycle=lambda c: c >= 2 and svc2.stop(),
    )
    assert svc2.run() == 0
    assert cyc2.reconciles == 0
    assert cyc2.ticks == []
    orch2.eventlog.close()


def test_service_cycle_tick_error_is_logged_and_loop_continues(tmp_path):
    cyc = _FakeCycle()
    cyc.raise_in_tick = True
    orch = _SvcOrch(tmp_path / "s", mode="dryrun")
    svc = AIGridService(
        orch,
        interval_s=0,
        clock=_Clock(),
        cycle=cyc,
        on_cycle=lambda c: c >= 2 and svc.stop(),
    )
    assert svc.run() == 0
    kinds = [e["kind"] for e in orch.eventlog.recent()]
    assert "CYCLE_TICK_ERROR" in kinds


def test_service_heartbeat_mirrors_candle_ts(tmp_path):
    cyc = _FakeCycle()
    cyc.seen_ts = {PAIR: NOW_MS}
    orch = _SvcOrch(tmp_path / "s", mode="dryrun")
    svc = AIGridService(
        orch, interval_s=0, clock=_Clock(), cycle=cyc, on_cycle=lambda c: svc.stop()
    )
    assert svc.run() == 0
    hb = json.loads((tmp_path / "s" / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["candle_ts_ms"] == {PAIR: NOW_MS}


# ================================================================== HERMETIC chain
def _install_ft_stubs() -> None:
    """Minimal freqtrade/talib surface so AIGridStrategy.py imports (same
    pattern as test_r11 — the strategy file itself is never modified)."""
    if "freqtrade" in sys.modules:
        return

    def _sl(stop_rate, current_rate, is_short=False, leverage=1.0):
        if current_rate == 0:
            return 1
        sl = 1 - (stop_rate / current_rate)
        if is_short:
            sl = -sl
        return max(sl, 0.0) * leverage

    freqtrade = types.ModuleType("freqtrade")
    persistence = types.ModuleType("freqtrade.persistence")
    strategy = types.ModuleType("freqtrade.strategy")

    class Trade: ...

    class IStrategy:
        def __init__(self, *a, **k): ...

    persistence.Trade = Trade
    strategy.IStrategy = IStrategy
    strategy.stoploss_from_absolute = _sl
    freqtrade.persistence = persistence
    freqtrade.strategy = strategy
    talib = types.ModuleType("talib")
    talib_abstract = types.ModuleType("talib.abstract")
    talib_abstract.EMA = lambda *a, **k: None
    talib.abstract = talib_abstract
    sys.modules.update(
        {
            "freqtrade": freqtrade,
            "freqtrade.persistence": persistence,
            "freqtrade.strategy": strategy,
            "talib": talib,
            "talib.abstract": talib_abstract,
        }
    )


class _FakeFillsReader:
    """Injectable FT fills source: empty open-trade list, zero network."""

    def fetch_open_trades(self):
        return []


def _closed_boundary_ms() -> int:
    """open_ms of the most recent fully-closed 5m candle (real time)."""
    now_ms = int(time.time() * 1000)
    return (now_ms // 300_000 - 1) * 300_000


@pytest.fixture(scope="module")
def strat_mod():
    _install_ft_stubs()
    spec = importlib.util.spec_from_file_location("aigrid_ft_strategy_r14", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hermetic_closed_candle_to_strategy_stake(strat_mod, tmp_path, monkeypatch):
    """FULL card chain on the real objects:

    closed candles -> orchestrator.process_candle (regime controller) ->
    deterministic plan -> guarded atomic publish -> ft_integration.json ->
    REAL AIGridStrategy reads the file and returns a level_stake entry.

    No LLM anywhere on the path; the only "network" (FT REST facts) is an
    injected dict-provider — the controller needs its 3-candle
    confirmation + cooldown, so the pump replays the last candles.
    """
    settings = _settings(tmp_path)
    orch = AIGridOrchestrator(
        settings,
        exchange_info=ExchangeInfo(EI_PATH),
        ft_fills_reader=_FakeFillsReader(),
    )
    orch.preflight_ok = True  # service would have verified preflight
    write_secret(settings.state_dir)
    publisher = PlanPublisher(
        orch.store,
        orch.eventlog,
        validator=orch.validator,
        ownership=orch.ownership,
        settings=settings,
        throttle_s=0.0,
        price_tick_of=orch._price_tick,
        clock=time.time,
    )
    end_ms = _closed_boundary_ms()
    df = _df(n=600, end_ms=end_ms)
    cyc = DeterministicCycle(
        orch,
        publisher,
        candle_loader=lambda pair: df,
        facts_provider=lambda: _facts(server_time_ms=int(time.time() * 1000)),
        whitelist=(PAIR,),
    )
    assert cyc.startup_reconcile()["status"] in ("FRESH_START", "RESTORED")
    # The REAL RegimeController starts paused/HIGH_VOLATILITY and confirms a
    # regime only after confirm(3) + cooldown(6) agreeing candles — fast-
    # forward that hysteresis exactly as a warm service would have it.
    ctrl = orch._controller_for(PAIR)
    ctrl.state.confirmed = "RANGE"
    ctrl.state.paused = False
    ctrl.state.pause_reason = ""
    # 5 new candles: too few for a hostile classify to flip the confirmed
    # regime (cooldown needs 6), so the publish still proves the full chain.
    cyc._seen[PAIR] = int(df.index[-6].value // 1_000_000)
    outcomes = cyc.tick()
    assert outcomes, "cycle produced no publish attempt"
    assert outcomes[0].status == "PUBLISHED", outcomes[0].reasons

    # ---------- the published file is the strategy's only input ----------
    integ_path = tmp_path / "state" / "ft_integration.json"
    payload = json.loads(integ_path.read_text(encoding="utf-8"))
    plan = payload["plans"][PAIR]
    assert payload["enabled"] is True
    assert payload["dry_run_verified"] is True
    assert plan["grid_levels_prices"] == sorted(plan["grid_levels_prices"], reverse=True)
    assert plan["level_stake"] > 0
    assert plan["stop_loss"] < plan["grid_lower"]
    assert payload["filled_level_prices"][PAIR] == []
    events = [e["kind"] for e in orch.eventlog.recent()]
    assert "PLAN_PUBLISHED" in events

    # ---------- REAL strategy consumes the REAL file ----------------------
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", integ_path)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", tmp_path / "strategy_sl_cache.json")
    s = object.__new__(strat_mod.AIGridStrategy)
    s._aigrid = {}
    s._plan = {}
    s._levels = []
    s._sl_cache = {}
    s._last_sl_pct = None
    s._last_closed = {}
    s._plan_expires_ms = None
    s.config = make_config()
    write_integration(integ_path.parent, s.config, **payload)
    s.bot_start()

    levels = sorted(float(x) for x in plan["grid_levels_prices"])
    top = levels[-1]
    prev = top + 0.05  # above the grid's highest level
    s.populate_indicators(
        pd.DataFrame(
            [(prev, prev, prev, prev), (prev, prev, top - 0.05, top - 0.02)],
            columns=["open", "high", "low", "close"],
        ),
        {"pair": PAIR},
    )
    trade = types.SimpleNamespace(
        pair=PAIR,
        is_short=False,
        open_rate=top,
        amount=0.0,
        stake_amount=0.0,
        leverage=1.0,
        has_open_orders=False,
    )
    out = s.adjust_trade_position(
        trade,
        None,
        top - 0.02,
        -0.01,
        10.0,
        1000.0,
        top - 0.02,
        top - 0.02,
        -0.01,
        -0.01,
    )
    assert out is not None, (
        "strategy refused the published plan "
        f"(plan={plan['plan_id']} levels={plan['grid_levels_prices']})"
    )
    stake, tag = out
    assert stake == pytest.approx(plan["level_stake"])
    assert str(tag).startswith("grid_level_")
    assert s.custom_stake_amount(
        PAIR,
        None,
        top - 0.02,
        plan["level_stake"],
        10.0,
        1000.0,
        1.0,
        None,
        "long",
    ) == pytest.approx(plan["level_stake"])
    orch.eventlog.close()  # NOT orch.close(): persist() clobbers
