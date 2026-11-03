"""R14 tests — guarded atomic publish (``aigrid/publish.py``).

Card R14 §2 (D-093 guards a–k) + addendum v2 (D-100) + addendum v4
(D-102). The publisher takes a deterministic plan + runtime facts and
either writes ``ft_integration.json`` atomically (tmp + os.replace) or
writes NOTHING and emits ``PUBLISH_BLOCKED`` with machine-readable
reasons. ``LIVE_REFUSED`` is special: not even a control file or
kill-switch artifact may be produced.

D-102 mandatory gates: (1) STARTUP_RECONCILIATION_PASS, (2) data
freshness >= 576 closed candles + regime PASS, (3) max_open_trades,
(4) CLOCK_SKEW < 5s, (5) pending orders not beyond TTL, (6)
restart-duplicate-check on (plan_id, pair). Plus D-100:
OPEN_PARTIAL_FILL_ACCOUNTED and NO_CONFLICTING_EXIT_ORDERS.

All tests hermetic: runtime facts are injected dicts; the only REST the
production path allows is read-only FT GETs — here even those are fakes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aigrid import preflight
from aigrid.exchange_info import ExchangeInfo
from aigrid.plans.schema import AIGridPlan, GridMode
from aigrid.plans.validator import PlanValidator
from aigrid.publish import PlanPublisher, RuntimeFacts
from aigrid.risk.ownership import OwnershipRegistry
from aigrid.settings import AIGridSettings, RiskLimits
from aigrid.state.store import StateStore
from tests._attest import facts, runtime_config, write_secret

ROOT = Path(__file__).resolve().parents[1]
EI_PATH = ROOT / ".orchestrator/fapi_exchangeinfo.json"
DATA_DIR = ROOT / "user_data/data/binance/futures"
PAIR = "ETH/USDT:USDT"
PAIR_B = "SOL/USDT:USDT"
NOW_MS = int(datetime(2026, 11, 3, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)


class _Events:
    def __init__(self):
        self.rows: list[tuple[str, str, dict]] = []

    def log(self, kind: str, symbol: str = "", payload: dict | None = None):
        self.rows.append((kind, symbol, payload or {}))

    def kinds(self) -> list[str]:
        return [k for k, _, _ in self.rows]

    def reasons(self) -> list[str]:
        out = []
        for kind, _, payload in self.rows:
            if kind == "PUBLISH_BLOCKED":
                out.extend(payload.get("reasons", []))
        return out


def _settings(**kw) -> AIGridSettings:
    base = {
        "enabled": True,
        "mode": "dryrun",
        "symbols": (PAIR,),
        "timeframe": "5m",
        "data_dir": DATA_DIR,
    }
    base.update(kw)
    return AIGridSettings(**base)


def _plan(pair: str = PAIR, **kw) -> AIGridPlan:
    # anchored to the publisher's injected NOW_MS clock — never wall clock
    now = datetime.fromtimestamp(NOW_MS / 1000.0, tz=UTC)
    args = {
        "plan_id": f"plan-{pair.split('/')[0].lower()}-r14",
        "plan_version": 1,
        "symbol": pair,
        "created_at": now - timedelta(seconds=60),
        "expires_at": now + timedelta(minutes=9),
        "market_regime": "RANGE",
        "confidence": 0.8,
        "grid_mode": GridMode.RANGE_GRID,
        "grid_lower": 85.0,
        "grid_upper": 95.0,
        "grid_levels": 3,
        "grid_spacing": 0.003,
        "max_capital": 100.0,
        "max_position_size": 100.0,
        "max_leverage": 1.0,
        "level_stake": 33.33,
        "level_notional": 33.33,
        "stop_loss": 82.45,
    }
    args.update(kw)
    return AIGridPlan(**args)  # descending levels [95.0, 90.0, 85.0]


def _facts(**kw) -> RuntimeFacts:
    args = {
        "dry_run": True,
        "trading_mode": "futures",
        "runmode": "dry_run",
        "pair_whitelist": (PAIR, PAIR_B),
        "max_open_trades": 3,
        "open_trades": [],
        "exchange_healthy": True,
        "server_time_ms": NOW_MS,
    }
    args.update(kw)
    args["runtime_config"] = runtime_config(
        dry_run=args["dry_run"],
        trading_mode=args["trading_mode"],
        max_open_trades=args["max_open_trades"],
        exchange={"pair_whitelist": list(args["pair_whitelist"] or ())},
    )
    args.setdefault("equity_usdt", None)
    return facts(**args)


def _publisher(
    tmp_path,
    *,
    events=None,
    ownership=None,
    validator="real",
    clock=None,
    throttle_s=0.0,
    settings=None,
) -> PlanPublisher:
    store = StateStore(tmp_path / "state")
    write_secret(store.state_dir)
    v = PlanValidator(ExchangeInfo(EI_PATH), RiskLimits()) if validator == "real" else validator
    return PlanPublisher(
        store,
        events if events is not None else _Events(),
        validator=v,
        ownership=ownership,
        settings=settings or _settings(),
        throttle_s=throttle_s,
        clock=clock or (lambda: NOW_MS / 1000.0),
    )


def _publish(pub: PlanPublisher, plan=None, **kw) -> object:
    args = {
        "facts": _facts(
            equity_usdt=1000,
        ),
        "candles_count": 600,
        "candle_close_ms": NOW_MS - 60_000,
        "last_close": 90.0,
        "startup_reconciled": True,
        "preflight_ok": True,
    }
    args.update(kw)
    return pub.publish(plan or _plan(), **args)


def _order(**kw) -> dict:
    order = {
        "order_id": "o1",
        "ft_is_entry": True,
        "ft_order_side": "buy",
        "ft_order_tag": "aigrid_x",
        "status": "open",
        "price": 90.0,
        "average": 90.0,
        "filled": 0.0,
        "amount": 1.0,
    }
    order.update(kw)
    return order


def _trade(orders, pair: str = PAIR, **kw) -> dict:
    t = {
        "trade_id": 1,
        "pair": pair,
        "strategy": "AIGridStrategy",
        "is_open": True,
        "orders": orders,
        "amount": 0.0,
        "current_rate": 90.0,
        "leverage": 1.0,
    }
    t.update(kw)
    return t


def _integration(store_dir: Path) -> dict:
    path = store_dir / "ft_integration.json"
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================== happy path
def test_publish_happy_path_writes_plan(tmp_path):
    ev = _Events()
    pub = _publisher(tmp_path, events=ev)
    plan = _plan()
    out = _publish(pub, plan)
    assert out.status == "PUBLISHED", out
    assert out.event_id
    payload = _integration(tmp_path / "state")
    block = dict(payload["attestation"])
    signature = block.pop("sig")
    assert block["config_hash"] == preflight.config_hash(
        _facts(
            equity_usdt=1000,
        ).runtime_config
    )
    assert hmac.compare_digest(
        signature,
        hmac.new(
            (tmp_path / "state" / ".attest_secret").read_bytes(),
            preflight.canonical_json(block),
            hashlib.sha256,
        ).hexdigest(),
    )
    assert payload["enabled"] is True
    assert payload["dry_run_verified"] is True
    assert payload["publisher"] == "deterministic"
    assert payload["plan_ttl_s"] == 600
    p = payload["plans"][PAIR]
    assert p["plan_id"] == plan.plan_id
    assert p["grid_levels_prices"] == plan.level_prices()
    assert p["grid_spacing"] == plan.grid_spacing
    assert p["stop_loss"] == plan.stop_loss
    assert p["level_stake"] == plan.level_stake
    assert p["max_capital"] == plan.max_capital
    assert p["max_position_size"] == plan.max_position_size
    assert p["max_leverage"] == 1.0
    assert p["expires_at"]
    assert payload["reconciliation_state"] == "VERIFIED"
    assert payload["filled_level_prices"][PAIR] == []
    kinds = ev.kinds()
    assert "PLAN_PUBLISHED" in kinds
    pub_e = next(p for k, _, p in ev.rows if k == "PLAN_PUBLISHED")
    assert pub_e["event_id"] == out.event_id
    # ledger persisted for restart-duplicate-check
    ledger = json.loads((tmp_path / "state" / "published_plans.json").read_text(encoding="utf-8"))
    assert f"{plan.plan_id}|{PAIR}" in ledger["seen"]


def test_publish_without_facts_is_blocked(tmp_path):
    """Tighten the gate: absent facts cannot prove dry-run or safe exposure."""
    pub = _publisher(tmp_path)
    integration_path = pub._store.state_dir / "ft_integration.json"
    out = _publish(pub, facts=None)
    assert out.status == "BLOCKED"
    assert "FACTS_UNAVAILABLE" in out.reasons
    assert not integration_path.exists()


# =============================================================== D-102 gates
def test_guard_startup_reconciliation_required(tmp_path):
    ev = _Events()
    pub = _publisher(tmp_path, events=ev)
    out = _publish(pub, startup_reconciled=False)
    assert out.status == "BLOCKED"
    assert "STARTUP_RECONCILIATION_REQUIRED" in out.reasons
    assert not (tmp_path / "state" / "ft_integration.json").exists()


def test_guard_insufficient_closed_candles(tmp_path):
    """D-102: fewer than 576 closed candles -> block (data_ambiente
    freshness gate)."""
    pub = _publisher(tmp_path)
    out = _publish(pub, candles_count=575)
    assert out.status == "BLOCKED"
    assert "INSUFFICIENT_CLOSED_CANDLES" in out.reasons


def test_guard_stale_market_data(tmp_path):
    """Last closed candle older than the staleness window -> DATA_STALE."""
    pub = _publisher(tmp_path)
    out = _publish(pub, candle_close_ms=NOW_MS - 31 * 60_000)
    assert out.status == "BLOCKED"
    assert "DATA_STALE" in out.reasons


def test_guard_clock_skew(tmp_path):
    """D-102: |server_time - local| must be < 5s."""
    ev = _Events()
    pub = _publisher(tmp_path, events=ev)
    out = _publish(pub, facts=_facts(equity_usdt=1000, server_time_ms=NOW_MS + 6_000))
    assert out.status == "BLOCKED"
    assert "CLOCK_SKEW" in out.reasons
    out_ok = _publish(pub, facts=_facts(equity_usdt=1000, server_time_ms=NOW_MS + 4_000))
    assert "CLOCK_SKEW" not in (out_ok.reasons or ())


def test_guard_max_open_trades(tmp_path):
    """D-102: publish blocked when the account is already at the cap."""
    trades = [_trade([], pair="XRP/USDT:USDT"), _trade([], pair="ADA/USDT:USDT")]
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, max_open_trades=2, open_trades=trades))
    assert out.status == "BLOCKED"
    assert "MAX_OPEN_TRADES_REACHED" in out.reasons
    out_ok = _publish(pub, facts=_facts(equity_usdt=1000, max_open_trades=-1, open_trades=trades))
    assert "MAX_OPEN_TRADES_REACHED" not in (out_ok.reasons or ())


def test_guard_restart_duplicate_plan(tmp_path):
    """D-102 (6): one (plan_id, pair) never publishes twice — even across a
    publisher restart, because the seen-set lives in state."""
    pub = _publisher(tmp_path)
    plan = _plan()
    assert _publish(pub, plan).status == "PUBLISHED"
    out = _publish(pub, plan)
    assert out.status == "BLOCKED" and "DUPLICATE_PLAN" in out.reasons
    # fresh publisher process on the same state dir -> still a duplicate
    pub2 = _publisher(tmp_path)
    out2 = _publish(pub2, plan)
    assert out2.status == "BLOCKED" and "DUPLICATE_PLAN" in out2.reasons


def test_guard_pending_order_beyond_ttl(tmp_path):
    """D-102 (5): a pending entry older than the plan TTL is stale ->
    block; a fresh pending entry blocks as PENDING_ENTRY_EXISTS."""
    stale = _order(order_id="st", order_date="2026-11-03T11:40:00+00:00")
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, open_trades=[_trade([stale])]))
    assert out.status == "BLOCKED"
    assert "PENDING_ORDER_BEYOND_TTL" in out.reasons

    fresh = _order(order_id="fr", order_date="2026-11-03T11:59:30+00:00")
    out2 = _publish(pub, facts=_facts(equity_usdt=1000, open_trades=[_trade([fresh])]))
    assert out2.status == "BLOCKED"
    assert "PENDING_ENTRY_EXISTS" in out2.reasons


# =============================================================== D-100 gates
def test_guard_open_partial_fill_accounted_and_mapped(tmp_path):
    """D-100 OPEN_PARTIAL_FILL_ACCOUNTED: a partial fill that maps to no
    plan level is a reconciliation failure -> entry/DCA locked for the
    pair (protective exits are untouched, only publishing is refused)."""
    bad = _order(
        order_id="u",
        status="open",
        filled=0.5,
        price=77.7,
        average=77.7,
        ft_order_tag=f"aigrid_{_plan().plan_id}",
    )
    pub = _publisher(tmp_path)
    plan = _plan()
    bad["ft_order_tag"] = f"aigrid_{plan.plan_id}"
    out = _publish(pub, plan, facts=_facts(equity_usdt=1000, open_trades=[_trade([bad])]))
    assert out.status == "BLOCKED"
    assert "PARTIAL_FILL_UNMAPPED" in out.reasons


def test_partial_fill_on_level_is_accounted_not_blocked(tmp_path):
    """A partial fill mapping to a level IS accounted (P0-5c) and appears
    in the published price-keyed fill book."""
    plan = _plan()
    part = _order(
        order_id="p",
        status="open",
        filled=0.5,
        price=90.0,
        average=90.0,
        ft_order_tag=f"aigrid_{plan.plan_id}",
    )
    pub = _publisher(tmp_path)
    # has_open trade on the pair -> PAIR_HAS_OPEN_TRADE is also raised,
    # but the fill itself must NOT be flagged unmapped.
    out = _publish(pub, plan, facts=_facts(equity_usdt=1000, open_trades=[_trade([part])]))
    assert out.status == "BLOCKED"
    assert "PARTIAL_FILL_UNMAPPED" not in out.reasons
    assert "PAIR_HAS_OPEN_TRADE" in out.reasons


def test_guard_conflicting_exit_orders(tmp_path):
    """D-100 NO_CONFLICTING_EXIT_ORDERS: an open exit order on the pair
    blocks a new plan."""
    ex = _order(order_id="e", ft_is_entry=False, ft_order_side="sell", status="open", filled=0.0)
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, open_trades=[_trade([ex])]))
    assert out.status == "BLOCKED"
    assert "CONFLICTING_EXIT_ORDERS" in out.reasons


# =============================================================== card §2 guards a-k
def test_guard_dry_run_false_live_refused(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, dry_run=False))
    assert out.status == "BLOCKED"
    assert "LIVE_REFUSED" in out.reasons


def test_guard_trading_mode_live_refused(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, trading_mode="live"))
    assert out.status == "BLOCKED"
    assert "LIVE_REFUSED" in out.reasons


def test_guard_runmode_live_refused(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, runmode="live"))
    assert out.status == "BLOCKED"
    assert "LIVE_REFUSED" in out.reasons


def test_guard_pair_not_in_whitelist(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, pair_whitelist=("SOL/USDT:USDT",)))
    assert out.status == "BLOCKED"
    assert "PAIR_NOT_IN_WHITELIST" in out.reasons


def test_guard_validator_rejects(tmp_path):
    """Validator verdict fail -> blocked; reasons surfaced (fail-closed)."""
    pub = _publisher(tmp_path)
    out = _publish(pub, last_close=3000.0)  # grid far from market
    assert out.status == "BLOCKED"
    assert any(r.startswith("PLAN_REJECTED") for r in out.reasons)


def test_guard_validator_unavailable(tmp_path):
    pub = _publisher(tmp_path, validator=None)
    out = _publish(pub)
    assert out.status == "BLOCKED"
    assert "VALIDATOR_UNAVAILABLE" in out.reasons


def test_guard_active_plan_exists(tmp_path):
    """One active plan per pair (crowded-plan skip)."""
    pub = _publisher(tmp_path)
    assert _publish(pub, _plan()).status == "PUBLISHED"
    out = _publish(pub, _plan(plan_id="plan-new-different"))
    assert out.status == "BLOCKED"
    assert "ACTIVE_PLAN_EXISTS" in out.reasons


def test_guard_plan_expired(tmp_path):
    now = datetime.fromtimestamp(NOW_MS / 1000.0, tz=UTC)
    plan = _plan(
        created_at=now - timedelta(minutes=30),
        expires_at=now - timedelta(minutes=20),
    )
    pub = _publisher(tmp_path)
    out = _publish(pub, plan)
    assert out.status == "BLOCKED"
    assert "PLAN_EXPIRED" in out.reasons


def test_guard_pending_entry(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, open_trades=[_trade([_order()])]))
    assert out.status == "BLOCKED"
    assert "PENDING_ENTRY_EXISTS" in out.reasons


def test_guard_pair_locked_or_owned(tmp_path):
    """Ownership lease on the pair (grid:, trend:, risk-engine, or a live
    AIGridStrategy trade) blocks publish."""
    reg = OwnershipRegistry()
    reg.acquire(PAIR, "trend", "trend:other-bot")
    pub = _publisher(tmp_path, ownership=reg)
    out = _publish(pub)
    assert out.status == "BLOCKED"
    assert "PAIR_LOCKED_OR_OWNED" in out.reasons

    # our own strategy holding an open trade on the pair = owned
    reg2 = OwnershipRegistry()
    pub2 = _publisher(tmp_path / "b", ownership=reg2)
    own_trade = _trade([], strategy="AIGridStrategy")
    out2 = _publish(pub2, facts=_facts(equity_usdt=1000, open_trades=[own_trade]))
    assert out2.status == "BLOCKED"
    assert "PAIR_HAS_OPEN_TRADE" in out2.reasons


def test_guard_exchange_unhealthy(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, exchange_healthy=False))
    assert out.status == "BLOCKED"
    assert "EXCHANGE_UNHEALTHY" in out.reasons


def test_guard_preflight_not_ok(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, preflight_ok=False)
    assert out.status == "BLOCKED"
    assert "PREFLIGHT_NOT_OK" in out.reasons


def test_guard_regime_not_range(tmp_path):
    """D-102: publish only on RANGE-regime PASS."""
    pub = _publisher(tmp_path)
    out = _publish(pub, _plan(market_regime="UPTREND"))
    assert out.status == "BLOCKED"
    assert "REGIME_NOT_RANGE" in out.reasons


def test_guard_stop_loss_blueprint_missing(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, _plan(stop_loss=None))
    assert out.status == "BLOCKED"
    assert "SL_BLUEPRINT_MISSING" in out.reasons


def test_blocked_publish_writes_no_integration_file(tmp_path):
    pub = _publisher(tmp_path)
    out = _publish(pub, facts=_facts(equity_usdt=1000, pair_whitelist=("SOL/USDT:USDT",)))
    assert out.status == "BLOCKED"
    assert not (tmp_path / "state" / "ft_integration.json").exists()
    assert not (tmp_path / "state" / "published_plans.json").exists()
    assert not list((tmp_path / "state").glob("*.tmp"))


# =============================================================== LIVE_REFUSED
def test_live_refused_writes_nothing(tmp_path):
    """LIVE_REFUSED: no control file at all — no ft_integration.json, no
    published-plan ledger, no tmp, no kill-switch artifact. Only the
    PUBLISH_BLOCKED event (SQLite log) may be recorded."""
    ev = _Events()
    pub = _publisher(tmp_path, events=ev)
    out = _publish(pub, facts=_facts(equity_usdt=1000, dry_run=False, runmode="live"))
    assert out.status == "BLOCKED"
    assert "LIVE_REFUSED" in out.reasons
    state = tmp_path / "state"
    assert list(state.glob("*.json")) == []
    assert list(state.glob("*.tmp")) == []
    assert list(state.glob("*kill*")) == []
    assert "PUBLISH_BLOCKED" in ev.kinds()


# =============================================================== atomic write
def test_atomic_write_temp_then_replace(tmp_path, monkeypatch):
    """The target appears only via os.replace of a fully-written tmp file
    in the same directory (POSIX-safe atomic publish)."""
    seen: list[Path] = []
    real_replace = os.replace

    def spy(src, dst):
        src, dst = Path(src), Path(dst)
        seen.append(src)
        assert src.exists() and src.suffix == ".tmp"
        assert src.parent == dst.parent  # same volume -> atomic rename
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    pub = _publisher(tmp_path)
    out = _publish(pub)
    assert out.status == "PUBLISHED"
    assert seen, "os.replace never called"
    assert not list((tmp_path / "state").glob("*.tmp"))


def test_crash_between_temp_and_replace_leaves_old_file(tmp_path, monkeypatch):
    """A crash after the tmp write but before os.replace leaves the
    previous integration file untouched and the tmp cleaned up."""
    pub = _publisher(tmp_path)
    assert _publish(pub, _plan()).status == "PUBLISHED"
    before = _integration(tmp_path / "state")
    # expire the active plan so the second publish is not blocked
    (tmp_path / "state" / "published_plans.json").unlink()
    pub2 = _publisher(tmp_path / "state" / "x" / "y" / "z")
    pub2._store = pub._store  # same state dir, fresh publisher
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        _publish(pub2, _plan(plan_id="plan-after-crash"))
    assert _integration(tmp_path / "state") == before
    assert not list((tmp_path / "state").glob("*.tmp"))


# =============================================================== throttle
def test_publish_throttled_when_file_fresh(tmp_path):
    """File mtime < 240s -> skip rewriting (busy/throttle guard). mtime is
    filesystem-real, so this test runs the publisher on the real clock."""
    now = time.time()
    pub = _publisher(tmp_path, throttle_s=240.0, clock=time.time)
    now_ms = int(now * 1000)
    now_dt = datetime.fromtimestamp(now, tz=UTC)
    fresh = {
        "facts": _facts(equity_usdt=2000, server_time_ms=now_ms, open_trades=[]),
        "candle_close_ms": now_ms - 60_000,
    }

    def live_plan(pair=PAIR, **kw):
        return _plan(pair, created_at=now_dt, expires_at=now_dt + timedelta(minutes=9), **kw)

    assert _publish(pub, live_plan(), **fresh).status == "PUBLISHED"
    out = _publish(pub, live_plan(PAIR_B, plan_id="plan-sol-diff"), **fresh)
    assert out.status == "SKIPPED"
    assert "PUBLISH_THROTTLED" in out.reasons
    payload = _integration(tmp_path / "state")
    assert PAIR_B not in payload["plans"]

    # age the file past the window -> a fresh plan may publish again
    target = tmp_path / "state" / "ft_integration.json"
    old = now - 300.0
    os.utime(target, (old, old))
    out2 = _publish(pub, live_plan(PAIR_B, plan_id="plan-sol-later"), **fresh)
    assert out2.status == "PUBLISHED", out2


# =============================================================== multi-pair merge
def test_publish_merges_nonexpired_plans(tmp_path):
    pub = _publisher(tmp_path)
    assert _publish(pub, _plan()).status == "PUBLISHED"
    out = _publish(
        pub,
        _plan(pair=PAIR_B, plan_id="plan-sol-b"),
        facts=_facts(equity_usdt=2000, open_trades=[]),
    )
    assert out.status == "PUBLISHED"
    payload = _integration(tmp_path / "state")
    assert set(payload["plans"]) == {PAIR, PAIR_B}
