"""Regime Controller — 3-level response: PAUSE / CONFIRM / PROTECT.

LEVEL 1 PAUSE   : stop opening new positions; cancel unfit pending entries;
                  keep all risk protections; trigger reanalysis.
LEVEL 2 CONFIRM : regime switches only after N consecutive agreeing candles
                  (hysteresis) and a cooldown between switches — prevents
                  mode flapping when the LLM changes its mind.
LEVEL 3 PROTECT : owned by the independent Risk Engine (see risk/engine.py);
                  this controller never force-closes positions by itself.

The controller NEVER closes existing positions on a regime change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import DOWNTREND, HIGH_VOLATILITY, RANGE, REGIMES, UPTREND


@dataclass
class ControllerState:
    confirmed: str = HIGH_VOLATILITY  # start fail-closed
    candidate: str | None = None
    candidate_count: int = 0
    candles_since_switch: int = 0
    paused: bool = True  # start paused (fail-closed)
    pause_reason: str = "INITIAL_SAFE_HOLD"
    transitions: list[tuple[int, str, str]] = field(default_factory=list)  # (candle_idx, from, to)


class RegimeController:
    """Hysteresis + cooldown wrapper around raw regime classifications."""

    def __init__(
        self,
        confirm_candles: int = 3,
        switch_cooldown_candles: int = 6,
        initial: str = HIGH_VOLATILITY,
    ):
        if not 1 <= confirm_candles <= 24:
            raise ValueError("confirm_candles out of bounds")
        if not 0 <= switch_cooldown_candles <= 500:
            raise ValueError("switch_cooldown_candles out of bounds")
        if initial not in REGIMES:
            raise ValueError(f"Unknown initial regime: {initial}")
        self.confirm_candles = confirm_candles
        self.switch_cooldown = switch_cooldown_candles
        self.state = ControllerState(
            confirmed=initial, paused=True, pause_reason="INITIAL_SAFE_HOLD"
        )

    # ------------------------------------------------------------------ update
    def update(self, raw: str, candle_index: int) -> str:
        """Feed one closed-candle regime classification; returns confirmed regime."""
        if raw not in REGIMES:
            raw = HIGH_VOLATILITY  # unknown classifications fail closed
        st = self.state
        st.candles_since_switch += 1

        if raw == st.confirmed:
            st.candidate, st.candidate_count = None, 0
        elif raw == st.candidate:
            st.candidate_count += 1
        else:
            st.candidate, st.candidate_count = raw, 1

        if (
            st.candidate is not None
            and st.candidate_count >= self.confirm_candles
            and st.candles_since_switch >= self.switch_cooldown
        ):
            previous = st.confirmed
            st.confirmed = st.candidate
            st.candidate, st.candidate_count = None, 0
            st.candles_since_switch = 0
            st.transitions.append((candle_index, previous, st.confirmed))
            self._on_confirmed(previous, st.confirmed, candle_index)
        return st.confirmed

    def _on_confirmed(self, previous: str, new: str, candle_index: int) -> None:
        if new == HIGH_VOLATILITY:
            self.pause("REGIME_HIGH_VOLATILITY", candle_index)
        elif previous == HIGH_VOLATILITY:
            self.resume(candle_index)
        else:
            # RANGE<->UPTREND<->DOWNTREND transitions: keep trading but the grid
            # mode must be re-planned; PAUSE until a fresh validated plan arrives.
            self.pause(f"REGIME_CHANGED_{previous}_TO_{new}", candle_index)

    # ------------------------------------------------------------------ pause/resume
    def pause(self, reason: str, candle_index: int = -1) -> None:
        self.state.paused = True
        self.state.pause_reason = reason

    def resume(self, candle_index: int = -1) -> None:
        self.state.paused = False
        self.state.pause_reason = ""

    # ------------------------------------------------------------------ queries
    @property
    def confirmed(self) -> str:
        return self.state.confirmed

    @property
    def paused(self) -> bool:
        return self.state.paused

    @property
    def pause_reason(self) -> str:
        return self.state.pause_reason

    def grid_mode_for(self, regime: str | None = None) -> str:
        """Regime -> grid mode mapping (SAFE_HOLD when paused/uncertain)."""
        regime = regime or self.state.confirmed
        if self.state.paused:
            return "SAFE_HOLD"
        return {
            RANGE: "RANGE_GRID",
            UPTREND: "LONG_GRID",
            DOWNTREND: "SHORT_GRID",
            HIGH_VOLATILITY: "SAFE_HOLD",
        }.get(regime, "SAFE_HOLD")

    def to_dict(self) -> dict:
        st = self.state
        return {
            "confirmed": st.confirmed,
            "paused": st.paused,
            "pause_reason": st.pause_reason,
            "candidate": st.candidate,
            "candidate_count": st.candidate_count,
            "candles_since_switch": st.candles_since_switch,
            "transitions": st.transitions[-20:],
        }
