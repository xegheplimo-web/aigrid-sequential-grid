"""AIGrid — TradingAgents + Adaptive AI Grid + Freqtrade integration layer.

Safety invariants (see ARCHITECTURE.md):
- LLM never places orders directly; it only produces validated AIGridPlan objects.
- Independent Risk Engine can veto everything.
- All features default to OFF (`AIGridSettings.enabled=False`).
- Local dry-run / shadow / backtest only.
"""

from __future__ import annotations

__version__ = "0.1.0"
