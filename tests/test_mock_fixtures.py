from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mock_api.main import FIXTURE_SHIFT_DAYS, fixture_shift_days, shift_fixture_dates


def test_tests_and_evaluations_run_on_fixed_fixture_dates() -> None:
    assert FIXTURE_SHIFT_DAYS == 0
    raw = '{"vencimiento": "2026-09-10", "valid_until": "2026-09-13T23:59:00-03:00"}'
    assert shift_fixture_dates(raw, 0) == raw


def test_today_anchor_moves_dates_and_billing_periods_together() -> None:
    days = fixture_shift_days("today", datetime(2026, 10, 1, 12, 0, tzinfo=UTC))
    assert days == 20
    raw = (
        '{"periodo": "2026-09", "vencimiento": "2026-09-10", '
        '"valid_until": "2026-09-13T23:59:00-03:00", "cuenta": "CUST-00125", "monto": 61500.00}'
    )
    assert shift_fixture_dates(raw, days) == (
        '{"periodo": "2026-09", "vencimiento": "2026-09-30", '
        '"valid_until": "2026-10-03T23:59:00-03:00", "cuenta": "CUST-00125", "monto": 61500.00}'
    )
    assert '"periodo": "2026-10"' in shift_fixture_dates('{"periodo": "2026-09"}', 25)


def test_unknown_fixture_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="MOCK_FIXTURE_ANCHOR"):
        fixture_shift_days("tomorrow", datetime.now(UTC))
