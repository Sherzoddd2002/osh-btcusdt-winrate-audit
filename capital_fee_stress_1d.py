"""BTCUSDT 1d fee/funding/slippage and $100-capital stress test.

Paper research only. Re-trains the 1d BASE model for each distinct
commission+slippage label-cost. The model for a given all-in cost is shared
by all fee/selection scenarios that have that same total friction.
Historical funding is downloaded from Binance's public USD-M Futures
/fapi/v1/fundingRate endpoint (or loaded from an exported CSV).

No exchange credentials, order endpoints, or live orders are used.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.parse
import urllib.request
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pipeline_v2 as pv
import research_v3 as rv
from selectivity_sweep_1d import thresholds_for_pct, wilson_interval_pct

BARS_PER_DAY = 96
HORIZON_BARS = 96
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um"
DEFAULT_FEES = {
    "maker_maker": ("maker", "maker"),
    "maker_taker": ("maker", "taker"),
    "taker_taker": ("taker", "taker"),
}
DEFAULT_SLIPPAGE_BPS = (0.0, 2.0, 5.0)


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_funding_history(
    raw: pd.DataFrame,
    symbol: str = "BTCUSDT",
    max_gap_hours: float = 8.0,
) -> pd.DataFrame:
    """Validate actual funding settlement rows; never synthesize rates."""
    if raw is None or raw.empty:
        raise ValueError("Funding history is empty")
    raw_attrs = dict(raw.attrs)
    cols = {str(c).lower(): c for c in raw.columns}
    time_col = (
        cols.get("fundingtime") or cols.get("funding_time")
        or cols.get("calc_time") or cols.get("calctime")
    )
    rate_col = (
        cols.get("fundingrate") or cols.get("funding_rate")
        or cols.get("last_funding_rate") or cols.get("lastfundingrate")
    )
    mark_col = cols.get("markprice") or cols.get("mark_price")
    interval_col = cols.get("funding_interval_hours")
    symbol_col = cols.get("symbol")
    mark_source_col = cols.get("mark_price_source")
    missing = [
        label for label, col in (
            ("fundingTime/calc_time", time_col), ("fundingRate/last_funding_rate", rate_col)
        ) if col is None
    ]
    if missing:
        raise ValueError(f"Funding CSV/API is missing required fields: {missing}")

    out = pd.DataFrame()
    if symbol_col is not None:
        symbols = raw[symbol_col].astype(str).str.upper().unique()
        if len(symbols) != 1 or symbols[0] != symbol.upper():
            raise ValueError(
                f"Funding symbol mismatch: expected only {symbol}, received {symbols.tolist()}"
            )
        out["symbol"] = raw[symbol_col].astype(str).str.upper()

    times = raw[time_col]
    if pd.api.types.is_datetime64_any_dtype(times):
        # Pandas datetime64 values are internally nanoseconds; never reinterpret
        # those integers as Unix milliseconds.
        out["funding_time"] = pd.to_datetime(times, utc=True, errors="coerce")
    elif pd.api.types.is_numeric_dtype(times):
        out["funding_time"] = pd.to_datetime(
            pd.to_numeric(times, errors="coerce"), unit="ms", utc=True, errors="coerce"
        )
    else:
        numeric = pd.to_numeric(times, errors="coerce")
        if numeric.notna().all():
            out["funding_time"] = pd.to_datetime(
                numeric, unit="ms", utc=True, errors="coerce"
            )
        else:
            out["funding_time"] = pd.to_datetime(times, utc=True, errors="coerce")

    out["funding_rate"] = pd.to_numeric(raw[rate_col], errors="coerce")
    if mark_col is None:
        out["mark_price"] = np.nan
        mark_price_source = "needs_15m_candle_proxy"
    else:
        out["mark_price"] = pd.to_numeric(raw[mark_col], errors="coerce")
        has_mark = out["mark_price"].notna()
        if has_mark.any() and not has_mark.all():
            raise ValueError("Funding markPrice column is only partially populated")
        mark_price_source = "binance_api" if has_mark.all() else "needs_15m_candle_proxy"
    if mark_source_col is not None:
        supplied_sources = raw[mark_source_col].dropna().astype(str).unique()
        if len(supplied_sources) > 1:
            raise ValueError("Funding CSV mixes multiple mark_price_source values")
        if len(supplied_sources) == 1:
            mark_price_source = supplied_sources[0]
    if interval_col is not None:
        out["funding_interval_hours"] = pd.to_numeric(raw[interval_col], errors="coerce")
        if out["funding_interval_hours"].isna().any() or (
            (out["funding_interval_hours"] <= 0)
            | (out["funding_interval_hours"] > 24)
        ).any():
            raise ValueError("funding_interval_hours contains invalid values")
    required_values = out[["funding_time", "funding_rate"]]
    if required_values.isna().any().any():
        counts = required_values.isna().sum()
        raise ValueError(f"Funding history contains invalid fields: {counts.to_dict()}")
    if not np.isfinite(out["funding_rate"].to_numpy(dtype=float)).all():
        raise ValueError("Funding rates must all be finite numeric values")
    if mark_price_source == "binance_api":
        if not np.isfinite(out["mark_price"].to_numpy(dtype=float)).all():
            raise ValueError("Funding mark prices must all be finite numeric values")
        if (out["mark_price"] <= 0).any():
            raise ValueError("Funding markPrice must be strictly positive")
    if out["funding_time"].duplicated().any():
        raise ValueError("Duplicate fundingTime rows detected")

    out = out.sort_values("funding_time").reset_index(drop=True)
    gaps = out["funding_time"].diff().dropna()
    max_gap = float(gaps.dt.total_seconds().max() / 3600.0) if len(gaps) else 0.0
    if max_gap > max_gap_hours + (1.0 / 60.0):
        raise ValueError(
            f"Funding history has a {max_gap:.2f}h gap; maximum accepted is "
            f"{max_gap_hours:.2f}h. Re-fetch or reconcile the funding data."
        )
    out.attrs.update(raw_attrs)
    out.attrs["max_gap_hours"] = max_gap
    out.attrs["symbol"] = symbol.upper()
    out.attrs["mark_price_source"] = mark_price_source
    return out


def fetch_binance_funding_history(
    symbol: str,
    start_ms: int,
    end_ms: int,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch all historical USD-M funding settlements with safe pagination."""
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("Funding fetch start/end timestamps are invalid")
    rows: list[dict] = []
    cursor = int(start_ms)
    pages = 0
    while cursor <= end_ms:
        pages += 1
        if pages > 100:
            raise RuntimeError("Funding API pagination exceeded 100 pages; aborting safely.")
        params = urllib.parse.urlencode({
            "symbol": symbol.upper(),
            "startTime": cursor,
            "endTime": int(end_ms),
            "limit": 1000,
        })
        request = urllib.request.Request(
            f"{FUNDING_URL}?{params}",
            headers={"User-Agent": "OSH-BTCUSDT-paper-research/1.0"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                page = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Binance public funding history could not be fetched. "
                "Check network/DNS access to fapi.binance.com, or supply "
                "--funding-csv with Binance funding history."
            ) from exc

        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected Binance funding API response: {page!r}")
        if not page:
            break
        rows.extend(page)
        page_times = [int(item["fundingTime"]) for item in page if "fundingTime" in item]
        if not page_times:
            raise RuntimeError("Binance funding response contains no fundingTime values")
        last_time = max(page_times)
        if last_time < cursor:
            raise RuntimeError("Binance funding API pagination did not advance")
        if len(page) < 1000 or last_time >= end_ms:
            break
        cursor = last_time + 1
        time.sleep(0.10)

    if not rows:
        raise RuntimeError("Binance returned no funding history for the requested period")
    raw = pd.DataFrame(rows).drop_duplicates(subset=["fundingTime"], keep="last")
    return validate_funding_history(raw, symbol=symbol)



def _download_vision_bytes(url: str, timeout: int = 45) -> bytes | None:
    """Download one public Binance Vision object; return None only for HTTP 404."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OSH-BTCUSDT-paper-research/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"Binance Vision request failed ({exc.code}): {url}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Binance Vision archive could not be reached. Check DNS/network access "
            "to data.binance.vision."
        ) from exc


def _read_vision_funding_zip(url: str, timeout: int = 45) -> pd.DataFrame | None:
    """Read and structurally validate one monthly/daily Binance Vision funding ZIP."""
    payload = _download_vision_bytes(url, timeout=timeout)
    if payload is None:
        return None

    checksum_payload = _download_vision_bytes(url + ".CHECKSUM", timeout=timeout)
    checksum_verified = False
    if checksum_payload:
        try:
            expected = checksum_payload.decode("ascii", errors="strict").strip().split()[0].lower()
        except (UnicodeDecodeError, IndexError) as exc:
            raise ValueError(f"Invalid checksum file for Binance archive: {url}") from exc
        actual = hashlib.sha256(payload).hexdigest()
        if expected != actual:
            raise ValueError(
                f"SHA-256 mismatch for Binance Vision archive {url}: "
                f"expected {expected}, got {actual}"
            )
        checksum_verified = True

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Corrupt member {bad_member} inside {url}")
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(
                    f"Expected exactly one CSV in {url}, found {len(csv_names)}"
                )
            with archive.open(csv_names[0]) as stream:
                frame = pd.read_csv(stream)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Downloaded Binance Vision object is not a valid ZIP: {url}") from exc

    normalized = {str(col).strip().lower() for col in frame.columns}
    required = {"calc_time", "funding_interval_hours", "last_funding_rate"}
    if not required.issubset(normalized):
        raise ValueError(
            f"Unexpected funding archive schema in {url}; "
            f"expected {sorted(required)}, got {list(frame.columns)}"
        )
    frame.attrs["archive_url"] = url
    frame.attrs["checksum_verified"] = checksum_verified
    return frame


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    return list(pd.date_range(first, last, freq="MS", tz="UTC"))


def _vision_funding_urls(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp | None = None,
) -> list[tuple[str, str]]:
    """Plan completed monthly funding archives only (no daily fundingRate files)."""
    symbol = symbol.upper()
    today = today or pd.Timestamp.now(tz="UTC").normalize()
    current_month = pd.Timestamp(year=today.year, month=today.month, day=1, tz="UTC")
    plan: list[tuple[str, str]] = []
    for month_start in _month_range(start, end):
        # Binance Vision publishes fundingRate as monthly archives. The
        # incomplete current month is intentionally not treated as a daily
        # archive family; its settlements must come from REST or a user CSV.
        if month_start >= current_month:
            continue
        period = month_start.strftime("%Y-%m")
        url = (
            f"{BINANCE_VISION_BASE}/monthly/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{period}.zip"
        )
        plan.append((f"monthly {period}", url))
    return plan


def vision_monthly_coverage(
    end_ms: int,
    today: pd.Timestamp | None = None,
) -> tuple[bool, pd.Timestamp]:
    """Return whether monthly-only Vision coverage needs a current-month truncation.

    The returned timestamp is the latest 15m candle open that can be kept
    when the current UTC month has no completed monthly funding archive.
    """
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    today = today or pd.Timestamp.now(tz="UTC").normalize()
    current_month = pd.Timestamp(year=today.year, month=today.month, day=1, tz="UTC")
    if end >= current_month:
        return True, current_month - pd.Timedelta(minutes=rv.TF_MINUTES)
    return False, end


def fetch_binance_vision_funding_history(
    symbol: str,
    start_ms: int,
    end_ms: int,
    timeout: int = 45,
    workers: int = 4,
) -> pd.DataFrame:
    """Download completed monthly USD-M funding archives when REST is unreachable.

    Binance Vision funding CSVs carry calc_time, funding_interval_hours, and
    last_funding_rate, but not the API's settlement markPrice. Mark-price proxy
    attachment is therefore a separate, explicit step using only completed
    BTCUSDT 15m candles. The incomplete current month is not present in the
    monthly archive and must be fetched from REST or supplied separately.
    """
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("Funding archive start/end timestamps are invalid")
    if workers < 1 or workers > 8:
        raise ValueError("Funding archive workers must be between 1 and 8")
    start = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    plan = _vision_funding_urls(symbol, start, end)
    if not plan:
        raise ValueError("No Binance Vision funding archives were planned for this date range")
    print(
        f"[funding] Binance Vision fallback: {len(plan)} completed monthly archive file(s), "
        f"up to {workers} concurrent downloads."
    )

    downloaded: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_read_vision_funding_zip, url, timeout): label
            for label, url in plan
        }
        completed = 0
        for future in as_completed(futures):
            label = futures[future]
            frame = future.result()
            completed += 1
            if frame is None:
                missing.append(label)
            else:
                downloaded[label] = frame
            if completed % 12 == 0 or completed == len(plan):
                print(f"[funding] Archive files checked: {completed}/{len(plan)}")

    if missing:
        sample = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise RuntimeError(
            f"Binance Vision funding archive is incomplete: {len(missing)} missing file(s): "
            f"{sample}{suffix}. Refusing to fill missing funding events."
        )

    frames = [downloaded[label] for label, _ in plan]
    raw = pd.concat(frames, ignore_index=True)
    raw["symbol"] = symbol.upper()
    raw = raw.rename(columns={
        "calc_time": "fundingTime",
        "last_funding_rate": "fundingRate",
    })
    # Monthly ZIPs contain the entire month. Keep only the requested time
    # window; do not let later rows from the final ZIP leak beyond the sample.
    raw_time_ms = pd.to_numeric(raw["fundingTime"], errors="coerce")
    needs_truncation, coverage_end = vision_monthly_coverage(end_ms)
    archive_end_ms = (
        int((coverage_end + pd.Timedelta(minutes=rv.TF_MINUTES)).timestamp() * 1000) - 1
        if needs_truncation else int(end_ms)
    )
    raw = raw.loc[
        raw_time_ms.between(int(start_ms), archive_end_ms)
    ].copy()
    raw = raw.drop_duplicates(subset=["fundingTime"], keep="last")
    result = validate_funding_history(raw, symbol=symbol)
    result.attrs["funding_source"] = "binance_vision_archive"
    result.attrs["mark_price_source"] = "needs_15m_candle_proxy"
    result.attrs["archive_files"] = int(len(plan))
    result.attrs["archive_checksum_verified_files"] = int(sum(
        bool(downloaded[label].attrs.get("checksum_verified"))
        for label, _ in plan
    ))
    result.attrs["archive_truncation_required"] = bool(needs_truncation)
    result.attrs["archive_coverage_end_utc"] = coverage_end.isoformat()
    return result


def attach_funding_mark_price_proxy(
    funding: pd.DataFrame,
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """Fill absent archive mark prices from the last completed BTC 15m candle close.

    This is an explicit proxy, not Binance's true markPrice. It is causal:
    the 15m close used for each funding event is available at or before the
    fundingTime. For the first event only, if no prior close exists, use the
    open of a candle exactly at that event time.
    """
    out = funding.copy()
    if "mark_price" not in out.columns:
        out["mark_price"] = np.nan
    missing = out["mark_price"].isna()
    if not missing.any():
        out.attrs.update(funding.attrs)
        out.attrs["mark_price_source"] = out.attrs.get(
            "mark_price_source", "binance_api"
        )
        return out
    if not missing.all():
        raise ValueError("Funding mark price is partially missing; refusing mixed proxy/API rows")

    candle_times = pd.to_datetime(candles["candle_time"], utc=True)
    completed = pd.DataFrame({
        "available_at": candle_times + pd.Timedelta(minutes=rv.TF_MINUTES),
        "proxy_price": pd.to_numeric(candles["close"], errors="coerce").to_numpy(float),
    }).dropna().sort_values("available_at")
    events = out[["funding_time"]].copy()
    events["_row"] = np.arange(len(events))
    events = events.sort_values("funding_time")
    merged = pd.merge_asof(
        events,
        completed,
        left_on="funding_time",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    # merge_asof may expose a read-only NumPy view; the first-event
    # candle-open fallback below needs to fill missing values in place.
    proxy = merged.sort_values("_row")["proxy_price"].to_numpy(dtype=float, copy=True)

    # First archive event can coincide with the very first available 15m candle.
    still_missing = ~np.isfinite(proxy)
    if still_missing.any():
        candle_open = pd.DataFrame({
            "candle_time": candle_times,
            "open_proxy": pd.to_numeric(candles["open"], errors="coerce").to_numpy(float),
        }).set_index("candle_time")["open_proxy"]
        event_times = pd.DatetimeIndex(out["funding_time"])
        exact = candle_open.reindex(event_times).to_numpy(float)
        proxy[still_missing] = exact[still_missing]

    if not np.isfinite(proxy).all() or np.any(proxy <= 0):
        raise ValueError(
            "Could not attach a valid causal 15m candle mark-price proxy to every funding event"
        )

    out["mark_price"] = proxy
    out.attrs.update(funding.attrs)
    out.attrs["mark_price_source"] = "previous_completed_15m_close_proxy"
    return out


def load_funding_history(
    csv_path: str | Path,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Funding CSV not found: {path}. Pass --fetch-funding or provide --funding-csv."
        )
    return validate_funding_history(pd.read_csv(path), symbol=symbol)


def check_funding_coverage(
    funding: pd.DataFrame,
    candles: pd.DataFrame,
    max_tail_lag_hours: float = 8.25,
) -> dict:
    """Fail closed if funding data does not span the backtest candle range."""
    first_candle = pd.Timestamp(candles["candle_time"].iloc[0])
    last_candle = pd.Timestamp(candles["candle_time"].iloc[-1])
    first_funding = pd.Timestamp(funding["funding_time"].iloc[0])
    last_funding = pd.Timestamp(funding["funding_time"].iloc[-1])
    start_lag = (first_funding - first_candle).total_seconds() / 3600.0
    tail_lag = (last_candle - last_funding).total_seconds() / 3600.0
    if start_lag > max_tail_lag_hours:
        raise ValueError(
            f"Funding history starts {start_lag:.2f}h after the BTC data begins; "
            "the funding series is not complete at the start."
        )
    if tail_lag > max_tail_lag_hours:
        raise ValueError(
            f"Funding history ends {tail_lag:.2f}h before the BTC data ends; "
            "the funding series is stale/incomplete at the end."
        )
    return {
        "rows": int(len(funding)),
        "first_funding_time_utc": first_funding.isoformat(),
        "last_funding_time_utc": last_funding.isoformat(),
        "max_interval_hours": float(funding.attrs.get("max_gap_hours", 0.0)),
        "max_tail_lag_hours": float(tail_lag),
        "coverage_verdict": "PASS",
    }


def trade_funding_cost(
    side: int,
    entry_price: float,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    exit_bar_open: pd.Timestamp,
    funding_times_ns: np.ndarray,
    funding_rates: np.ndarray,
    funding_marks: np.ndarray,
) -> tuple[float, int, int]:
    """Funding cost in bps of entry notional; ambiguous edge events are adverse-only.

    A 15m candle cannot identify the exact second of a barrier exit. Funding
    settlements inside the entry/exit boundary candles are therefore treated
    conservatively: payments are counted, potential receipts are not.
    """
    if side not in (-1, 1) or entry_price <= 0:
        raise ValueError("Invalid side or entry price for funding calculation")
    lo = int(np.searchsorted(funding_times_ns, entry_time.value, side="left"))
    hi = int(np.searchsorted(funding_times_ns, exit_time.value, side="right"))
    if hi <= lo:
        return 0.0, 0, 0

    times = funding_times_ns[lo:hi]
    rates = funding_rates[lo:hi]
    marks = funding_marks[lo:hi]
    signed_cost_bps = side * rates * (marks / entry_price) * 10_000.0

    entry_bar_end = entry_time + pd.Timedelta(minutes=rv.TF_MINUTES)
    ambiguous = (times <= entry_bar_end.value) | (times >= exit_bar_open.value)
    applied = np.where(ambiguous, np.maximum(signed_cost_bps, 0.0), signed_cost_bps)
    charged_or_received = int(np.count_nonzero(applied))
    return float(applied.sum()), int(len(times)), int(np.count_nonzero(ambiguous))


def thresholds_for_selection(
    p_cal: np.ndarray,
    pct: float,
) -> tuple[float, float]:
    return thresholds_for_pct(p_cal, pct)


def build_trade_records(
    selected: np.ndarray,
    sides: np.ndarray,
    local_offset: int,
    original_indices: np.ndarray,
    labels: pd.DataFrame,
    candles: pd.DataFrame,
    funding_times_ns: np.ndarray,
    funding_rates: np.ndarray,
    funding_marks: np.ndarray,
    fold_number: int | None,
) -> list[dict]:
    records: list[dict] = []
    raw_exit = labels["exit_ret"].to_numpy(float)
    hold_bars = labels["holding_bars"].to_numpy(float)
    path_high = labels["path_high"].to_numpy(float)
    path_low = labels["path_low"].to_numpy(float)
    entry_prices = labels["entry_price"].to_numpy(float)
    candle_times = pd.to_datetime(candles["candle_time"], utc=True)

    for local_idx, side in zip(np.asarray(selected, dtype=int), np.asarray(sides, dtype=int)):
        filtered_idx = int(local_offset + local_idx)
        original_idx = int(original_indices[filtered_idx])
        held = int(hold_bars[original_idx])
        entry_idx = original_idx + 1
        exit_idx = original_idx + held
        if entry_idx >= len(candles) or exit_idx >= len(candles):
            raise ValueError("Selected trade has an invalid entry/exit candle index")

        entry_price = float(entry_prices[original_idx])
        underlying_return = float(raw_exit[original_idx])
        exit_price = entry_price * (1.0 + underlying_return)
        entry_time = pd.Timestamp(candle_times.iloc[entry_idx])
        exit_bar_open = pd.Timestamp(candle_times.iloc[exit_idx])
        exit_time = exit_bar_open + pd.Timedelta(minutes=rv.TF_MINUTES)
        funding_bps, funding_event_count, ambiguous_count = trade_funding_cost(
            int(side), entry_price, entry_time, exit_time, exit_bar_open,
            funding_times_ns, funding_rates, funding_marks,
        )

        high_seen = float(path_high[original_idx])
        low_seen = float(path_low[original_idx])
        if side == 1:
            mae = low_seen / entry_price - 1.0
            mfe = high_seen / entry_price - 1.0
            direction = "LONG"
        else:
            mae = entry_price / high_seen - 1.0
            mfe = entry_price / low_seen - 1.0
            direction = "SHORT"

        records.append({
            "fold": fold_number,
            "entry_time_utc": entry_time.isoformat(),
            "exit_time_est_utc": exit_time.isoformat(),
            "side": direction,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "underlying_exit_return": underlying_return,
            "directional_gross_return": float(side * underlying_return),
            "holding_bars": held,
            "holding_hours_est": float(held * rv.TF_MINUTES / 60.0),
            "mae_return": float(mae),
            "mfe_return": float(mfe),
            "funding_cost_bps": funding_bps,
            "funding_events_in_hold_window": funding_event_count,
            "funding_boundary_ambiguous_events": ambiguous_count,
        })
    return records


def add_execution_costs(
    trade: dict,
    entry_fee_bps: float,
    exit_fee_bps: float,
    slippage_bps: float,
    bnb_discount_pct: float,
) -> dict:
    discount = 1.0 - bnb_discount_pct / 100.0
    exit_price_ratio = float(trade["exit_price"]) / float(trade["entry_price"])
    commission_bps = (entry_fee_bps + exit_fee_bps * exit_price_ratio) * discount
    net_bps = (
        float(trade["directional_gross_return"]) * 10_000.0
        - commission_bps
        - slippage_bps
        - float(trade["funding_cost_bps"])
    )
    out = dict(trade)
    out.update({
        "entry_fee_bps": entry_fee_bps * discount,
        "exit_fee_bps_at_exit_notional": exit_fee_bps * exit_price_ratio * discount,
        "commission_cost_bps": float(commission_bps),
        "slippage_cost_bps": float(slippage_bps),
        "net_return_bps": float(net_bps),
        "net_return": float(net_bps / 10_000.0),
    })
    return out


def summarize_net_trades(trades: list[dict]) -> dict:
    n = len(trades)
    if not n:
        return {
            "trades": 0,
            "win_rate_pct": None,
            "win_rate_wilson95_pct": [None, None],
            "expectancy_bps": None,
            "profit_factor": None,
            "avg_net_win_bps": None,
            "avg_net_loss_bps": None,
            "bootstrap_ci95_expectancy_bps": [None, None],
            "avg_commission_bps": None,
            "avg_slippage_bps": None,
            "avg_funding_cost_bps": None,
            "avg_holding_hours": None,
            "avg_mae_bps": None,
            "worst_mae_bps": None,
        }

    net = np.asarray([t["net_return"] for t in trades], dtype=float)
    net_bps = net * 10_000.0
    winners = net > 0
    losers = net <= 0
    gross_wins = float(net[winners].sum())
    gross_losses = float(-net[losers].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else None
    ci_lo, ci_hi = rv.block_bootstrap_ci(net, block=20, reps=2000, seed=42)
    wins = int(winners.sum())
    mae_values = np.asarray([t["mae_return"] for t in trades], dtype=float)

    return {
        "trades": int(n),
        "wins": wins,
        "losses_or_breakeven": int(n - wins),
        "win_rate_pct": float(winners.mean() * 100.0),
        "win_rate_wilson95_pct": wilson_interval_pct(wins, n),
        "expectancy_bps": float(net_bps.mean()),
        "profit_factor": float(pf) if pf is not None else None,
        "avg_net_win_bps": float(net_bps[winners].mean()) if winners.any() else None,
        "avg_net_loss_bps": float(-net_bps[losers].mean()) if losers.any() else None,
        "bootstrap_ci95_expectancy_bps": [
            float(ci_lo) if ci_lo is not None else None,
            float(ci_hi) if ci_hi is not None else None,
        ],
        "avg_commission_bps": float(np.mean([t["commission_cost_bps"] for t in trades])),
        "avg_slippage_bps": float(np.mean([t["slippage_cost_bps"] for t in trades])),
        "avg_funding_cost_bps": float(np.mean([t["funding_cost_bps"] for t in trades])),
        "funding_settlement_rows_in_position_windows": int(sum(
            t["funding_events_in_hold_window"] for t in trades
        )),
        "funding_boundary_ambiguous_events": int(sum(
            t["funding_boundary_ambiguous_events"] for t in trades
        )),
        "avg_holding_hours": float(np.mean([t["holding_hours_est"] for t in trades])),
        "avg_mae_bps": float(mae_values.mean() * 10_000.0),
        "worst_mae_bps": float(mae_values.min() * 10_000.0),
    }


def simulate_capital(
    trades: list[dict],
    capital_usdt: float,
    position_cap_usdt: float,
    cash_reserve_pct: float = 10.0,
) -> dict:
    """Compounding account simulation with capped notional and a cash reserve.

    Notional is capped at the chosen $25/$50/$100 level and at the configured
    fraction of current equity. It can shrink after losses but does not grow
    above the chosen cap. This is not an exchange liquidation simulation.
    """
    if capital_usdt <= 0 or position_cap_usdt <= 0:
        raise ValueError("Capital and position cap must be positive")
    if not 0 <= cash_reserve_pct < 100:
        raise ValueError("cash_reserve_pct must be in [0, 100)")
    equity = float(capital_usdt)
    peak_equity = float(capital_usdt)
    max_closed_dd = 0.0
    max_intratrade_dd = 0.0
    gross_total = commission_total = slippage_total = 0.0
    funding_paid = funding_received = 0.0
    pnl_rows: list[float] = []
    notional_rows: list[float] = []
    loss_streak = max_loss_streak = 0
    skipped_after_ruin = 0

    for trade in trades:
        if equity <= 0:
            skipped_after_ruin += 1
            continue
        notional = min(position_cap_usdt, equity * (1.0 - cash_reserve_pct / 100.0))
        if notional <= 0:
            skipped_after_ruin += 1
            continue

        commission_bps = float(trade["commission_cost_bps"])
        slippage_bps = float(trade["slippage_cost_bps"])
        funding_bps = float(trade["funding_cost_bps"])
        gross_pnl = notional * float(trade["directional_gross_return"])
        commission = notional * commission_bps / 10_000.0
        slippage = notional * slippage_bps / 10_000.0
        funding_cash = notional * funding_bps / 10_000.0
        pnl = gross_pnl - commission - slippage - funding_cash

        entry_fee = notional * float(trade["entry_fee_bps"]) / 10_000.0
        # Worst path price excursion plus all positive funding charges is an
        # adverse mark-to-market stress. Funding credits are not used to improve it.
        trough_equity = (
            equity + notional * float(trade["mae_return"])
            - entry_fee - max(funding_cash, 0.0)
        )
        max_intratrade_dd = max(
            max_intratrade_dd,
            max(0.0, 1.0 - max(0.0, trough_equity) / max(peak_equity, 1e-12)),
        )

        equity += pnl
        peak_equity = max(peak_equity, equity)
        max_closed_dd = max(
            max_closed_dd,
            max(0.0, 1.0 - max(0.0, equity) / max(peak_equity, 1e-12)),
        )
        gross_total += gross_pnl
        commission_total += commission
        slippage_total += slippage
        funding_paid += max(funding_cash, 0.0)
        funding_received += max(-funding_cash, 0.0)
        pnl_rows.append(float(pnl))
        notional_rows.append(float(notional))

        if pnl < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

        if equity <= 0:
            equity = 0.0

    pnl_array = np.asarray(pnl_rows, dtype=float)
    return {
        "initial_capital_usdt": float(capital_usdt),
        "position_cap_usdt": float(position_cap_usdt),
        "initial_notional_usdt": float(
            min(position_cap_usdt, capital_usdt * (1.0 - cash_reserve_pct / 100.0))
        ),
        "cash_reserve_pct": float(cash_reserve_pct),
        "executed_trades": int(len(pnl_rows)),
        "skipped_after_equity_depletion": int(skipped_after_ruin),
        "ending_balance_usdt": float(equity),
        "net_pnl_usdt": float(equity - capital_usdt),
        "return_pct": float((equity / capital_usdt - 1.0) * 100.0),
        "max_closed_trade_drawdown_pct": float(max_closed_dd * 100.0),
        "max_intratrade_mae_drawdown_pct": float(max_intratrade_dd * 100.0),
        "gross_pnl_usdt": float(gross_total),
        "commission_paid_usdt": float(commission_total),
        "slippage_cost_usdt": float(slippage_total),
        "funding_paid_usdt": float(funding_paid),
        "funding_received_usdt": float(funding_received),
        "net_funding_cost_usdt": float(funding_paid - funding_received),
        "mean_position_notional_usdt": (
            float(np.mean(notional_rows)) if notional_rows else None
        ),
        "worst_trade_pnl_usdt": (
            float(pnl_array.min()) if len(pnl_array) else None
        ),
        "max_consecutive_losses": int(max_loss_streak),
        "risk_note": (
            "Trade-close and candle-path MAE stress only; maintenance margin, "
            "liquidation engine, order-book depth and exact intrabar exit timing "
            "are not simulated."
        ),
    }


def _build_fee_scenarios(
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_grid: tuple[float, ...],
    bnb_discount_pct: float,
) -> list[dict]:
    maker = _number(maker_fee_bps, "maker_fee_bps")
    taker = _number(taker_fee_bps, "taker_fee_bps")
    discount = _number(bnb_discount_pct, "bnb_discount_pct")
    if maker < 0 or taker < 0 or not 0 <= discount <= 100:
        raise ValueError("Fee rates must be non-negative and BNB discount in [0, 100]")
    if not slippage_grid or any(not np.isfinite(x) or x < 0 for x in slippage_grid):
        raise ValueError("Slippage values must be non-negative")
    fee_sides = {
        "maker_maker": (maker, maker),
        "maker_taker": (maker, taker),
        "taker_taker": (taker, taker),
    }
    scenarios = []
    for fee_name, (entry_fee, exit_fee) in fee_sides.items():
        discounted_entry = entry_fee * (1.0 - discount / 100.0)
        discounted_exit = exit_fee * (1.0 - discount / 100.0)
        for slip in slippage_grid:
            total_for_labels = discounted_entry + discounted_exit + slip
            scenarios.append({
                "name": f"{fee_name}_slip_{slip:g}bps",
                "fee_mode": fee_name,
                "entry_fee_bps": entry_fee,
                "exit_fee_bps": exit_fee,
                "bnb_discount_pct": discount,
                "slippage_round_trip_bps": float(slip),
                "label_all_in_cost_bps": float(total_for_labels),
                "model_cost_key": f"{total_for_labels:.6f}",
            })
    return scenarios


def _scenario_trade_metrics(
    raw_trades: list[dict],
    scenario: dict,
    capital_sizes: tuple[float, ...],
    initial_capital: float,
    cash_reserve_pct: float,
) -> tuple[dict, list[dict]]:
    scored = [
        add_execution_costs(
            t,
            scenario["entry_fee_bps"],
            scenario["exit_fee_bps"],
            scenario["slippage_round_trip_bps"],
            scenario["bnb_discount_pct"],
        )
        for t in raw_trades
    ]
    stats = summarize_net_trades(scored)
    portfolio = {
        f"{size:g}": simulate_capital(
            scored, initial_capital, size, cash_reserve_pct=cash_reserve_pct
        )
        for size in capital_sizes
    }
    return {"metrics": stats, "capital_scenarios": portfolio}, scored


def run_capital_fee_stress(
    candles: pd.DataFrame,
    funding: pd.DataFrame,
    selection_pcts: tuple[float, ...] = (0.02, 0.05, 0.10),
    slippage_grid: tuple[float, ...] = DEFAULT_SLIPPAGE_BPS,
    maker_fee_bps: float = 2.0,
    taker_fee_bps: float = 5.0,
    bnb_discount_pct: float = 0.0,
    capital_sizes: tuple[float, ...] = (25.0, 50.0, 100.0),
    initial_capital: float = 100.0,
    cash_reserve_pct: float = 10.0,
    folds_count: int = 5,
    train_days: int = 365,
    test_days: int = 90,
    holdout_days: int = 180,
    source: str = "provided DataFrame",
) -> dict:
    pcts = tuple(float(x) for x in selection_pcts)
    if not pcts or len(set(pcts)) != len(pcts) or any(not 0 < x < 1 for x in pcts):
        raise ValueError("selection_pcts must be unique fractions strictly between 0 and 1")
    if min(folds_count, train_days, test_days, holdout_days) < 1:
        raise ValueError("fold count and all date windows must be positive")
    if initial_capital <= 0 or any(x <= 0 for x in capital_sizes):
        raise ValueError("Capital and position caps must be positive")
    if any(x > initial_capital for x in capital_sizes):
        raise ValueError("No position cap may exceed the initial account capital")
    if not 0 <= cash_reserve_pct < 100:
        raise ValueError("cash_reserve_pct must be in [0, 100)")

    candles = rv.validate_market_data(candles)
    funding = validate_funding_history(funding)
    funding = attach_funding_mark_price_proxy(funding, candles)
    funding_coverage = check_funding_coverage(funding, candles)
    funding_times_ns = np.fromiter(
        (pd.Timestamp(value).value for value in funding["funding_time"]),
        dtype=np.int64, count=len(funding),
    )
    funding_rates = funding["funding_rate"].to_numpy(float)
    funding_marks = funding["mark_price"].to_numpy(float)

    fee_scenarios = _build_fee_scenarios(
        maker_fee_bps, taker_fee_bps, tuple(slippage_grid), bnb_discount_pct
    )
    unique_costs = sorted({float(s["label_all_in_cost_bps"]) for s in fee_scenarios})
    X_all, _ = pv.make_features(candles)
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    holdout_rows = BARS_PER_DAY * holdout_days
    if len(candles) <= holdout_rows:
        raise ValueError("Dataset is shorter than the requested untouched holdout.")
    development_cut_time = candles["candle_time"].iloc[len(candles) - holdout_rows]

    model_outputs: dict[str, dict] = {}
    training_audit: list[dict] = []

    for all_in_cost_bps in unique_costs:
        labels = rv.barrier_labels(
            candles, HORIZON_BARS, all_in_cost_bps / 10_000.0
        )
        y3 = labels["label3"].to_numpy()
        y_all = np.where(y3 == -1, 0, np.where(y3 == 0, 1, 2)).astype("int8")
        weights_all = labels["sample_weight"].to_numpy(float)
        exit_returns_all = labels["exit_ret"].to_numpy(float)
        valid = labels["valid"].to_numpy(bool)
        usable = valid & X_all.notna().all(axis=1).to_numpy()
        original_indices = np.flatnonzero(usable)
        Xi = X_all.loc[usable].reset_index(drop=True).fillna(0.0)
        yi = y_all[usable]
        wi = weights_all[usable]
        ri = exit_returns_all[usable]
        ti = candles["candle_time"].loc[usable].reset_index(drop=True)
        cut_idx = int((ti < development_cut_time).sum())

        folds = pv.make_folds(
            cut_idx, BARS_PER_DAY * train_days,
            BARS_PER_DAY * test_days, HORIZON_BARS + 2, folds_count,
        )
        if not folds:
            raise ValueError(
                f"Insufficient development history for cost {all_in_cost_bps:g} bps."
            )

        oos_by_pct = {pct: [] for pct in pcts}
        fold_raw_by_pct = {pct: [] for pct in pcts}
        thresholds_last: dict[float, tuple[float, float]] = {}
        final_model = None

        for fold_no, (tr0, tr1, te0, te1) in enumerate(folds, 1):
            # Calibration tail is excluded from both fitting and early stopping.
            cal0 = max(
                tr0 + 100,
                tr1 - BARS_PER_DAY * min(90, train_days),
            )
            if cal0 >= tr1:
                raise ValueError("Training fold is too short for a held-back calibration tail.")
            model = rv.fit_model(
                Xi.iloc[tr0:cal0], yi[tr0:cal0], wi[tr0:cal0], HORIZON_BARS
            )
            p_cal = rv.predict_proba_3class(model, Xi.iloc[cal0:tr1])
            p_test = rv.predict_proba_3class(model, Xi.iloc[te0:te1])

            for pct in pcts:
                thr_up, thr_dn = thresholds_for_selection(p_cal, pct)
                selected, sides, _ = rv.select_trades(
                    p_test[:, 2], p_test[:, 0], ri[te0:te1],
                    HORIZON_BARS, 0.0, thr_up, thr_dn,
                )
                records = build_trade_records(
                    selected, sides, te0, original_indices, labels, candles,
                    funding_times_ns, funding_rates, funding_marks, fold_no,
                )
                oos_by_pct[pct].extend(records)
                fold_raw_by_pct[pct].append({
                    "fold": fold_no,
                    "test_start_utc": pd.Timestamp(ti.iloc[te0]).isoformat(),
                    "test_end_utc": pd.Timestamp(ti.iloc[te1 - 1]).isoformat(),
                    "threshold_up": thr_up,
                    "threshold_down": thr_dn,
                    "test_rows": int(te1 - te0),
                    "raw_trades": records,
                })
                thresholds_last[pct] = (thr_up, thr_dn)
            final_model = model

        if final_model is None:
            raise RuntimeError(f"No model was trained for {all_in_cost_bps:g} bps")
        p_holdout = rv.predict_proba_3class(final_model, Xi.iloc[cut_idx:])
        holdout_by_pct: dict[float, list[dict]] = {}
        for pct in pcts:
            thr_up, thr_dn = thresholds_last[pct]
            selected, sides, _ = rv.select_trades(
                p_holdout[:, 2], p_holdout[:, 0], ri[cut_idx:],
                HORIZON_BARS, 0.0, thr_up, thr_dn,
            )
            holdout_by_pct[pct] = build_trade_records(
                selected, sides, cut_idx, original_indices, labels, candles,
                funding_times_ns, funding_rates, funding_marks, None,
            )
        key = f"{all_in_cost_bps:.6f}"
        model_outputs[key] = {
            "oos_raw": oos_by_pct,
            "fold_raw": fold_raw_by_pct,
            "holdout_raw": holdout_by_pct,
            "feature_count": int(Xi.shape[1]),
            "fold_count": int(len(folds)),
        }
        training_audit.append({
            "label_all_in_cost_bps": float(all_in_cost_bps),
            "model_fits": int(len(folds)),
            "folds": int(len(folds)),
            "feature_count": int(Xi.shape[1]),
            "threshold_calibration": "per-fold development-only calibration tail",
            "holdout_use": "final frozen model and thresholds only; never fit or tune",
        })

    scenarios_out: dict[str, dict] = {}
    gates_out: list[dict] = []
    for scenario in fee_scenarios:
        model_key = scenario["model_cost_key"]
        model_result = model_outputs[model_key]
        selection_out: dict[str, dict] = {}
        for pct in pcts:
            raw_oos = model_result["oos_raw"][pct]
            raw_holdout = model_result["holdout_raw"][pct]
            oos_result, scored_oos = _scenario_trade_metrics(
                raw_oos, scenario, capital_sizes, initial_capital, cash_reserve_pct
            )
            hold_result, scored_holdout = _scenario_trade_metrics(
                raw_holdout, scenario, capital_sizes, initial_capital, cash_reserve_pct
            )

            fold_results = []
            fold_expectancies = []
            for fold in model_result["fold_raw"][pct]:
                scored_fold = [
                    add_execution_costs(
                        t, scenario["entry_fee_bps"], scenario["exit_fee_bps"],
                        scenario["slippage_round_trip_bps"], scenario["bnb_discount_pct"],
                    )
                    for t in fold["raw_trades"]
                ]
                fold_metrics = summarize_net_trades(scored_fold)
                fold_expectancies.append(fold_metrics["expectancy_bps"])
                fold_results.append({
                    "fold": fold["fold"],
                    "test_start_utc": fold["test_start_utc"],
                    "test_end_utc": fold["test_end_utc"],
                    "threshold_up": fold["threshold_up"],
                    "threshold_down": fold["threshold_down"],
                    **fold_metrics,
                })

            valid_fold_expectancies = [
                x for x in fold_expectancies if x is not None
            ]
            positive_fold_count = sum(
                x is not None and x > 0 for x in fold_expectancies
            )
            # A fold with no executed trades is not counted as a positive fold.
            positive_fold_pct = (
                float(positive_fold_count / max(1, len(fold_expectancies)) * 100.0)
            )
            median_fold_exp = (
                float(np.median(valid_fold_expectancies))
                if valid_fold_expectancies else None
            )
            oos_stats = oos_result["metrics"]
            hold_stats = hold_result["metrics"]
            oos_pf = oos_stats["profit_factor"]
            hold_pf = hold_stats["profit_factor"]
            ci_low = oos_stats["bootstrap_ci95_expectancy_bps"][0]

            checks = {
                "median_fold_expectancy_positive": bool(
                    median_fold_exp is not None and median_fold_exp > 0
                ),
                "positive_folds_at_least_75_pct": bool(positive_fold_pct >= 75.0),
                "oos_profit_factor_at_least_1_30": bool(
                    oos_pf is not None and oos_pf >= 1.30
                ),
                "oos_bootstrap_lower_bound_positive": bool(
                    ci_low is not None and ci_low > 0
                ),
                "holdout_expectancy_positive": bool(
                    hold_stats["expectancy_bps"] is not None
                    and hold_stats["expectancy_bps"] > 0
                ),
                "holdout_profit_factor_above_1": bool(
                    hold_pf is not None and hold_pf > 1.0
                ),
                "holdout_bootstrap_lower_bound_positive": bool(
                    hold_stats["bootstrap_ci95_expectancy_bps"][0] is not None
                    and hold_stats["bootstrap_ci95_expectancy_bps"][0] > 0
                ),
                "holdout_at_least_30_trades": bool(hold_stats["trades"] >= 30),
            }
            verdict = "GO" if all(checks.values()) else "NO-GO"
            selection_name = f"{pct * 100:g}%"
            selection_out[selection_name] = {
                "oos": {
                    **oos_stats,
                    "median_fold_expectancy_bps": median_fold_exp,
                    "positive_folds_pct": positive_fold_pct,
                    "folds": fold_results,
                    "capital_scenarios": oos_result["capital_scenarios"],
                },
                "untouched_holdout": {
                    **hold_stats,
                    "capital_scenarios": hold_result["capital_scenarios"],
                    "note": (
                        "Frozen final development model and thresholds. "
                        "Funding settlement rows are actual historical Binance data; "
                        "boundary events are adverse-only due 15m exit-time uncertainty."
                    ),
                },
                "checks": checks,
                "verdict": verdict,
            }
            gates_out.append({
                "fee_scenario": scenario["name"],
                "selection": selection_name,
                "passed": int(sum(checks.values())),
                "total": len(checks),
                "verdict": verdict,
            })

        scenarios_out[scenario["name"]] = {
            "fee_mode": scenario["fee_mode"],
            "maker_fee_bps_per_side": maker_fee_bps,
            "taker_fee_bps_per_side": taker_fee_bps,
            "entry_fee_bps": scenario["entry_fee_bps"],
            "exit_fee_bps": scenario["exit_fee_bps"],
            "bnb_discount_pct": scenario["bnb_discount_pct"],
            "slippage_round_trip_bps": scenario["slippage_round_trip_bps"],
            "label_all_in_cost_bps": scenario["label_all_in_cost_bps"],
            "selections": selection_out,
        }

    return {
        "metadata": {
            "generated_at_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
            "source": source,
            "symbol": "BTCUSDT",
            "market": "Binance USDⓈ-M perpetual futures funding schedule; candle input from provided PostgreSQL table",
            "rows": int(len(candles)),
            "first_candle_utc": pd.Timestamp(candles["candle_time"].iloc[0]).isoformat(),
            "last_candle_utc": pd.Timestamp(candles["candle_time"].iloc[-1]).isoformat(),
            "horizon": "1d",
            "horizon_bars": HORIZON_BARS,
            "funding": funding_coverage,
            "funding_method": (
                "Actual Binance funding settlement timestamps and rates. API markPrice is "
                "used when available; Binance Vision archive rows use the previous completed "
                "BTCUSDT 15m close as an explicit causal proxy. Funding events within entry "
                "or exit boundary candles count payments but ignore potential receipts."
            ),
            "funding_mark_price_source": str(
                funding.attrs.get("mark_price_source", "unknown")
            ),
            "funding_archive_coverage_end_utc": funding.attrs.get(
                "archive_coverage_end_utc"
            ),
            "btc_candles_truncated_for_monthly_funding_archive": bool(
                funding.attrs.get("archive_truncation_required", False)
            ),
            "label_cost_components": (
                "Model barriers are calibrated to commission plus slippage; variable "
                "historical funding is applied only to post-prediction trade PnL."
            ),
            "fees": "Entry and exit commission use notional at each fill; exit notional scales with exit/entry price.",
            "slippage": "User-defined round-trip stress assumption; not estimated from order-book replay.",
            "capital_method": (
                "No leverage assumed. Position notional is capped by $25/$50/$100 and "
                "90% of current equity by default; no upward sizing above the selected cap."
            ),
            "capital_comparison_rule": (
                "OOS and untouched-holdout portfolio simulations each start from the "
                "initial capital independently; their PnL values must not be added together."
            ),
            "multiple_comparison_warning": (
                "Fee, slippage and selection grids create multiple candidate results. "
                "Choose candidates on development folds only, then confirm once on a "
                "new forward period; do not select a winner from the holdout table."
            ),
            "risk_limitations": (
                "No exchange maintenance-margin or liquidation engine, order-book queue/fill "
                "probability, exact intrabar exit seconds, taxes, or withdrawal costs."
            ),
            "live_orders": False,
        },
        "config": {
            "initial_capital_usdt": float(initial_capital),
            "position_caps_usdt": [float(x) for x in capital_sizes],
            "cash_reserve_pct": float(cash_reserve_pct),
            "maker_fee_bps_per_side": float(maker_fee_bps),
            "taker_fee_bps_per_side": float(taker_fee_bps),
            "bnb_discount_pct": float(bnb_discount_pct),
            "slippage_round_trip_bps": [float(x) for x in slippage_grid],
            "selection_pcts_per_side": [float(x) for x in pcts],
            "folds": int(folds_count),
            "train_days": int(train_days),
            "test_days": int(test_days),
            "holdout_days": int(holdout_days),
        },
        "model_training_audit": training_audit,
        "scenarios": scenarios_out,
        "gates": gates_out,
    }


def _parse_float_list(value: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated numbers") from exc
    if not values or any(not np.isfinite(x) for x in values):
        raise SystemExit(f"{name} must contain finite numbers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BTCUSDT 1d fee/funding/slippage and $100 account stress test. Paper only."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv(
            "OSH_DB_URL",
            "postgresql+psycopg2://postgres@127.0.0.1:5433/osh",
        ),
        help="SQLAlchemy PostgreSQL URL; defaults to OSH_DB_URL when set.",
    )
    parser.add_argument("--table", default="public.btcusdt_15m")
    parser.add_argument("--funding-csv", default="binance_btcusdt_funding_history.csv")
    parser.add_argument(
        "--fetch-funding", action="store_true",
        help="Download historical funding rows from Binance REST or completed monthly Vision archives.",
    )
    parser.add_argument(
        "--funding-source", choices=("auto", "api", "vision"), default="auto",
        help="auto tries REST then monthly Vision archives; vision bypasses the REST host.",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--maker-fee-bps", type=float, default=2.0)
    parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser.add_argument("--bnb-discount-pct", type=float, default=0.0)
    parser.add_argument("--slippage-bps", default="0,2,5")
    parser.add_argument("--selection-pcts", default="0.02,0.05,0.10")
    parser.add_argument("--position-caps", default="25,50,100")
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--cash-reserve-pct", type=float, default=10.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument(
        "--allow-vision-monthly-truncate",
        action="store_true",
        help=(
            "When Vision is used and REST is unavailable, explicitly truncate BTC candles "
            "to the last completed UTC month covered by the monthly funding archive."
        ),
    )
    parser.add_argument("--out", default="capital_fee_stress_1d.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    maker_fee = float(args.maker_fee_bps)
    taker_fee = float(args.taker_fee_bps)
    bnb_discount = float(args.bnb_discount_pct)
    slippage_grid = _parse_float_list(args.slippage_bps, "--slippage-bps")
    selection_pcts = _parse_float_list(args.selection_pcts, "--selection-pcts")
    position_caps = _parse_float_list(args.position_caps, "--position-caps")

    if min(args.folds, args.train_days, args.test_days, args.holdout_days) < 1:
        raise SystemExit("Fold count and all date windows must be positive")
    if args.initial_capital <= 0 or any(x <= 0 for x in position_caps):
        raise SystemExit("Initial capital and position caps must be positive")
    if any(x > args.initial_capital for x in position_caps):
        raise SystemExit("Position caps cannot exceed initial capital")
    if not 0 <= args.cash_reserve_pct < 100:
        raise SystemExit("Cash reserve must be in [0, 100)")
    if maker_fee < 0 or taker_fee < 0 or not 0 <= bnb_discount <= 100:
        raise SystemExit("Fee inputs are invalid")

    candles = rv.load_btc(args.db_url, args.table)

    funding_path = Path(args.funding_csv)
    if args.fetch_funding:
        start_ms = int(pd.Timestamp(candles["candle_time"].iloc[0]).timestamp() * 1000)
        end_ms = int(
            (pd.Timestamp(candles["candle_time"].iloc[-1])
             + pd.Timedelta(minutes=rv.TF_MINUTES)).timestamp() * 1000
        )
        print("[funding] Fetching public historical funding records from Binance...")
        funding = None
        if args.funding_source in ("auto", "api"):
            try:
                funding = fetch_binance_funding_history(args.symbol, start_ms, end_ms)
                funding.attrs["funding_source"] = "binance_rest_api"
                print("[funding] Source: USD-M REST API (actual API markPrice).")
            except RuntimeError as exc:
                if args.funding_source == "api":
                    raise
                print(f"[funding] REST API unavailable: {exc}")
                print("[funding] Switching to Binance Vision monthly/daily funding archives.")
        if funding is None:
            needs_truncation, archive_cutoff = vision_monthly_coverage(end_ms)
            if needs_truncation and not args.allow_vision_monthly_truncate:
                cutoff_text = archive_cutoff.isoformat()
                raise RuntimeError(
                    "Binance Vision fundingRate fallback has completed monthly archives only; "
                    "the current UTC month is not yet archived. Restore Binance REST access or "
                    "supply a funding CSV for the current month. To run a clearly truncated "
                    f"research sample ending at {cutoff_text}, rerun with "
                    "--allow-vision-monthly-truncate."
                )
            funding = fetch_binance_vision_funding_history(
                args.symbol, start_ms, end_ms,
            )
            if needs_truncation:
                candles = candles.loc[
                    pd.to_datetime(candles["candle_time"], utc=True) <= archive_cutoff
                ].copy().reset_index(drop=True)
                if candles.empty:
                    raise RuntimeError(
                        "Monthly funding archive cutoff leaves no BTC candles to test."
                    )
                print(
                    "[funding] WARNING: current-month candles excluded because their "
                    f"funding archive is not yet published. Last candle kept: {archive_cutoff}."
                )
            print(
                "[funding] Source: Binance Vision completed monthly archives; markPrice "
                "will be proxied from the previous completed BTC 15m candle."
            )
        funding = attach_funding_mark_price_proxy(funding, candles)
        funding["mark_price_source"] = str(
            funding.attrs.get("mark_price_source", "unknown")
        )
        funding_path.parent.mkdir(parents=True, exist_ok=True)
        funding.to_csv(funding_path, index=False)
        print(
            f"[funding] Saved {len(funding):,} settlement rows to {funding_path.resolve()} "
            f"(mark-price source: {funding.attrs.get('mark_price_source', 'unknown')})"
        )
    else:
        funding = load_funding_history(funding_path, symbol=args.symbol)

    result = run_capital_fee_stress(
        candles=candles,
        funding=funding,
        selection_pcts=selection_pcts,
        slippage_grid=slippage_grid,
        maker_fee_bps=maker_fee,
        taker_fee_bps=taker_fee,
        bnb_discount_pct=bnb_discount,
        capital_sizes=position_caps,
        initial_capital=args.initial_capital,
        cash_reserve_pct=args.cash_reserve_pct,
        folds_count=args.folds,
        train_days=args.train_days,
        test_days=args.test_days,
        holdout_days=args.holdout_days,
        source=f"PostgreSQL {args.table}",
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== OSH BTCUSDT 1D CAPITAL / COST STRESS — PAPER ONLY ===")
    print(
        "Fee mode | Slip | Sel | OOS N | OOS win% | OOS PF | OOS exp | "
        "Hold N | Hold win% | Hold exp | "
        "OOS $25 | OOS $50 | OOS $100 | "
        "Hold $25 | Hold $50 | Hold $100 | Verdict"
    )
    for scenario_name, scenario in result["scenarios"].items():
        for selection_name, selection in scenario["selections"].items():
            oos = selection["oos"]
            hold = selection["untouched_holdout"]
            caps_oos = oos["capital_scenarios"]
            caps_hold = hold["capital_scenarios"]
            print(
                f"{scenario_name:<28} | "
                f"{scenario['slippage_round_trip_bps']:>4.1f} | "
                f"{selection_name:>3} | "
                f"{oos['trades']:>5} | "
                f"{oos['win_rate_pct'] if oos['win_rate_pct'] is not None else 'n/a':>7} | "
                f"{oos['profit_factor'] if oos['profit_factor'] is not None else 'n/a':>6} | "
                f"{oos['expectancy_bps'] if oos['expectancy_bps'] is not None else 'n/a':>8} | "
                f"{hold['trades']:>6} | "
                f"{hold['win_rate_pct'] if hold['win_rate_pct'] is not None else 'n/a':>9} | "
                f"{hold['expectancy_bps'] if hold['expectancy_bps'] is not None else 'n/a':>9} | "
                f"{caps_oos.get('25', {}).get('net_pnl_usdt', 0):>7.2f} | "
                f"{caps_oos.get('50', {}).get('net_pnl_usdt', 0):>7.2f} | "
                f"{caps_oos.get('100', {}).get('net_pnl_usdt', 0):>8.2f} | "
                f"{caps_hold.get('25', {}).get('net_pnl_usdt', 0):>8.2f} | "
                f"{caps_hold.get('50', {}).get('net_pnl_usdt', 0):>8.2f} | "
                f"{caps_hold.get('100', {}).get('net_pnl_usdt', 0):>9.2f} | "
                f"{selection['verdict']}"
            )
    print(f"[report] {output.resolve()}")
    print("[safety] live_orders=false; no API trading keys or order endpoints are used.")


if __name__ == "__main__":
    main()
