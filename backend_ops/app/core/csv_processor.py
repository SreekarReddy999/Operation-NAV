from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

import pandas as pd


class CSVValidationError(ValueError):
    """Raised when an uploaded fund CSV cannot be used."""


@dataclass
class ParsedFund:
    fund_name: str
    source_file: str
    data: pd.DataFrame
    warnings: list[str]
    cleaning: dict[str, int]


def _normalise_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


DATE_ALIASES = {
    "date", "effectivedate", "navdate", "valuedate", "asofdate"
}
NAV_ALIASES = {
    "nav", "navvalue", "netassetvalue", "unitnav"
}


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    if not content:
        raise CSVValidationError("The uploaded CSV is empty.")

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise CSVValidationError("The uploaded CSV contains no tabular data.") from exc
        except Exception as exc:
            last_error = exc

    raise CSVValidationError(f"Could not read the CSV: {last_error}")


def parse_fund_csv(content: bytes, filename: str) -> ParsedFund:
    """
    Parse one CSV representing exactly one fund.

    Required logical columns: Date and NAV. Common header variations are accepted.
    Fund identity is derived from the uploaded filename because the agreed CSV
    format does not contain a Fund Name column.
    """
    if Path(filename).suffix.lower() != ".csv":
        raise CSVValidationError(f"{filename}: only .csv files are supported.")

    raw = _read_csv_bytes(content)
    if raw.empty:
        raise CSVValidationError(f"{filename}: the CSV has a header but no data rows.")

    normalised = [_normalise_header(col) for col in raw.columns]
    date_positions = [i for i, name in enumerate(normalised) if name in DATE_ALIASES]
    nav_positions = [i for i, name in enumerate(normalised) if name in NAV_ALIASES]

    if len(date_positions) != 1 or len(nav_positions) != 1:
        raise CSVValidationError(
            f"{filename}: expected exactly one Date column and one NAV column. "
            f"Recognised headers include Date/Effective Date and NAV/NAV Value. "
            f"Found columns: {list(raw.columns)}"
        )

    date_col = raw.columns[date_positions[0]]
    nav_col = raw.columns[nav_positions[0]]

    frame = raw[[date_col, nav_col]].copy()
    rows_before = len(frame)

    # The Colab notebook uses day-first parsing; retain that convention for Indian NAV exports.
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=True)

    nav_text = (
        frame[nav_col]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.strip()
    )
    frame["nav"] = pd.to_numeric(nav_text, errors="coerce")

    invalid_dates = int(frame["date"].isna().sum())
    invalid_nav = int(frame["nav"].isna().sum())
    non_positive_nav = int((frame["nav"].notna() & (frame["nav"] <= 0)).sum())

    frame = frame.dropna(subset=["date", "nav"])
    frame = frame[frame["nav"] > 0].copy()

    duplicate_dates = int(frame.duplicated(subset=["date"]).sum())
    # Match the existing notebook's cleaning rule: retain the first record for a duplicate date.
    frame = frame.drop_duplicates(subset=["date"], keep="first")
    frame = frame.sort_values("date").reset_index(drop=True)

    if frame.empty:
        raise CSVValidationError(
            f"{filename}: no valid observations remain after date/NAV cleaning."
        )

    fund_name = Path(filename).stem.strip() or "Unnamed fund"
    frame["fund_name"] = fund_name
    frame["source_file"] = Path(filename).name
    frame = frame[["date", "fund_name", "nav", "source_file"]]

    warnings = []
    if invalid_dates:
        warnings.append(f"Removed {invalid_dates} rows with invalid dates.")
    if invalid_nav:
        warnings.append(f"Removed {invalid_nav} rows with missing or invalid NAV.")
    if non_positive_nav:
        warnings.append(f"Removed {non_positive_nav} rows with zero or negative NAV.")
    if duplicate_dates:
        warnings.append(f"Removed {duplicate_dates} duplicate-date rows.")
    if len(frame) < 252:
        warnings.append(
            "Fewer than 252 valid observations: a one-year trading-history forecast "
            "will not be generated."
        )

    return ParsedFund(
        fund_name=fund_name,
        source_file=Path(filename).name,
        data=frame,
        warnings=warnings,
        cleaning={
            "rows_received": rows_before,
            "rows_retained": len(frame),
            "invalid_dates_removed": invalid_dates,
            "invalid_nav_removed": invalid_nav,
            "non_positive_nav_removed": non_positive_nav,
            "duplicate_dates_removed": duplicate_dates,
        },
    )
