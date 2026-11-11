from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

# 0 means the first worksheet.
# If your production workbook has a known sheet name, you can
# replace 0 with that exact sheet name.
NAV_SHEET_NAME = 0

MIN_RECOMMENDED_OBSERVATIONS = 252

DATE_ALIASES = {
    "date",
    "effectivedate",
    "navdate",
    "valuedate",
    "asofdate",
}

NAV_ALIASES = {
    "nav",
    "navvalue",
    "netassetvalue",
    "unitnav",
}


# ---------------------------------------------------------
# Custom exception
# ---------------------------------------------------------

class CSVValidationError(Exception):
    """Raised when an uploaded fund file cannot be validated."""


# ---------------------------------------------------------
# Parsed output structure
# ---------------------------------------------------------

@dataclass
class ParsedFund:
    fund_name: str
    source_file: str
    data: pd.DataFrame
    warnings: list[str]
    cleaning: dict[str, Any]


# ---------------------------------------------------------
# Text and header normalization
# ---------------------------------------------------------

def normalize_text(value: Any) -> str:
    """
    Normalize header text by removing spaces and special
    characters and converting it to lowercase.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    return re.sub(r"[^a-z0-9]", "", text)


def normalise_header(value: Any) -> str:
    """Backward-compatible alias for header normalization."""
    return normalize_text(value)


# ---------------------------------------------------------
# CSV reader
# ---------------------------------------------------------

def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    """
    Read CSV content while trying common file encodings.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
            )
        except UnicodeDecodeError as error:
            last_error = error
        except pd.errors.EmptyDataError as error:
            raise CSVValidationError(
                "The CSV file is empty."
            ) from error

    raise CSVValidationError(
        f"Could not decode CSV file. Last error: {last_error}"
    )


# ---------------------------------------------------------
# Excel reader
# ---------------------------------------------------------

def read_sbi_raw(
    content: bytes,
    filename: str,
    sheet_name=NAV_SHEET_NAME,
) -> pd.DataFrame:
    """
    Read an Excel worksheet without assuming that the
    first row contains the actual column headers.

    This supports report-style SBI Excel files where
    titles, blank rows or report information appear
    before the NAV table.
    """

    extension = Path(filename).suffix.lower()

    if extension == ".xls":
        engine = "xlrd"
    elif extension == ".xlsx":
        engine = "openpyxl"
    else:
        raise CSVValidationError(
            f"Unsupported Excel extension: {extension}"
        )

    try:
        raw_df = pd.read_excel(
            io.BytesIO(content),
            sheet_name=sheet_name,
            header=None,
            engine=engine,
        )

        if raw_df.empty:
            raise CSVValidationError(
                "Workbook sheet is empty."
            )

        return raw_df

    except CSVValidationError:
        raise
    except Exception as error:
        raise CSVValidationError(
            f"Failed to read {filename}: {error}"
        ) from error


# ---------------------------------------------------------
# Excel NAV header detection
# ---------------------------------------------------------

def find_nav_header_row(raw_df: pd.DataFrame):
    """
    Scan every worksheet row and locate the row containing
    both an effective-date header and a NAV header.

    Returns the matching row index, or None if not found.
    """

    for index, row in raw_df.iterrows():

        normalized_values = {
            normalize_text(value)
            for value in row.tolist()
            if normalize_text(value)
        }

        has_date = bool(
            normalized_values.intersection(DATE_ALIASES)
        )

        has_nav = bool(
            normalized_values.intersection(NAV_ALIASES)
        )

        if has_date and has_nav:
            return index

    return None


# ---------------------------------------------------------
# Extract NAV table from Excel
# ---------------------------------------------------------

def extract_nav_table(
    raw_df: pd.DataFrame,
    header_row: int,
) -> pd.DataFrame:
    """
    Extract only the date and NAV columns from the
    detected header row onwards.
    """

    header = raw_df.iloc[header_row]

    date_columns = []
    nav_columns = []

    for col_index, value in header.items():

        normalized = normalize_text(value)

        if normalized in DATE_ALIASES:
            date_columns.append(col_index)

        elif normalized in NAV_ALIASES:
            nav_columns.append(col_index)

    if len(date_columns) != 1 or len(nav_columns) != 1:
        raise CSVValidationError(
            "Could not uniquely identify the Effective Date "
            "and NAV columns."
        )

    effective_date_col = date_columns[0]
    nav_col = nav_columns[0]

    nav_df = raw_df.iloc[header_row + 1:][
        [effective_date_col, nav_col]
    ].copy()

    nav_df.columns = ["date", "nav"]

    return (
        nav_df
        .dropna(how="all")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Read Excel and locate NAV table
# ---------------------------------------------------------

def _read_excel_bytes(
    content: bytes,
    filename: str,
) -> pd.DataFrame:
    """
    Read the configured worksheet, detect the real NAV
    header row, and return the extracted NAV table.
    """

    raw_df = read_sbi_raw(
        content=content,
        filename=filename,
        sheet_name=NAV_SHEET_NAME,
    )

    header_row = find_nav_header_row(raw_df)

    if header_row is None:
        raise CSVValidationError(
            f"{filename}: Could not find a row containing "
            "both an Effective Date and NAV header. "
            "Please check the selected worksheet."
        )

    return extract_nav_table(
        raw_df=raw_df,
        header_row=header_row,
    )


# ---------------------------------------------------------
# Common file reader
# ---------------------------------------------------------

def _read_fund_file(
    content: bytes,
    filename: str,
) -> pd.DataFrame:
    """
    Read CSV, XLS or XLSX and return a DataFrame
    containing date and nav columns.
    """

    extension = Path(filename).suffix.lower()

    if extension == ".csv":

        raw_df = _read_csv_bytes(content)

        # CSV files are expected to have headers.
        normalized_headers = {
            normalize_text(column): column
            for column in raw_df.columns
        }

        date_columns = [
            original
            for normalized, original in normalized_headers.items()
            if normalized in DATE_ALIASES
        ]

        nav_columns = [
            original
            for normalized, original in normalized_headers.items()
            if normalized in NAV_ALIASES
        ]

        if len(date_columns) != 1 or len(nav_columns) != 1:
            raise CSVValidationError(
                f"{filename}: Expected exactly one Date/Effective "
                f"Date column and one NAV/NAV Value column. "
                f"Found columns: {list(raw_df.columns)}"
            )

        nav_df = raw_df[
            [date_columns[0], nav_columns[0]]
        ].copy()

        nav_df.columns = ["date", "nav"]

        return nav_df

    if extension in {".xls", ".xlsx"}:
        return _read_excel_bytes(
            content=content,
            filename=filename,
        )

    raise CSVValidationError(
        f"{filename}: Unsupported file format. "
        "Supported formats are CSV, XLS and XLSX."
    )


# ---------------------------------------------------------
# NAV value cleaning
# ---------------------------------------------------------

def _clean_nav_values(nav_series: pd.Series) -> pd.Series:
    """
    Remove common currency formatting before converting
    NAV values into numeric values.
    """

    cleaned = (
        nav_series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


# ---------------------------------------------------------
# Common parser for CSV, XLS and XLSX
# ---------------------------------------------------------

def parse_fund_file(
    content: bytes,
    filename: str,
) -> ParsedFund:
    """
    Parse and validate a fund NAV file.

    Supported formats:
    - CSV
    - XLS
    - XLSX

    Output columns:
    - date
    - fund_name
    - nav
    - source_file
    """

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise CSVValidationError(
            f"{filename}: Unsupported file format. "
            "Supported formats are CSV, XLS and XLSX."
        )

    if not content:
        raise CSVValidationError(
            f"{filename}: The uploaded file is empty."
        )

    try:
        nav_df = _read_fund_file(
            content=content,
            filename=filename,
        )

    except CSVValidationError:
        raise

    except Exception as error:
        raise CSVValidationError(
            f"{filename}: Failed to process file: {error}"
        ) from error

    if nav_df.empty:
        raise CSVValidationError(
            f"{filename}: No NAV records were found."
        )

    # -----------------------------------------------------
    # Track original row count
    # -----------------------------------------------------

    original_rows = len(nav_df)

    # -----------------------------------------------------
    # Date conversion
    # -----------------------------------------------------

    nav_df["date"] = pd.to_datetime(
        nav_df["date"],
        errors="coerce",
        dayfirst=True,
    )

    # -----------------------------------------------------
    # NAV conversion
    # -----------------------------------------------------

    nav_df["nav"] = _clean_nav_values(
        nav_df["nav"]
    )

    # -----------------------------------------------------
    # Remove invalid records
    # -----------------------------------------------------

    invalid_date_rows = int(
        nav_df["date"].isna().sum()
    )

    invalid_nav_rows = int(
        nav_df["nav"].isna().sum()
    )

    non_positive_nav_rows = int(
        (nav_df["nav"] <= 0).sum()
    )

    nav_df = nav_df.dropna(
        subset=["date", "nav"]
    ).copy()

    nav_df = nav_df[
        nav_df["nav"] > 0
    ].copy()

    # -----------------------------------------------------
    # Remove duplicate dates
    # -----------------------------------------------------

    duplicate_date_rows = int(
        nav_df.duplicated(subset=["date"]).sum()
    )

    nav_df = nav_df.drop_duplicates(
        subset=["date"],
        keep="first",
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    nav_df = nav_df.sort_values(
        by="date"
    ).reset_index(drop=True)

    if nav_df.empty:
        raise CSVValidationError(
            f"{filename}: No valid positive NAV records "
            "remain after cleaning."
        )

    # -----------------------------------------------------
    # Standardized fund information
    # -----------------------------------------------------

    fund_name = Path(filename).stem

    nav_df["fund_name"] = fund_name
    nav_df["source_file"] = filename

    nav_df = nav_df[
        [
            "date",
            "fund_name",
            "nav",
            "source_file",
        ]
    ]

    # -----------------------------------------------------
    # Cleaning summary
    # -----------------------------------------------------

    final_rows = len(nav_df)

    removed_rows = original_rows - final_rows

    cleaning = {
        "original_rows": original_rows,
        "final_rows": final_rows,
        "removed_rows": removed_rows,
        "invalid_date_rows": invalid_date_rows,
        "invalid_nav_rows": invalid_nav_rows,
        "non_positive_nav_rows": non_positive_nav_rows,
        "duplicate_date_rows": duplicate_date_rows,
    }

    # -----------------------------------------------------
    # Warnings
    # -----------------------------------------------------

    warnings = []

    if final_rows < MIN_RECOMMENDED_OBSERVATIONS:
        warnings.append(
            f"Only {final_rows} valid NAV observations found. "
            f"At least {MIN_RECOMMENDED_OBSERVATIONS} "
            "observations are recommended for historical analysis."
        )

    if removed_rows > 0:
        warnings.append(
            f"{removed_rows} rows were removed during data cleaning."
        )

    return ParsedFund(
        fund_name=fund_name,
        source_file=filename,
        data=nav_df,
        warnings=warnings,
        cleaning=cleaning,
    )


# ---------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------

def parse_fund_csv(
    content: bytes,
    filename: str,
) -> ParsedFund:
    """
    Preserve compatibility with existing code that calls
    parse_fund_csv(). This wrapper accepts CSV files only.
    """

    if Path(filename).suffix.lower() != ".csv":
        raise CSVValidationError(
            f"{filename}: parse_fund_csv() accepts CSV files only. "
            "Use parse_fund_file() for CSV, XLS and XLSX."
        )

    return parse_fund_file(
        content=content,
        filename=filename,
    )
