
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
    product_name: str | None = None


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
# SBI report metadata extraction
# ---------------------------------------------------------

def extract_report_field(
    raw_df: pd.DataFrame,
    field_name: str,
) -> str | None:
    """
    Extract a report-header value located to the right of
    a label such as Product or Fund.
    """
    target = normalize_text(field_name)

    for _, row in raw_df.iterrows():
        values = row.tolist()

        for col_index, value in enumerate(values):
            if normalize_text(value) != target:
                continue

            # Find the next non-empty cell to the right.
            for next_value in values[col_index + 1:]:
                if pd.notna(next_value):
                    extracted = str(next_value).strip()

                    if extracted:
                        return extracted

    return None


def extract_report_metadata(
    raw_df: pd.DataFrame,
) -> tuple[str, str | None]:
    """
    Extract the Fund and Product names from the SBI report header.
    """
    fund_name = extract_report_field(raw_df, "Fund")
    product_name = extract_report_field(raw_df, "Product")

    if not fund_name:
        raise CSVValidationError(
            "Could not find the Fund name in the Excel report header."
        )

    return fund_name, product_name


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

    Supports report-style SBI Excel files where titles,
    blank rows or report information appear before the NAV table.
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
) -> tuple[pd.DataFrame, str, str | None]:
    """
    Read the configured worksheet, extract report metadata,
    detect the NAV header row, and return the NAV table,
    fund name and product name.
    """
    raw_df = read_sbi_raw(
        content=content,
        filename=filename,
        sheet_name=NAV_SHEET_NAME,
    )

    # Extract Product and Fund names from the report header.
    fund_name, product_name = extract_report_metadata(raw_df)

    # Locate the actual NAV table.
    header_row = find_nav_header_row(raw_df)

    if header_row is None:
        raise CSVValidationError(
            f"{filename}: Could not find a row containing "
            "both an Effective Date and NAV header. "
            "Please check the selected worksheet."
        )

    nav_df = extract_nav_table(
        raw_df=raw_df,
        header_row=header_row,
    )

    return nav_df, fund_name, product_name


# ---------------------------------------------------------
# Common file reader
# ---------------------------------------------------------

def _read_fund_file(
    content: bytes,
    filename: str,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """
    Read CSV, XLS or XLSX and return a DataFrame
    containing date and NAV columns, along with metadata.
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

        return nav_df, None, None

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
        nav_df, extracted_fund_name, product_name = _read_fund_file(
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

    # Excel: use the actual Fund name from the report.
    # CSV: retain the existing filename-based behavior.
    fund_name = extracted_fund_name or Path(filename).stem

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
        product_name=product_name,
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
