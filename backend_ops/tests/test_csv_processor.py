from app.core.csv_processor import parse_fund_csv, CSVValidationError


def test_parse_and_clean_csv():
    content = b"Date,NAV\n04-01-2020,10.0\n05-01-2020,10.5\n05-01-2020,10.6\nbad,11\n"
    parsed = parse_fund_csv(content, "Equity_Fund.csv")
    assert parsed.fund_name == "Equity_Fund"
    assert len(parsed.data) == 2
    assert parsed.data["nav"].tolist() == [10.0, 10.5]
    assert parsed.cleaning["duplicate_dates_removed"] == 1
    assert parsed.cleaning["invalid_dates_removed"] == 1


def test_rejects_missing_nav_column():
    try:
        parse_fund_csv(b"Date,Price\n01-01-2020,10\n", "bad.csv")
    except CSVValidationError:
        return
    raise AssertionError("Expected CSVValidationError")
