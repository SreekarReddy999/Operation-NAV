from __future__ import annotations

from typing import Annotated

from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.core.csv_processor import CSVValidationError, parse_fund_file
from app.core.forecast_engine import analyse_funds


app = FastAPI(
    title="Fund NAV Analysis API",
    version="0.1.0",
    description=(
        "Accepts one CSV, XLS, or XLSX file per fund (Date and NAV columns), "
        "analyses each fund independently, and returns separate historical "
        "and 12-month forecast chart series."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "fund-nav-analysis-api", "version": "0.1.0"}


@app.post("/api/v1/analyze")
async def analyze_fund_files(
        files: Annotated[
            list[UploadFile],
            File(description="Upload one CSV, XLS, or XLSX file per fund."),
        ],
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one fund file.",
        )

    parsed_funds = []
    file_errors = []

    for upload in files:
        filename = upload.filename or "unnamed_file"

        try:
            content = await upload.read()

            if len(content) > 25 * 1024 * 1024:
                file_errors.append({
                    "filename": filename,
                    "error": "File exceeds the 25 MB upload limit.",
                })
                continue

            parsed_funds.append(
                parse_fund_file(content, filename)
            )

        except CSVValidationError as exc:
            file_errors.append({
                "filename": filename,
                "error": str(exc),
            })

        except Exception as exc:
            file_errors.append({
                "filename": filename,
                "error": f"Unexpected processing error: {exc}",
            })

        finally:
            await upload.close()

    if not parsed_funds:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No valid fund files were supplied.",
                "file_errors": file_errors,
            },
        )

    funds_for_analysis = [
        {
            "fund_name": parsed.fund_name,
            "source_file": parsed.source_file,
            "data": parsed.data,
            "warnings": parsed.warnings,
            "cleaning": parsed.cleaning,
        }
        for parsed in parsed_funds
    ]

    try:
        results = analyse_funds(funds_for_analysis)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Fund analysis failed: {exc}",
        ) from exc

    return {
        "application": "Fund NAV Analysis",
        "fund_count": len(results),
        "chart_contract": {
            "one_result_per_fund": True,
            "historical_series": "historical",
            "forecast_series": "forecast",
            "x_axis": "date",
            "y_axis": "nav",
        },
        "funds": results,
        "file_errors": file_errors,
        "disclaimer": (
            "Forecasts are estimates, not guarantees or investment "
            "recommendations. Historical performance does not "
            "guarantee future results."
        ),
    }


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Use OpenAPI 3.0.3 for Swagger UI file-upload compatibility
    schema["openapi"] = "3.0.3"

    components = schema.get("components", {}).get("schemas", {})

    for component in components.values():
        properties = component.get("properties", {})

        for prop in properties.values():

            # Handle arrays of uploaded files
            items = prop.get("items", {})

            if items.get("contentMediaType") == "application/octet-stream":
                items.pop("contentMediaType", None)
                items["format"] = "binary"

            # Handle individual uploaded files
            if prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi