"""CSV/Spreadsheet reader — reads CSV/Excel files with stats and previews."""

import asyncio
import os
from typing import List, Literal, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import truncate, safe_tool_call, require_input
from src.constants import CSV_MAX_OUTPUT_CHARS

# ─── Module overview ───────────────────────────────────────────────
# Reads CSV/Excel/TSV files and returns column info, statistics,
# sample rows, optional filtering, and groupby aggregations.
# Schema is enforced via args_schema.
# ───────────────────────────────────────────────────────────────────

AggFunc = Literal["sum", "mean", "count", "min", "max"]


# Takes (path, head, describe, columns, filter_column, filter_value,
# groupby, agg, agg_column). Reads file with pandas and returns formatted output.
@safe_tool_call("reading file")
async def csv_reader(
    path: str,
    head: int = 5,
    describe: bool = True,
    columns: Optional[List[str]] = None,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    groupby: Optional[str] = None,
    agg: AggFunc = "sum",
    agg_column: Optional[str] = None,
) -> str:
    """Read a CSV/Excel/TSV file and return columns, stats, and sample rows."""
    try:
        import pandas as pd
    except ImportError:
        return "Error: pandas is not installed. Run: pip install pandas openpyxl"

    err = require_input(path or "", "file_path")
    if err:
        return err

    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            return f"Error: Unsupported file type '{ext}'. Supported: .csv, .xlsx, .xls, .tsv"

        # Apply filter if specified
        if filter_column and filter_value is not None and filter_column in df.columns:
            df = df[df[filter_column].astype(str).str.contains(str(filter_value), case=False, na=False)]

        # Select specific columns
        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            if valid_cols:
                df = df[valid_cols]

        # Handle groupby aggregation
        if groupby and groupby in df.columns:
            if agg_column and agg_column in df.columns:
                result_df = df.groupby(groupby)[agg_column].agg(agg)
                return (
                    f"**Aggregation: {agg}({agg_column}) by {groupby}**\n\n"
                    f"{result_df.to_markdown()}"
                )
            result_df = df.groupby(groupby).size()
            return (
                f"**Count by {groupby}:**\n\n"
                f"{result_df.to_markdown()}"
            )

        sections = []

        sections.append(
            f"**File:** {os.path.basename(path)}\n"
            f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"**Columns:** {', '.join(df.columns.tolist())}"
        )

        # Column types and missing values
        type_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_str = f" ({missing} missing)" if missing > 0 else ""
            type_info.append(f"  - **{col}**: {dtype}{missing_str}")
        sections.append("**Column Info:**\n" + "\n".join(type_info))

        if describe:
            numeric_df = df.select_dtypes(include=["number"])
            if not numeric_df.empty:
                desc = numeric_df.describe().round(2)
                sections.append(f"**Statistics:**\n{desc.to_markdown()}")

        head_rows = min(head, len(df))
        if head_rows > 0:
            sample = df.head(head_rows)
            sections.append(f"**First {head_rows} rows:**\n{sample.to_markdown(index=False)}")

        result = "\n\n".join(sections)
        return truncate(result, CSV_MAX_OUTPUT_CHARS, "\n\n[Output truncated]")

    except Exception as e:
        return f"Error reading file: {str(e)}"


class CsvReaderInput(BaseModel):
    """Inputs for the csv_reader tool."""
    path: str = Field(description="Path to the CSV/XLSX/XLS/TSV file.")
    head: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Number of sample rows to show (0-100).",
    )
    describe: bool = Field(
        default=True,
        description="Whether to include statistical summary for numeric columns.",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of column names to keep (others dropped).",
    )
    filter_column: Optional[str] = Field(
        default=None,
        description="Column name to filter on (use with filter_value).",
    )
    filter_value: Optional[str] = Field(
        default=None,
        description="Substring to match against filter_column (case-insensitive).",
    )
    groupby: Optional[str] = Field(
        default=None,
        description="Column to group rows by. Combine with agg + agg_column for aggregation.",
    )
    agg: AggFunc = Field(
        default="sum",
        description="Aggregation function: sum, mean, count, min, or max.",
    )
    agg_column: Optional[str] = Field(
        default=None,
        description="Column to aggregate. If omitted, groupby returns row counts.",
    )


class CsvReaderTool(BaseTool):
    name: str = "csv_reader"
    description: str = (
        "Read and analyze CSV or Excel spreadsheet files. Returns column info, statistics, "
        "and sample data. Can filter rows and perform aggregations."
        "\n\nUSE FOR:"
        "\n- Reading data files: path='data/sales.csv'"
        "\n- Previewing data: path='data.csv', head=10"
        "\n- Filtering: path='data.csv', filter_column='country', filter_value='USA'"
        "\n- Aggregation: path='data.csv', groupby='category', agg='sum', agg_column='revenue'"
        "\n\nSUPPORTS: .csv, .xlsx, .xls, .tsv"
        "\n\nDO NOT USE FOR: creating charts (use create_chart), complex analysis (use python_repl)"
    )
    args_schema: Type[BaseModel] = CsvReaderInput

    # Forwards every validated parameter to csv_reader.
    async def _arun(self, **kwargs) -> str:
        return await csv_reader(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


csv_tool = CsvReaderTool()
