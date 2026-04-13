"""CSV/Spreadsheet reader — reads CSV/Excel files with stats and previews."""

import os
from langchain_core.tools import tool
from src.utils import parse_tool_input, truncate, safe_tool_call, require_input
from src.constants import CSV_MAX_OUTPUT_CHARS


# ─── Module overview ───────────────────────────────────────────────
# Reads CSV/Excel/TSV files and returns column info, statistics,
# sample rows, and optional filtering or groupby aggregations.
# ───────────────────────────────────────────────────────────────────

# Takes a query string (file path or JSON with options). Reads the file with pandas.
# Returns formatted column info, stats, and sample data or aggregation results.
@safe_tool_call("reading file")
async def csv_reader(query: str) -> str:
    """Read and analyze CSV or Excel spreadsheet files. Returns column info, statistics, and sample data. Can filter rows and perform aggregations.

    USE FOR:
    - Reading data files: 'data/sales.csv'
    - Previewing data: '{"path": "data.csv", "head": 10}'
    - Filtering: '{"path": "data.csv", "filter": {"column": "country", "value": "USA"}}'
    - Aggregation: '{"path": "data.csv", "groupby": "category", "agg": "sum", "column": "revenue"}'

    SUPPORTS: .csv, .xlsx, .xls, .tsv

    DO NOT USE FOR: creating charts (use create_chart), complex analysis (use python_repl)"""
    try:
        import pandas as pd
    except ImportError:
        return "Error: pandas is not installed. Run: pip install pandas openpyxl"

    # Parse input
    raw_path, options = parse_tool_input(query, {
        "head": 5, "describe": True,
    })
    file_path = options.get("path", raw_path)
    err = require_input(file_path or "", "file_path")
    if err:
        return err
    head_rows = options.get("head", 5)
    show_describe = options.get("describe", True)
    selected_columns = options.get("columns")
    filter_spec = options.get("filter")
    groupby_spec = options.get("groupby")
    agg_func = options.get("agg", "sum")
    agg_column = options.get("column")

    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    try:
        # Read file based on extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
        else:
            return f"Error: Unsupported file type '{ext}'. Supported: .csv, .xlsx, .xls, .tsv"

        # Apply filter if specified
        if filter_spec and isinstance(filter_spec, dict):
            col = filter_spec.get("column", "")
            val = filter_spec.get("value", "")
            if col in df.columns:
                df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]

        # Select specific columns
        if selected_columns:
            valid_cols = [c for c in selected_columns if c in df.columns]
            if valid_cols:
                df = df[valid_cols]

        # Handle groupby aggregation
        if groupby_spec and groupby_spec in df.columns:
            agg_func_name = agg_func
            agg_col = agg_column
            if agg_col and agg_col in df.columns:
                result_df = df.groupby(groupby_spec)[agg_col].agg(agg_func_name)
                return (
                    f"**Aggregation: {agg_func_name}({agg_col}) by {groupby_spec}**\n\n"
                    f"{result_df.to_markdown()}"
                )
            else:
                result_df = df.groupby(groupby_spec).size()
                return (
                    f"**Count by {groupby_spec}:**\n\n"
                    f"{result_df.to_markdown()}"
                )

        # Build output
        sections = []

        # File info
        sections.append(
            f"**File:** {os.path.basename(file_path)}\n"
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

        # Statistical summary for numeric columns
        if show_describe:
            numeric_df = df.select_dtypes(include=["number"])
            if not numeric_df.empty:
                desc = numeric_df.describe().round(2)
                sections.append(f"**Statistics:**\n{desc.to_markdown()}")

        # Sample data
        head_rows = min(head_rows, len(df))
        if head_rows > 0:
            sample = df.head(head_rows)
            sections.append(f"**First {head_rows} rows:**\n{sample.to_markdown(index=False)}")

        result = "\n\n".join(sections)

        result = truncate(result, CSV_MAX_OUTPUT_CHARS, "\n\n[Output truncated]")

        return result

    except Exception as e:
        return f"Error reading file: {str(e)}"


csv_tool = tool(csv_reader)
