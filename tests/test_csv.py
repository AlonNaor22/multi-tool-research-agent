"""Tests for the CSV/Spreadsheet reader tool."""

import pytest
from pydantic import ValidationError


class TestCsvReader:
    """Tests for the CSV reader tool."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_content = "name,age,city,salary\nAlice,30,NYC,75000\nBob,25,LA,65000\nCharlie,35,Chicago,85000\n"
        filepath = tmp_path / "test_data.csv"
        filepath.write_text(csv_content)
        return str(filepath)

    @pytest.fixture
    def sample_tsv(self, tmp_path):
        """Create a temporary TSV file."""
        tsv_content = "name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA\n"
        filepath = tmp_path / "test_data.tsv"
        filepath.write_text(tsv_content)
        return str(filepath)

    @pytest.mark.asyncio
    async def test_basic_csv_read(self, sample_csv):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader(sample_csv)
        assert "3 rows x 4 columns" in result
        assert "name" in result
        assert "age" in result
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_column_info(self, sample_csv):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader(sample_csv)
        assert "Column Info" in result
        assert "salary" in result

    @pytest.mark.asyncio
    async def test_statistics(self, sample_csv):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader(sample_csv)
        assert "Statistics" in result
        assert "mean" in result.lower() or "50%" in result

    @pytest.mark.asyncio
    async def test_custom_head(self, sample_csv):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader(sample_csv, head=2)
        assert "First 2 rows" in result

    @pytest.mark.asyncio
    async def test_filter(self, sample_csv):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader(sample_csv, filter_column="city", filter_value="NYC")
        assert "Alice" in result
        assert "1 rows" in result

    @pytest.mark.asyncio
    async def test_select_columns(self, sample_csv):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader(sample_csv, columns=["name", "city"])
        assert "name" in result
        assert "city" in result

    @pytest.mark.asyncio
    async def test_groupby_aggregation(self, sample_csv):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader(sample_csv, groupby="city", agg="sum", agg_column="salary")
        assert "sum(salary)" in result or "Aggregation" in result

    @pytest.mark.asyncio
    async def test_tsv_support(self, sample_tsv):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader(sample_tsv)
        assert "2 rows x 3 columns" in result
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader("/nonexistent/path.csv")
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_unsupported_extension(self, tmp_path):
        from src.tools.csv_tool import csv_reader

        filepath = tmp_path / "test.json"
        filepath.write_text("{}")
        result = await csv_reader(str(filepath))
        assert "Unsupported" in result

    @pytest.mark.asyncio
    async def test_empty_input(self):
        from src.tools.csv_tool import csv_reader

        result = await csv_reader("")
        assert "Error" in result


class TestCsvReaderSchema:
    """Pydantic args_schema validation at the LangChain boundary."""

    def test_missing_path_rejected(self):
        from src.tools.csv_tool import CsvReaderInput
        with pytest.raises(ValidationError):
            CsvReaderInput()

    def test_invalid_agg_rejected(self):
        from src.tools.csv_tool import CsvReaderInput
        with pytest.raises(ValidationError):
            CsvReaderInput(path="data.csv", agg="median")

    def test_head_out_of_range_rejected(self):
        from src.tools.csv_tool import CsvReaderInput
        with pytest.raises(ValidationError):
            CsvReaderInput(path="data.csv", head=-1)
        with pytest.raises(ValidationError):
            CsvReaderInput(path="data.csv", head=200)

    def test_valid_input_parses(self):
        from src.tools.csv_tool import CsvReaderInput
        parsed = CsvReaderInput(
            path="data.csv", head=10, describe=False,
            filter_column="x", filter_value="y",
            groupby="cat", agg="mean", agg_column="val",
        )
        assert parsed.head == 10
        assert parsed.agg == "mean"


class TestCsvTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        from src.tools.csv_tool import csv_tool, CsvReaderInput
        assert csv_tool.name == "csv_reader"
        assert csv_tool.args_schema is CsvReaderInput
