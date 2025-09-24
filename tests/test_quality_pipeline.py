import pytest
import pandas as pd
import io
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mcp_server import QualityPipeline


class TestQualityPipeline:
    """Test cases for QualityPipeline class"""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 28, 40],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['HR', 'IT', 'HR', 'IT', 'Finance']
        })

    @pytest.mark.unit
    def test_run_all_checks_basic(self, sample_df):
        """Test running all checks with basic configuration"""
        pipeline = QualityPipeline(sample_df)
        result = pipeline.run_all_checks()

        assert 'timestamp' in result
        assert 'file_format' in result
        assert 'summary_stats' in result
        assert 'checks' in result
        assert 'overall_passed' in result
        assert 'total_issues' in result

    @pytest.mark.unit
    def test_run_all_checks_with_schema(self, sample_df):
        """Test running checks with schema validation"""
        schema = {
            'id': 'int',
            'name': 'str',
            'age': 'int',
            'salary': 'int',
            'department': 'str'
        }

        pipeline = QualityPipeline(sample_df, schema=schema)
        result = pipeline.run_all_checks()

        assert 'checks' in result
        assert 'data_types' in result['checks']
        assert result['checks']['data_types']['passed'] is True

    @pytest.mark.unit
    def test_run_all_checks_with_rules(self, sample_df):
        """Test running checks with value range rules"""
        rules = {
            'age': {'min': 20, 'max': 50},
            'department': {'allowed': ['HR', 'IT', 'Finance', 'Sales']}
        }

        pipeline = QualityPipeline(sample_df, rules=rules)
        result = pipeline.run_all_checks()

        assert 'checks' in result
        assert 'value_ranges' in result['checks']
        assert result['checks']['value_ranges']['passed'] is True

    @pytest.mark.unit
    def test_run_all_checks_with_failures(self, sample_df):
        """Test running checks with expected failures"""
        schema = {
            'age': 'str'  # Wrong type, age is int
        }
        rules = {
            'salary': {'max': 10000}  # Too low, all salaries exceed this
        }

        pipeline = QualityPipeline(sample_df, schema=schema, rules=rules)
        result = pipeline.run_all_checks(min_rows=10)  # Also fail row count

        assert result['overall_passed'] is False
        assert result['total_issues'] > 0
        assert result['checks']['row_count']['passed'] is False
        assert result['checks']['data_types']['passed'] is False
        assert result['checks']['value_ranges']['passed'] is False

    @pytest.mark.unit
    def test_run_all_checks_empty_dataframe(self):
        """Test running checks on empty DataFrame"""
        empty_df = pd.DataFrame()
        pipeline = QualityPipeline(empty_df)
        result = pipeline.run_all_checks()

        assert result['overall_passed'] is False
        assert result['checks']['row_count']['passed'] is False
        assert result['summary_stats']['statistics']['total_rows'] == 0

    @pytest.mark.unit
    def test_run_all_checks_file_format(self, sample_df):
        """Test file format is included in result"""
        pipeline = QualityPipeline(sample_df)
        result = pipeline.run_all_checks(file_format='json')

        assert result['file_format'] == 'json'

    @pytest.mark.unit
    def test_pipeline_with_duplicates(self):
        """Test pipeline detects duplicate rows"""
        df_with_dupes = pd.DataFrame({
            'id': [1, 1, 2, 3, 3],
            'value': [10, 10, 20, 30, 30]
        })

        pipeline = QualityPipeline(df_with_dupes)
        result = pipeline.run_all_checks()

        stats = result['summary_stats']['statistics']
        assert stats['duplicate_rows'] == 2

    @pytest.mark.unit
    def test_pipeline_with_missing_values(self, sample_df):
        """Test pipeline handles missing values"""
        sample_df.loc[2, 'salary'] = None
        sample_df.loc[3, 'name'] = None

        pipeline = QualityPipeline(sample_df)
        result = pipeline.run_all_checks()

        stats = result['summary_stats']['statistics']
        assert stats['missing_values']['salary'] == 1
        assert stats['missing_values']['name'] == 1

    @pytest.mark.integration
    def test_pipeline_comprehensive(self):
        """Comprehensive test with various data issues"""
        # Create DataFrame with various issues
        df = pd.DataFrame({
            'id': [1, 2, 2, 4, None],  # Duplicate and missing
            'name': ['Alice', 'Bob', 123, 'David', 'Eve'],  # Mixed types
            'age': [25, 150, 35, -5, 40],  # Out of range values
            'email': ['alice@example.com', 'invalid', None, 'david@test.com', 'eve@example.com']
        })

        schema = {
            'id': 'int',
            'name': 'str',
            'age': 'int',
            'email': 'str'
        }

        rules = {
            'age': {'min': 0, 'max': 120}
        }

        pipeline = QualityPipeline(df, schema=schema, rules=rules)
        result = pipeline.run_all_checks()

        # Should have multiple issues
        assert result['overall_passed'] is False
        assert result['total_issues'] > 0

        # Check specific issues
        assert result['checks']['data_types']['passed'] is False  # Mixed types in name
        assert result['checks']['value_ranges']['passed'] is False  # Age out of range

        # Check statistics
        stats = result['summary_stats']['statistics']
        assert stats['duplicate_rows'] == 1  # One duplicate row
        assert stats['missing_values']['id'] == 1  # One missing ID