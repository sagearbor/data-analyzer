import pytest
import pandas as pd
import numpy as np
import sys
import os
import warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mcp_server import QualityChecker


class TestQualityChecker:
    """Test cases for QualityChecker class"""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 28, 40],
            'salary': [50000.5, 60000.75, 55000.25, 65000.0, 70000.5],
            'hire_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
            'is_active': ['true', 'false', 'true', 'true', 'false']
        })

    @pytest.mark.unit
    def test_check_row_count_pass(self, sample_df):
        """Test row count check when it passes"""
        checker = QualityChecker(sample_df)
        result = checker.check_row_count(min_rows=3)
        assert result['passed'] is True
        assert 'Found 5 rows' in result['message']

    @pytest.mark.unit
    def test_check_row_count_fail(self, sample_df):
        """Test row count check when it fails"""
        checker = QualityChecker(sample_df)
        result = checker.check_row_count(min_rows=10)
        assert result['passed'] is False
        assert 'Expected at least 10' in result['message']

    @pytest.mark.unit
    def test_check_row_count_empty_df(self):
        """Test row count check with empty DataFrame"""
        empty_df = pd.DataFrame()
        checker = QualityChecker(empty_df)
        result = checker.check_row_count(min_rows=1)
        assert result['passed'] is False
        assert 'Found 0 rows' in result['message']

    @pytest.mark.unit
    def test_check_data_types_all_valid(self, sample_df):
        """Test data type validation when all types are valid"""
        schema = {
            'id': 'int',
            'name': 'str',
            'age': 'int',
            'salary': 'float',
            'hire_date': 'str'
        }
        checker = QualityChecker(sample_df, schema=schema)
        result = checker.check_data_types()
        assert result['passed'] is True
        assert len(result.get('invalid_types', [])) == 0

    @pytest.mark.unit
    def test_check_data_types_with_invalid(self, sample_df):
        """Test data type validation with invalid types"""
        # Add column with mixed types
        sample_df['mixed'] = ['1', 2, 'three', 4, 5]
        schema = {
            'id': 'int',
            'name': 'str',
            'age': 'int',
            'salary': 'float',
            'hire_date': 'str',
            'mixed': 'int'
        }
        checker = QualityChecker(sample_df, schema=schema)
        result = checker.check_data_types()
        assert result['passed'] is False
        assert len(result.get('invalid_types', [])) > 0
        # Check that mixed column is flagged as invalid
        invalid_cols = [item['column'] for item in result['invalid_types']]
        assert 'mixed' in invalid_cols

    @pytest.mark.unit
    def test_check_data_types_bool_validation(self, sample_df):
        """Test boolean type validation"""
        schema = {
            'is_active': 'bool'
        }
        checker = QualityChecker(sample_df[['is_active']], schema=schema)
        result = checker.check_data_types()
        # The is_active column has 'true'/'false' strings which should be valid for bool
        assert result['passed'] is True

    @pytest.mark.unit
    def test_check_data_types_datetime_validation(self, sample_df):
        """Test datetime type validation"""
        schema = {
            'hire_date': 'datetime'
        }
        checker = QualityChecker(sample_df[['hire_date']], schema=schema)
        result = checker.check_data_types()
        # The hire_date column has valid date strings
        assert result['passed'] is True

    @pytest.mark.unit
    def test_check_value_ranges_numeric_pass(self, sample_df):
        """Test value range validation for numeric columns - passing"""
        rules = {
            'age': {'min': 20, 'max': 50},
            'salary': {'min': 40000, 'max': 80000}
        }
        checker = QualityChecker(sample_df, rules=rules)
        result = checker.check_value_ranges()
        assert result['passed'] is True
        assert len(result.get('range_violations', [])) == 0

    @pytest.mark.unit
    def test_check_value_ranges_numeric_fail(self, sample_df):
        """Test value range validation for numeric columns - failing"""
        rules = {
            'age': {'min': 30, 'max': 35},  # Will fail for ages 25, 28, and 40
            'salary': {'max': 50000}  # Will fail for most salaries
        }
        checker = QualityChecker(sample_df, rules=rules)
        result = checker.check_value_ranges()
        assert result['passed'] is False
        assert len(result.get('range_violations', [])) > 0

    @pytest.mark.unit
    def test_check_value_ranges_categorical(self, sample_df):
        """Test allowed values validation for categorical columns"""
        rules = {
            'name': {'allowed': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']}
        }
        checker = QualityChecker(sample_df, rules=rules)
        result = checker.check_value_ranges()
        assert result['passed'] is True
        assert len(result.get('range_violations', [])) == 0

    @pytest.mark.unit
    def test_check_value_ranges_empty_rules(self, sample_df):
        """Test value range validation with empty rules"""
        checker = QualityChecker(sample_df, rules={})
        result = checker.check_value_ranges()
        assert result['passed'] is True
        assert 'No value range rules' in result['message']

    @pytest.mark.unit
    def test_check_value_ranges_with_nulls(self, sample_df):
        """Test value range validation with null values"""
        sample_df.loc[2, 'salary'] = None  # Add null value
        rules = {
            'salary': {'min': 40000, 'max': 80000}
        }
        checker = QualityChecker(sample_df, rules=rules)
        result = checker.check_value_ranges()
        # Should still pass for non-null values
        assert result['passed'] is True

    @pytest.mark.unit
    def test_get_summary_stats(self, sample_df):
        """Test summary statistics generation"""
        checker = QualityChecker(sample_df)
        result = checker.get_summary_stats()
        assert result['passed'] is True
        assert 'statistics' in result
        stats = result['statistics']
        assert stats['total_rows'] == 5
        assert stats['total_columns'] == 6
        assert 'numeric_columns' in stats
        assert 'categorical_columns' in stats

    @pytest.mark.unit
    def test_get_summary_stats_with_duplicates(self):
        """Test summary statistics with duplicate rows"""
        df_with_dupes = pd.DataFrame({
            'id': [1, 1, 2, 3, 3],
            'value': [10, 10, 20, 30, 30]
        })
        checker = QualityChecker(df_with_dupes)
        result = checker.get_summary_stats()
        assert result['passed'] is True
        stats = result['statistics']
        assert stats['total_rows'] == 5
        assert stats['duplicate_rows'] == 2  # Two duplicate rows

    @pytest.mark.unit
    def test_get_summary_stats_memory_usage(self, sample_df):
        """Test that memory usage is calculated"""
        checker = QualityChecker(sample_df)
        result = checker.get_summary_stats()
        assert result['passed'] is True
        stats = result['statistics']
        assert 'memory_usage_mb' in stats
        assert stats['memory_usage_mb'] > 0

    @pytest.mark.unit
    def test_auto_detect_types(self, sample_df):
        """Test automatic type detection"""
        checker = QualityChecker(sample_df)
        detected_types = checker._auto_detect_types()
        assert detected_types['id'] == 'int'
        assert detected_types['name'] == 'str'
        assert detected_types['age'] == 'int'
        assert detected_types['salary'] == 'float'
        assert detected_types['hire_date'] == 'datetime'