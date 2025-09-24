"""
Unit tests for DataLoader class
"""
import pytest
import pandas as pd
import io
import sys
import os
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import DataLoader


class TestDataLoader:
    """Test DataLoader class functionality"""

    @pytest.mark.unit
    def test_detect_encoding_utf8(self, create_test_csv):
        """Test UTF-8 encoding detection"""
        encoding = DataLoader.detect_encoding(create_test_csv)
        assert encoding in ['utf-8', 'ascii']  # ASCII is subset of UTF-8

    @pytest.mark.unit
    def test_detect_encoding_with_special_chars(self, temp_dir):
        """Test encoding detection with special characters"""
        # Create file with special characters
        test_file = os.path.join(temp_dir, "special_chars.csv")
        content = "name,city\nJosé,São Paulo\nFrançois,Montréal"
        with open(test_file, 'w', encoding='latin-1') as f:
            f.write(content)

        encoding = DataLoader.detect_encoding(test_file)
        assert encoding in ['latin-1', 'iso-8859-1', 'windows-1252']

    @pytest.mark.unit
    def test_load_csv_from_file(self, create_test_csv):
        """Test loading CSV from file path"""
        df = DataLoader.load_csv(create_test_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10  # sample data has 10 rows
        assert list(df.columns) == ['id', 'name', 'age', 'department', 'salary', 'hire_date', 'is_active']

    @pytest.mark.unit
    def test_load_csv_from_string(self, sample_csv_data):
        """Test loading CSV from string content"""
        df = DataLoader.load_csv(sample_csv_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    @pytest.mark.unit
    def test_load_csv_from_stringio(self, sample_csv_data):
        """Test loading CSV from StringIO object"""
        string_io = io.StringIO(sample_csv_data)
        df = DataLoader.load_csv(string_io)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    @pytest.mark.unit
    def test_load_csv_with_missing_values(self):
        """Test CSV loading handles missing values correctly"""
        csv_content = "a,b,c\n1,2,3\n4,,6\n,8,9"
        df = DataLoader.load_csv(csv_content)
        assert df.isna().sum().sum() == 2  # Two missing values

    @pytest.mark.unit
    def test_load_csv_empty_file(self):
        """Test handling of empty CSV"""
        csv_content = ""
        df = DataLoader.load_csv(csv_content)
        assert len(df) == 0

    @pytest.mark.unit
    def test_load_csv_headers_only(self):
        """Test CSV with headers but no data"""
        csv_content = "col1,col2,col3"
        df = DataLoader.load_csv(csv_content)
        assert list(df.columns) == ['col1', 'col2', 'col3']
        assert len(df) == 0

    @pytest.mark.unit
    def test_load_data_csv_format(self, sample_csv_data):
        """Test load_data method with CSV format"""
        df = DataLoader.load_data(sample_csv_data, file_format="csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    @pytest.mark.unit
    def test_load_data_unsupported_format(self):
        """Test load_data with unsupported format raises error"""
        with pytest.raises(ValueError, match="Unsupported file format"):
            DataLoader.load_data("dummy data", file_format="xml")

    @pytest.mark.unit
    def test_load_csv_different_delimiters(self, temp_dir):
        """Test CSV loading with different delimiters"""
        # Test semicolon delimiter
        test_file = os.path.join(temp_dir, "semicolon.csv")
        content = "name;age;city\nAlice;30;Boston\nBob;25;NYC"
        with open(test_file, 'w') as f:
            f.write(content)

        df = DataLoader.load_csv(test_file, sep=';')
        assert len(df) == 2
        assert list(df.columns) == ['name', 'age', 'city']

    @pytest.mark.unit
    def test_load_csv_with_quotes(self):
        """Test CSV with quoted fields"""
        csv_content = '''name,description,value
"Smith, John","A person with, commas",100
"Doe, Jane","Another ""quoted"" field",200'''
        df = DataLoader.load_csv(csv_content)
        assert len(df) == 2
        assert df.iloc[0]['name'] == 'Smith, John'

    @pytest.mark.unit
    def test_load_csv_with_different_types(self):
        """Test CSV with mixed data types"""
        csv_content = """int_col,float_col,str_col,bool_col,date_col
1,1.5,text,true,2023-01-01
2,2.5,more,false,2023-01-02
3,3.5,data,1,2023-01-03
4,4.5,here,0,2023-01-04"""
        df = DataLoader.load_csv(csv_content)
        assert len(df) == 4
        # Check that columns are loaded (type inference happens elsewhere)
        assert 'int_col' in df.columns
        assert 'float_col' in df.columns
        assert 'bool_col' in df.columns