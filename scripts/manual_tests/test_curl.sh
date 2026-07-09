#!/bin/bash

# Test dictionary parsing endpoint
echo "Testing POST /api/v1/dictionary/parse..."
curl -X POST "http://localhost:8000/api/v1/dictionary/parse" \
  -H "X-API-Key: test-key-12345" \
  -F "dictionary_file=@tests/test_data/dictionaries/simple_csv_dict.csv" \
  -F "save_program=true" \
  -F "program_name=TestProgram"

echo -e "\n\nTesting GET /api/v1/dictionary/{dict_id}..."
curl -X GET "http://localhost:8000/api/v1/dictionary/TestProgram" \
  -H "X-API-Key: test-key-12345"
