"""
Create a test validation program in the database for testing API endpoints
"""

import uuid
from datetime import datetime
from src.program_cache import ProgramDatabase, ValidationProgram, generate_program_name, compute_dictionary_hash

# Initialize database
db = ProgramDatabase()

# Create test program
test_program = ValidationProgram(
    program_id=str(uuid.uuid4()),
    name=generate_program_name("test_dictionary.csv"),
    aliases=["test_program", "api_test_1"],
    dictionary_source="test_dictionary.csv",
    dictionary_hash=compute_dictionary_hash("test dictionary content"),
    dictionary_format="REDCap CSV",
    generated_code="""def validate_logic_rules(df):
    violations = []
    # Test rule 1: age must be between 0 and 120
    for idx, row in df.iterrows():
        if 'age' in row and (row['age'] < 0 or row['age'] > 120):
            violations.append({
                'row': idx,
                'field': 'age',
                'rule': 'age_range',
                'message': 'Age must be between 0 and 120'
            })
    return violations
""",
    schema={
        "patient_id": {"type": "int", "required": True},
        "age": {"type": "int", "required": True, "min": 0, "max": 120},
        "gender": {"type": "str", "required": True, "allowed_values": ["M", "F", "Other"]},
        "enrollment_date": {"type": "datetime", "required": True}
    },
    conditional_rules=[
        {
            "rule_id": "age_range",
            "description": "Age must be between 0 and 120",
            "condition": "age >= 0 and age <= 120",
            "severity": "error"
        },
        {
            "rule_id": "gender_values",
            "description": "Gender must be M, F, or Other",
            "condition": "gender in ['M', 'F', 'Other']",
            "severity": "error"
        }
    ],
    created_by="test_user",
    created_at=datetime.now(),
    model_used="gpt-4",
    generation_time_seconds=2.5,
    num_fields=4,
    num_basic_rules=4,
    num_logic_rules=2
)

# Save to database
success = db.save_program(test_program)

if success:
    print(f"✓ Test program created successfully")
    print(f"  Program ID: {test_program.program_id}")
    print(f"  Name: {test_program.name}")
    print(f"  Aliases: {test_program.aliases}")
    print(f"\nYou can now run the API tests!")
else:
    print("✗ Failed to create test program")
