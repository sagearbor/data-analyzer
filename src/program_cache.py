"""
SQLite Database Layer for Named Program Cache System

This module provides persistent storage for validation programs, enabling:
- Caching of generated validation code based on dictionary hash
- Named programs with user-friendly aliases
- Execution history tracking
- Admin-only soft delete functionality
"""

from __future__ import annotations

import os
import sqlite3
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationProgram:
    """
    Represents a named validation program with metadata and execution history.

    Attributes:
        program_id: Unique identifier (UUID)
        name: Auto-generated or user-specified name (e.g., "20241202-143022-ClinicalTrial")
        aliases: User-friendly names for easy reference

        dictionary_source: Original dictionary filename
        dictionary_hash: MD5 hash of dictionary content for cache lookup
        dictionary_format: Format identifier (e.g., "REDCap CSV", "FHIR JSON")

        generated_code: Python validation code as string
        schema: Field definitions extracted from dictionary
        conditional_rules: Logic rules extracted from dictionary

        created_by: Username of creator
        created_at: Timestamp of creation
        last_used: Timestamp of last execution
        use_count: Number of times program has been executed
        model_used: LLM model used for generation (e.g., "gpt-5-nano")
        generation_time_seconds: Time taken to generate code

        num_fields: Number of fields in schema
        num_basic_rules: Number of basic validation rules
        num_logic_rules: Number of conditional logic rules

        status: "active" or "deleted"
        deleted_at: Timestamp of deletion (if deleted)
        deleted_by: Username who deleted (if deleted)
        deletion_reason: Reason for deletion (if deleted)

        version: Version number (for future versioning support)
        parent_program_id: Reference to parent program (for future versioning)
    """
    program_id: str
    name: str
    aliases: List[str]

    dictionary_source: str
    dictionary_hash: str
    dictionary_format: str

    generated_code: str
    schema: Dict[str, Any]
    conditional_rules: List[Dict[str, Any]]

    created_by: str
    created_at: datetime
    last_used: Optional[datetime] = None
    use_count: int = 0
    model_used: str = "gpt-5-nano"
    generation_time_seconds: float = 0.0

    num_fields: int = 0
    num_basic_rules: int = 0
    num_logic_rules: int = 0

    status: str = "active"
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    deletion_reason: Optional[str] = None

    version: int = 1
    parent_program_id: Optional[str] = None

    def __post_init__(self):
        """Validate field values after initialization"""
        if self.status not in ("active", "deleted"):
            raise ValueError(f"Invalid status: {self.status}. Must be 'active' or 'deleted'")

        if not self.aliases:
            self.aliases = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        if self.deleted_at:
            data['deleted_at'] = self.deleted_at.isoformat()
        return data


class ProgramDatabase:
    """
    SQLite database for managing validation programs.

    Features:
    - Thread-safe with connection per operation
    - WAL mode for better concurrency
    - Parameterized queries for SQL injection prevention
    - Separate .py files for generated code
    - Schema and rules stored as JSON in database
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection and create tables if needed.

        Args:
            db_path: Path to SQLite database file. If None, uses ~/.data_analyzer/programs.db
        """
        # Default database location
        if db_path is None:
            home_dir = Path.home()
            self.data_dir = home_dir / ".data_analyzer"
            self.db_path = self.data_dir / "programs.db"
        else:
            self.db_path = Path(db_path)
            self.data_dir = self.db_path.parent

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.programs_dir = self.data_dir / "programs"
        self.programs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ProgramDatabase at {self.db_path}")
        logger.info(f"Programs directory: {self.programs_dir}")

        # Initialize database schema
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Create a new database connection with proper settings.

        Returns:
            SQLite connection with WAL mode enabled
        """
        conn = sqlite3.Connection(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name

        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")

        return conn

    def _init_database(self):
        """Create tables and indexes if they don't exist"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Programs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS programs (
                    program_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    dictionary_source TEXT,
                    dictionary_hash TEXT,
                    dictionary_format TEXT,
                    generated_code_path TEXT,
                    schema_json TEXT,
                    rules_json TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    model_used TEXT,
                    generation_time_seconds REAL,
                    num_fields INTEGER,
                    num_basic_rules INTEGER,
                    num_logic_rules INTEGER,
                    status TEXT DEFAULT 'active',
                    deleted_at TIMESTAMP,
                    deleted_by TEXT,
                    deletion_reason TEXT,
                    version INTEGER DEFAULT 1,
                    parent_program_id TEXT,
                    FOREIGN KEY (parent_program_id) REFERENCES programs(program_id)
                )
            """)

            # Aliases table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS aliases (
                    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_id TEXT NOT NULL,
                    alias TEXT NOT NULL UNIQUE,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (program_id) REFERENCES programs(program_id)
                )
            """)

            # Execution history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_id TEXT NOT NULL,
                    executed_at TIMESTAMP,
                    executed_by TEXT,
                    data_file TEXT,
                    rows_processed INTEGER,
                    field_violations_found INTEGER,
                    logic_violations_found INTEGER,
                    execution_time_seconds REAL,
                    FOREIGN KEY (program_id) REFERENCES programs(program_id)
                )
            """)

            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_program_name ON programs(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dict_hash ON programs(dictionary_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON programs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON programs(last_used DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_program_alias ON aliases(alias)")

            conn.commit()
            logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            conn.close()

    def save_program(self, program: ValidationProgram) -> bool:
        """
        Save program to database and write code to file.

        Args:
            program: ValidationProgram instance to save

        Returns:
            True if successful, False otherwise
        """
        conn = self._get_connection()
        try:
            # Write generated code to file
            code_filename = f"{program.program_id}.py"
            code_path = self.programs_dir / code_filename

            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(program.generated_code)

            logger.info(f"Wrote program code to {code_path}")

            # Insert program into database
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO programs (
                    program_id, name, dictionary_source, dictionary_hash,
                    dictionary_format, generated_code_path, schema_json, rules_json,
                    created_by, created_at, last_used, use_count, model_used,
                    generation_time_seconds, num_fields, num_basic_rules,
                    num_logic_rules, status, version, parent_program_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program.program_id,
                program.name,
                program.dictionary_source,
                program.dictionary_hash,
                program.dictionary_format,
                str(code_path),
                json.dumps(program.schema),
                json.dumps(program.conditional_rules),
                program.created_by,
                program.created_at.isoformat() if program.created_at else None,
                program.last_used.isoformat() if program.last_used else None,
                program.use_count,
                program.model_used,
                program.generation_time_seconds,
                program.num_fields,
                program.num_basic_rules,
                program.num_logic_rules,
                program.status,
                program.version,
                program.parent_program_id
            ))

            # Insert aliases
            for alias in program.aliases:
                try:
                    cursor.execute("""
                        INSERT INTO aliases (program_id, alias, created_by, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (program.program_id, alias, program.created_by, datetime.now().isoformat()))
                except sqlite3.IntegrityError:
                    logger.warning(f"Alias '{alias}' already exists, skipping")

            conn.commit()
            logger.info(f"Saved program {program.program_id} with name '{program.name}'")
            return True

        except sqlite3.IntegrityError as e:
            logger.error(f"Program with name '{program.name}' already exists: {e}")
            return False
        except Exception as e:
            logger.error(f"Error saving program: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def load_program(self, id_or_name_or_alias: str) -> Optional[ValidationProgram]:
        """
        Load program by ID, name, or alias.

        Search order:
        1. Try as program_id (UUID)
        2. Try as exact name match
        3. Try as alias

        Args:
            id_or_name_or_alias: Program ID, name, or alias to search for

        Returns:
            ValidationProgram instance if found, None otherwise
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Try as program_id first
            cursor.execute("""
                SELECT * FROM programs WHERE program_id = ? AND status = 'active'
            """, (id_or_name_or_alias,))
            row = cursor.fetchone()

            # Try as name if not found
            if row is None:
                cursor.execute("""
                    SELECT * FROM programs WHERE name = ? AND status = 'active'
                """, (id_or_name_or_alias,))
                row = cursor.fetchone()

            # Try as alias if still not found
            if row is None:
                cursor.execute("""
                    SELECT p.* FROM programs p
                    JOIN aliases a ON p.program_id = a.program_id
                    WHERE a.alias = ? AND p.status = 'active'
                """, (id_or_name_or_alias,))
                row = cursor.fetchone()

            if row is None:
                logger.info(f"Program not found: {id_or_name_or_alias}")
                return None

            # Load generated code from file
            code_path = Path(row['generated_code_path'])
            if code_path.exists():
                with open(code_path, 'r', encoding='utf-8') as f:
                    generated_code = f.read()
            else:
                logger.warning(f"Code file not found: {code_path}")
                generated_code = ""

            # Load aliases for this program
            cursor.execute("""
                SELECT alias FROM aliases WHERE program_id = ?
            """, (row['program_id'],))
            aliases = [r['alias'] for r in cursor.fetchall()]

            # Parse JSON fields
            schema = json.loads(row['schema_json']) if row['schema_json'] else {}
            rules = json.loads(row['rules_json']) if row['rules_json'] else []

            # Parse datetime fields
            created_at = datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            last_used = datetime.fromisoformat(row['last_used']) if row['last_used'] else None
            deleted_at = datetime.fromisoformat(row['deleted_at']) if row['deleted_at'] else None

            program = ValidationProgram(
                program_id=row['program_id'],
                name=row['name'],
                aliases=aliases,
                dictionary_source=row['dictionary_source'],
                dictionary_hash=row['dictionary_hash'],
                dictionary_format=row['dictionary_format'],
                generated_code=generated_code,
                schema=schema,
                conditional_rules=rules,
                created_by=row['created_by'],
                created_at=created_at,
                last_used=last_used,
                use_count=row['use_count'],
                model_used=row['model_used'],
                generation_time_seconds=row['generation_time_seconds'],
                num_fields=row['num_fields'],
                num_basic_rules=row['num_basic_rules'],
                num_logic_rules=row['num_logic_rules'],
                status=row['status'],
                deleted_at=deleted_at,
                deleted_by=row['deleted_by'],
                deletion_reason=row['deletion_reason'],
                version=row['version'],
                parent_program_id=row['parent_program_id']
            )

            logger.info(f"Loaded program {program.program_id} (name: {program.name})")
            return program

        except Exception as e:
            logger.error(f"Error loading program: {e}")
            return None
        finally:
            conn.close()

    def search_programs(
        self,
        query: Optional[str] = None,
        dictionary_source: Optional[str] = None,
        dictionary_hash: Optional[str] = None,
        status: str = "active"
    ) -> List[ValidationProgram]:
        """
        Search programs with optional filters.

        Args:
            query: Text to search in name, aliases, or dictionary_source
            dictionary_source: Exact match on dictionary source filename
            dictionary_hash: Exact match on dictionary hash (for cache lookup)
            status: Filter by status ("active" or "deleted")

        Returns:
            List of matching ValidationProgram instances
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Build WHERE clause
            where_clauses = ["p.status = ?"]
            params = [status]

            if dictionary_hash:
                where_clauses.append("p.dictionary_hash = ?")
                params.append(dictionary_hash)

            if dictionary_source:
                where_clauses.append("p.dictionary_source = ?")
                params.append(dictionary_source)

            if query:
                where_clauses.append("""(
                    p.name LIKE ? OR
                    p.dictionary_source LIKE ? OR
                    EXISTS (SELECT 1 FROM aliases a WHERE a.program_id = p.program_id AND a.alias LIKE ?)
                )""")
                search_pattern = f"%{query}%"
                params.extend([search_pattern, search_pattern, search_pattern])

            where_sql = " AND ".join(where_clauses)

            sql = f"""
                SELECT DISTINCT p.* FROM programs p
                WHERE {where_sql}
                ORDER BY p.last_used DESC NULLS LAST, p.created_at DESC
                LIMIT 100
            """

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            programs = []
            for row in rows:
                # Load each program (includes code file and aliases)
                program = self.load_program(row['program_id'])
                if program:
                    programs.append(program)

            logger.info(f"Search found {len(programs)} programs")
            return programs

        except Exception as e:
            logger.error(f"Error searching programs: {e}")
            return []
        finally:
            conn.close()

    def list_all_programs(self, status: str = "active", limit: int = 100) -> List[ValidationProgram]:
        """
        List all programs, ordered by last_used descending.

        Args:
            status: Filter by status ("active" or "deleted")
            limit: Maximum number of programs to return

        Returns:
            List of ValidationProgram instances
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM programs
                WHERE status = ?
                ORDER BY last_used DESC NULLS LAST, created_at DESC
                LIMIT ?
            """, (status, limit))

            rows = cursor.fetchall()

            programs = []
            for row in rows:
                program = self.load_program(row['program_id'])
                if program:
                    programs.append(program)

            logger.info(f"Listed {len(programs)} programs with status '{status}'")
            return programs

        except Exception as e:
            logger.error(f"Error listing programs: {e}")
            return []
        finally:
            conn.close()

    def create_alias(self, program_id: str, alias: str, created_by: str) -> bool:
        """
        Create a globally unique alias for a program.

        Args:
            program_id: Program ID to create alias for
            alias: Alias string (must be globally unique)
            created_by: Username creating the alias

        Returns:
            True if successful, False if alias already exists
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Check if program exists
            cursor.execute("SELECT program_id FROM programs WHERE program_id = ?", (program_id,))
            if cursor.fetchone() is None:
                logger.error(f"Program {program_id} does not exist")
                return False

            # Insert alias
            cursor.execute("""
                INSERT INTO aliases (program_id, alias, created_by, created_at)
                VALUES (?, ?, ?, ?)
            """, (program_id, alias, created_by, datetime.now().isoformat()))

            conn.commit()
            logger.info(f"Created alias '{alias}' for program {program_id}")
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"Alias '{alias}' already exists")
            return False
        except Exception as e:
            logger.error(f"Error creating alias: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def delete_alias(self, alias: str) -> bool:
        """
        Delete an alias.

        Args:
            alias: Alias to delete

        Returns:
            True if successful, False otherwise
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM aliases WHERE alias = ?", (alias,))

            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Deleted alias '{alias}'")
            else:
                logger.warning(f"Alias '{alias}' not found")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting alias: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def increment_use_count(self, program_id: str) -> None:
        """
        Increment use count and update last_used timestamp.

        Args:
            program_id: Program ID to update
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE programs
                SET use_count = use_count + 1,
                    last_used = ?
                WHERE program_id = ?
            """, (datetime.now().isoformat(), program_id))

            conn.commit()
            logger.info(f"Incremented use count for program {program_id}")

        except Exception as e:
            logger.error(f"Error incrementing use count: {e}")
            conn.rollback()
        finally:
            conn.close()

    def record_execution(self, program_id: str, stats: Dict[str, Any]) -> None:
        """
        Record program execution in history table.

        Args:
            program_id: Program ID that was executed
            stats: Dictionary with execution statistics:
                - executed_by: Username
                - data_file: Input data filename
                - rows_processed: Number of rows validated
                - field_violations_found: Number of field-level violations
                - logic_violations_found: Number of logic rule violations
                - execution_time_seconds: Time taken to execute
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO execution_history (
                    program_id, executed_at, executed_by, data_file,
                    rows_processed, field_violations_found,
                    logic_violations_found, execution_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program_id,
                datetime.now().isoformat(),
                stats.get('executed_by', 'unknown'),
                stats.get('data_file', ''),
                stats.get('rows_processed', 0),
                stats.get('field_violations_found', 0),
                stats.get('logic_violations_found', 0),
                stats.get('execution_time_seconds', 0.0)
            ))

            conn.commit()
            logger.info(f"Recorded execution for program {program_id}")

            # Also increment use count
            self.increment_use_count(program_id)

        except Exception as e:
            logger.error(f"Error recording execution: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _verify_admin(self, password: str) -> bool:
        """
        Verify admin password against environment variable.

        Args:
            password: Password to verify

        Returns:
            True if password matches, False otherwise
        """
        admin_password = os.getenv("ADMIN_PW", "dataAnalyzerAdmin666")
        return password == admin_password

    def delete_program(
        self,
        program_id: str,
        deleted_by: str,
        reason: str,
        admin_password: str
    ) -> bool:
        """
        Soft delete a program (admin only).

        Args:
            program_id: Program ID to delete
            deleted_by: Username performing deletion
            reason: Reason for deletion
            admin_password: Admin password for verification

        Returns:
            True if successful

        Raises:
            PermissionError: If admin password is incorrect
        """
        if not self._verify_admin(admin_password):
            raise PermissionError("Invalid admin password")

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE programs
                SET status = 'deleted',
                    deleted_at = ?,
                    deleted_by = ?,
                    deletion_reason = ?
                WHERE program_id = ?
            """, (datetime.now().isoformat(), deleted_by, reason, program_id))

            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Soft deleted program {program_id} by {deleted_by}")
            else:
                logger.warning(f"Program {program_id} not found for deletion")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting program: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def restore_program(self, program_id: str, admin_password: str) -> bool:
        """
        Restore a soft-deleted program (admin only).

        Args:
            program_id: Program ID to restore
            admin_password: Admin password for verification

        Returns:
            True if successful

        Raises:
            PermissionError: If admin password is incorrect
        """
        if not self._verify_admin(admin_password):
            raise PermissionError("Invalid admin password")

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE programs
                SET status = 'active',
                    deleted_at = NULL,
                    deleted_by = NULL,
                    deletion_reason = NULL
                WHERE program_id = ? AND status = 'deleted'
            """, (program_id,))

            restored = cursor.rowcount > 0
            conn.commit()

            if restored:
                logger.info(f"Restored program {program_id}")
            else:
                logger.warning(f"Program {program_id} not found or not deleted")

            return restored

        except Exception as e:
            logger.error(f"Error restoring program: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_execution_history(
        self,
        program_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get execution history for a program or all programs.

        Args:
            program_id: Optional program ID to filter by
            limit: Maximum number of records to return

        Returns:
            List of execution history records as dictionaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if program_id:
                cursor.execute("""
                    SELECT * FROM execution_history
                    WHERE program_id = ?
                    ORDER BY executed_at DESC
                    LIMIT ?
                """, (program_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM execution_history
                    ORDER BY executed_at DESC
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()

            history = []
            for row in rows:
                history.append({
                    'execution_id': row['execution_id'],
                    'program_id': row['program_id'],
                    'executed_at': row['executed_at'],
                    'executed_by': row['executed_by'],
                    'data_file': row['data_file'],
                    'rows_processed': row['rows_processed'],
                    'field_violations_found': row['field_violations_found'],
                    'logic_violations_found': row['logic_violations_found'],
                    'execution_time_seconds': row['execution_time_seconds']
                })

            return history

        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            return []
        finally:
            conn.close()


# Utility functions for testing and command-line usage

def compute_dictionary_hash(dictionary_content: str) -> str:
    """
    Compute MD5 hash of dictionary content for cache lookup.

    Args:
        dictionary_content: Raw dictionary text

    Returns:
        MD5 hash as hex string
    """
    return hashlib.md5(dictionary_content.encode('utf-8')).hexdigest()


def generate_program_name(dictionary_source: str) -> str:
    """
    Generate auto-name for program based on timestamp and source.

    Format: YYYYMMDD-HHMMSS-SourceName
    Example: "20241202-143022-ClinicalTrial"

    Args:
        dictionary_source: Source dictionary filename

    Returns:
        Generated program name
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Extract base name without extension
    source_name = Path(dictionary_source).stem

    # Clean name: keep only alphanumeric and hyphens
    source_name = ''.join(c for c in source_name if c.isalnum() or c == '-')

    # Truncate to reasonable length
    source_name = source_name[:30]

    return f"{timestamp}-{source_name}"


# Testing function
def test_database():
    """Test the database functionality"""
    import uuid

    print("Testing ProgramDatabase...")

    # Initialize database
    db = ProgramDatabase()

    # Create test program
    test_program = ValidationProgram(
        program_id=str(uuid.uuid4()),
        name=generate_program_name("test_dictionary.csv"),
        aliases=["test-prog", "my-validator"],
        dictionary_source="test_dictionary.csv",
        dictionary_hash=compute_dictionary_hash("test content"),
        dictionary_format="REDCap CSV",
        generated_code="def validate(data):\n    pass",
        schema={"field1": {"type": "int", "required": True}},
        conditional_rules=[{"rule": "field1 > 0"}],
        created_by="test_user",
        created_at=datetime.now(),
        model_used="gpt-5-nano",
        generation_time_seconds=1.5,
        num_fields=1,
        num_basic_rules=1,
        num_logic_rules=1
    )

    # Save program
    success = db.save_program(test_program)
    print(f"Save program: {'✓' if success else '✗'}")

    # Load by ID
    loaded = db.load_program(test_program.program_id)
    print(f"Load by ID: {'✓' if loaded else '✗'}")

    # Load by name
    loaded = db.load_program(test_program.name)
    print(f"Load by name: {'✓' if loaded else '✗'}")

    # Load by alias
    loaded = db.load_program("test-prog")
    print(f"Load by alias: {'✓' if loaded else '✗'}")

    # Search programs
    results = db.search_programs(query="test")
    print(f"Search programs: {'✓' if len(results) > 0 else '✗'}")

    # Record execution
    db.record_execution(test_program.program_id, {
        'executed_by': 'test_user',
        'data_file': 'test_data.csv',
        'rows_processed': 100,
        'field_violations_found': 5,
        'logic_violations_found': 2,
        'execution_time_seconds': 0.5
    })
    print("Record execution: ✓")

    # Get execution history
    history = db.get_execution_history(test_program.program_id)
    print(f"Get execution history: {'✓' if len(history) > 0 else '✗'}")

    print("\nTest complete!")
    print(f"Database location: {db.db_path}")
    print(f"Programs directory: {db.programs_dir}")


if __name__ == "__main__":
    test_database()
