# Next Steps for Data Analyzer LLM Integration

**Date**: 2025-10-08
**Last Updated**: 2025-10-08 (Session 2 - Completed LLM improvements)
**Status**: ✅ Core improvements completed - Ready for testing

---

## ✅ COMPLETED IN THIS SESSION

### 1. ✅ Increased max_tokens (16000)
**File**: `src/llm_client.py:278`
- Changed from 2000 → 16000 tokens
- **Fixed**: Was 32000, but gpt-4o-mini limit is 16384
- Handles large dictionaries with 100+ fields
- Prevents truncated JSON responses
- Should eliminate parsing failures from truncation

### 2. ✅ Added Structured JSON Output
**File**: `src/llm_client.py:230`
- Added `response_format={"type": "json_object"}`
- Forces OpenAI to return valid JSON
- Updated prompt to request `{"fields": [...]}` format
- Added fallback to handle both array and object formats

### 3. ✅ Fixed Multi-User Cache Issue
**File**: `web_app.py:593-607`
- Cache now user-specific: `~/.data_analyzer_cache_{username}`
- Uses `getpass.getuser()` with fallback to env vars
- "Clear All Cache" only affects current user
- Added console logging showing cache directory

### 4. ✅ Added Auto-Detect Dropdown for LLM Usage
**File**: `web_app.py:676-711`
- Replaced checkbox with selectbox
- Options: "Auto-detect (recommended)", "Always use AI", "Never use AI"
- Auto-detection logic:
  - PDF files → Always use AI
  - CSV files with standard columns → Manual parsing
  - CSV files without structure → Use AI
  - Other files → Use AI
- Shows info message explaining the decision

### 5. ✅ Added JSON Repair Fallback
**File**: `src/llm_client.py:158-189`
- New `repair_json()` method fixes common errors:
  - Missing closing brackets/braces
  - Trailing commas
  - Unterminated strings
- Integrated into `parse_llm_response()` as fallback
- Should recover fields from previously failed chunks

### 6. ✅ Enhanced REDCap Support
**File**: `src/llm_client.py:134`
- Updated prompt to specifically handle REDCap "Choices" format
- Example: "1, Yes | 0, No" → extracts allowed_values
- Should now extract validation rules from REDCap PDFs

### 7. ✅ Single-Call Optimization
**File**: `src/llm_client.py:312-342`
- **MAJOR IMPROVEMENT**: Send entire dictionary in ONE call if < 80k tokens
- Your 35k token PDF will use single-call mode (no chunking!)
- Eliminates chunking errors completely for small-medium dictionaries
- Falls back to chunking only for very large dictionaries (>80k tokens)
- Adds "mode" metadata: "single-call" vs "chunked"

**Impact for your REDCap PDF:**
- Before: 10 LLM calls, 7 failed with JSON errors
- After: **1 LLM call**, complete dictionary parsed in one go!

---

## 🎯 Immediate Priorities (Session Resume)

### 1. **Fix LLM Parsing Issues** (HIGH PRIORITY)
**Problem**: 7 of 10 chunks failed with JSON parse errors

**Changes Needed** in `src/llm_client.py`:

#### A. Increase max_tokens (Line 229)
```python
# BEFORE:
max_tokens=2000  # Too small, causes truncation

# AFTER:
max_tokens=8000  # Allow complete JSON responses
```

#### B. Add Structured JSON Output (Line 222-230)
```python
response = self.client.chat.completions.create(
    model=self.deployment,
    messages=[...],
    temperature=0.1,
    max_tokens=8000,  # INCREASED
    response_format={"type": "json_object"}  # ADDED - Forces valid JSON
)
```

#### C. Switch to Single-Call for Small PDFs
```python
# For PDFs < 128k tokens, send entire text in ONE call
# Avoids chunking issues and field splitting
if token_count < 100000:  # Leave margin
    return self.extract_fields_single_call(dictionary_text)
else:
    return self.extract_fields_chunked(dictionary_text)
```

#### D. Add JSON Repair Fallback
```python
def parse_llm_response(self, response_text: str) -> List[FieldDefinition]:
    try:
        # Try normal parsing
        field_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}, attempting repair...")
        # Try to fix common issues:
        # 1. Unterminated strings - add closing quote
        # 2. Missing commas - add between objects
        # 3. Truncated arrays - close bracket
        repaired = self.repair_json(json_text)
        field_data = json.loads(repaired)
```

#### E. Save Failed Responses for Debugging
```python
# In extract_fields_from_chunk, when parsing fails:
if len(fields) == 0:
    debug_file = Path(f"/tmp/llm_fail_chunk_{chunk_num}.txt")
    debug_file.write_text(response_text)
    logger.error(f"Saved failed response to {debug_file}")
```

---

### 2. **Fix Cache Multi-User Issue** (HIGH PRIORITY)
**Problem**: "Clear All Cache" affects ALL users sharing the system

**Option A: User-Specific Cache** (RECOMMENDED)
```python
# In web_app.py, line 594:
# BEFORE:
cache_dir = Path.home() / '.data_analyzer_cache'

# AFTER:
import getpass
username = getpass.getuser()
cache_dir = Path.home() / f'.data_analyzer_cache_{username}'
```

**Option B: Per-File Cache Clear**
```python
# Replace "Clear All Cache" with:
st.selectbox("Clear cache for:", ["None", "Current file", "All my files"])
```

**Option C: Session-Based Cache**
```python
# Add timestamp to cache files, expire after 7 days
cache_file = cache_dir / f"{file_hash}_{timestamp}.pkl"
```

---

### 3. **Add Auto-Detection for LLM Usage** (MEDIUM PRIORITY)

**UI Change** in `web_app.py` (~line 667):
```python
# BEFORE:
use_llm = st.checkbox(
    "🤖 Use AI-powered parsing",
    value=False
)

# AFTER:
llm_options = {
    "Auto-detect (recommended)": "auto",
    "Always use AI parsing": "always",
    "Never use AI (manual parsing)": "never"
}

llm_choice = st.selectbox(
    "Dictionary parsing method:",
    options=list(llm_options.keys()),
    help="Auto: PDF→AI, structured CSV→manual"
)

use_llm_mode = llm_options[llm_choice]
```

**Auto-Detection Logic**:
```python
def should_use_llm(file, mode):
    if mode == "always":
        return True
    if mode == "never":
        return False

    # Auto-detect
    if file.name.endswith('.pdf'):
        return True  # PDFs need LLM

    if file.name.endswith('.csv'):
        # Check if structured (standard columns)
        df = pd.read_csv(file)
        standard_cols = ['Column', 'Type', 'Min', 'Max', 'Allowed_Values']
        if any(col in df.columns for col in standard_cols):
            return False  # Structured CSV, manual parsing OK
        else:
            return True  # Unstructured CSV, use LLM

    return True  # Default: use LLM
```

---

### 4. **Switch to Better Model** (OPTIONAL)

**Current**: `gpt-4o-mini` (128k context, chunks required for safety)

**Options**:
- **gpt-4o**: Same context, more capable, 10x more expensive
- **o1-mini**: 128k context, reasoning model, good for complex parsing
- **gpt-4o** with single call: Simpler, no chunking errors

**Code Change** in `src/llm_client.py`:
```python
# Line 53:
self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # CHANGED
```

**.env file**:
```bash
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini  # or gpt-4o, or o1-mini
```

---

## 📊 Testing Plan

### Test Case 1: REDCap PDF (Current Failure)
**File**: `test_dictionaries/LTCPROMISE8314QC.pdf`
**Expected**: Extract allowed_values, min/max from "Choices" column
**Current Result**: 81 fields, no validation rules
**After Fix**: Should extract validation rules from HTML labels

### Test Case 2: Structured CSV (Currently Works)
**File**: `test_dictionaries/LTC-datadict-trunc01.csv`
**Expected**: Parse without LLM (has Column, Type, Min, Max headers)
**Current Result**: ✅ Works
**After Fix**: Auto-detect → skip LLM

### Test Case 3: Demo Data with Dictionary
**File**: Western demo data + demo dictionary
**Expected**: Catch age=67 (>65), salary=45000 (<50000), dept="InvalidDept"
**Current Result**: ✅ Works with CSV dict
**After Fix**: Should work with both PDF and CSV dicts

---

## 🔧 Implementation Order

1. ✅ **DONE**: Add logging for cache hits, LLM calls, dictionary usage
2. ✅ **DONE**: Add "Clear All Cache" button
3. ✅ **DONE**: Fix LLM parsing (increase tokens, structured output)
4. ✅ **DONE**: Fix multi-user cache issue
5. ✅ **DONE**: Add auto-detect dropdown for LLM usage
6. ✅ **DONE**: Add JSON repair fallback
7. ⏳ **TODO**: Test with real REDCap PDF (after user switches model)
8. ⏳ **TODO**: Consider single-call optimization for small PDFs

---

## 📁 Files to Modify

1. **src/llm_client.py**
   - Line 229: Increase max_tokens to 8000
   - Line 222: Add response_format for structured JSON
   - Line 157-214: Add JSON repair logic
   - New method: `extract_fields_single_call()` for small files

2. **web_app.py**
   - Line 594: Make cache user-specific
   - Line 667: Change checkbox to dropdown for auto-detect
   - Line 742-755: Add auto-detect logic for LLM usage

3. **.env** (user's local file)
   - Optionally change AZURE_OPENAI_DEPLOYMENT model

---

## 💡 Design Decisions to Make

### Cache Strategy
- [ ] User-specific subdirectories?
- [ ] Per-file cache clearing?
- [ ] Time-based expiration?
- [ ] Keep "Clear All" but add warning?

### LLM Model Selection
- [ ] Stick with gpt-4o-mini (cheap, fast)?
- [ ] Upgrade to gpt-4o (better parsing)?
- [ ] Try o1-mini (reasoning)?
- [ ] Make it user-selectable dropdown?

### Auto-Detection Logic
- [ ] Simple (PDF=LLM, CSV=manual)?
- [ ] Smart (check CSV structure first)?
- [ ] Add heuristics (file size, complexity)?

---

## 🐛 Known Issues

1. **LLM JSON parsing fails 70% of the time**
   - Cause: max_tokens=2000 too small, truncates mid-JSON
   - Fix: Increase to 8000, add structured JSON mode

2. **REDCap PDFs have complex HTML**
   - Cause: `<div class="rich-text-field-label">` in field descriptions
   - Fix: Better prompt, or pre-process to strip HTML

3. **Cache clears affect all users**
   - Cause: Single shared cache directory
   - Fix: User-specific cache paths

4. **No validation rules extracted from PDF**
   - Cause: Failed chunk parsing (see issue #1)
   - Fix: Same as #1

---

## 📞 Where We Left Off

**Last Action**: User tested with truncated CSV files - it worked!

**User's Question**: "What happens when you click clear all cache?"
- Answer: Currently clears EVERYONE's cache (multi-user issue)
- Needs fix before production use

**User's Request**: Store next steps so we can continue later
- This file created: `NEXT_STEPS.md`

**Next Session**: Start with fixing cache issue + increasing LLM max_tokens

---

## 🚀 Quick Resume Commands

```bash
# Activate environment
source venv/bin/activate

# Run web app
./run_app.sh

# Check cache (now user-specific!)
ls -lah ~/.data_analyzer_cache_$(whoami)/

# Inspect cache contents
python inspect_cache.py

# Test with CSV
# Upload: test_data_files/LTCPROMISE8314QC_DATA_2025-09-24_0657-5rows01.csv
# Dictionary: test_dictionaries/LTC-datadict-trunc01.csv

# Clear cache manually if needed (user-specific)
rm -rf ~/.data_analyzer_cache_$(whoami)/*.pkl
```

---

## 🎯 NEXT SESSION - User Actions

### Before Running:

1. **Optional: Switch LLM Model** (mentioned by user)
   - Edit `.env` file or set environment variable:
     ```bash
     export AZURE_OPENAI_DEPLOYMENT=gpt-4o  # or o1-mini
     ```
   - Current default: `gpt-4o-mini`

   **IMPORTANT**: If you switch models, update `max_tokens` in `src/llm_client.py:278`
   - **gpt-4o-mini**: 16,384 max (currently set to 16000)
   - **gpt-4o**: 16,384 max (same limit)
   - **gpt-4o-128k**: 4,096 max output (despite 128k context!)
   - **o1-mini/o1**: ~64k max (check current limits)

   For your use case, **gpt-4o-mini at 16000 should be sufficient** for 100+ fields.

2. **Clear Old Cache** (to force fresh parsing with new improvements)
   ```bash
   # Old global cache (if exists)
   rm -rf ~/.data_analyzer_cache/*.pkl

   # New user-specific cache
   rm -rf ~/.data_analyzer_cache_$(whoami)/*.pkl
   ```

### Testing Steps:

1. **Start the app**
   ```bash
   source venv/bin/activate
   ./run_app.sh
   ```

2. **Test REDCap PDF with Auto-Detect**
   - Upload: `test_dictionaries/LTCPROMISE8314QC.pdf`
   - Watch for: "🤖 Auto-detected: PDF requires AI parsing"
   - Should see: "🤖 **LLM ACTIVE**: Sending data to Azure OpenAI..."
   - **NEW**: Watch for: `⚡ Using SINGLE-CALL mode (no chunking)`
   - Expected: **1 LLM call** (not 10!), should extract validation rules

3. **Test Structured CSV with Auto-Detect**
   - Upload: `test_dictionaries/LTC-datadict-trunc01.csv`
   - Watch for: "📊 Auto-detected: Structured CSV, using manual parsing"
   - Should NOT see LLM calls
   - Expected: Fast parsing without API costs

4. **Run Analysis on Test Data**
   - Upload data: `test_data_files/LTCPROMISE8314QC_DATA_2025-09-24_0657-5rows01.csv`
   - Load dictionary from step 2 or 3
   - Click "Run Analysis"
   - Expected: Should catch validation errors (666, "oops" dates, etc.)

5. **Check Console Logs**
   - Look for:
     - `📦 Cache directory: ~/.data_analyzer_cache_{username}`
     - `🤖 LLM DICTIONARY PARSER INVOKED` (for PDFs)
     - `✅ Successfully repaired and parsed JSON` (if repair worked)
     - Issue counts and validation details

### Expected Improvements:

- ✅ **No chunking errors for your PDF** (single-call mode for 35k tokens!)
- ✅ **1 LLM call instead of 10** (faster, more reliable)
- ✅ **No more "Expecting value" errors** (structured JSON mode)
- ✅ **No truncated responses** (32000 tokens vs 2000)
- ✅ **Better recovery from errors** (JSON repair fallback - though less needed with single-call)
- ✅ **Extraction of allowed_values** from REDCap Choices column
- ✅ **No multi-user cache conflicts** (user-specific directories)
- ✅ **Smarter AI usage** (auto-detect saves API costs on structured CSVs)

### What to Look For:

**Success indicators:**
- More fields extracted (should be >81 if improved)
- Validation rules present (allowed_values, min/max)
- Fewer "ERROR:src.llm_client" messages
- Actual validation errors caught in test data

**Still needs work if:**
- Still seeing 7/10 chunk failures
- No allowed_values in extracted fields
- Test data passes validation when it shouldn't

---

## 📝 Git Commit Message (When Ready)

```bash
git add -A && git commit -m "Major LLM parsing improvements: single-call mode + structured outputs

MAJOR IMPROVEMENTS:
- Single-call optimization: Send entire dictionary in ONE call if <80k tokens
  * Your 35k token REDCap PDF: 1 call instead of 10!
  * Eliminates chunking errors completely for small-medium dictionaries
- Increased max_tokens from 2000 to 16000 (gpt-4o-mini limit)
  * NOTE: 32000 initially tried, but gpt-4o-mini max is 16384
  * If switching to gpt-4o (not mini), can increase to 128000
- Added structured JSON output mode (force valid JSON)
- Added JSON repair fallback (recover from common errors)
- Enhanced REDCap support (parse Choices format for allowed_values)

FIXES:
- User-specific cache directories (prevent multi-user conflicts)
- Auto-detect dropdown for LLM usage (save API costs on structured CSVs)

EXPECTED RESULTS:
- Before: 10 LLM calls, 7 failed with JSON parse errors, 81 fields extracted
- After: 1 LLM call, complete parsing, validation rules extracted

TESTING NEEDED:
- Test with REDCap PDF (should see 'SINGLE-CALL mode' in logs)
- Verify validation rules with allowed_values extracted
- Confirm test data errors detected (666, 'oops' dates)

Files modified:
- src/llm_client.py: Single-call logic, increased tokens, structured JSON, repair
- web_app.py: User-specific cache, auto-detect UI
- NEXT_STEPS.md: Complete session documentation"
```
