# Debugging Guide - Dictionary Upload Issues

## Issue: PDF Being Treated as CSV

### Symptoms:
- Upload PDF file
- See: "📊 Auto-detected: Structured CSV, using manual parsing"
- See: "✅ Parsed 0 field definitions from CSV"
- When forcing LLM: Only finds missing values, not validation errors

### Root Causes to Check:

#### 1. **Filename Has Wrong Extension**
**Check**: What's the actual filename shown in the UI?

Look for the line that says:
```
📎 Uploaded file: filename.ext (mime-type)
```

**Common Issues**:
- File named `dictionary.pdf.csv` (ends with .csv, treated as CSV)
- File has `.csv` anywhere at the end
- File is actually a CSV, not a PDF

**Fix**: Rename file to end with `.pdf` only

---

#### 2. **LLM Not Available**
**Check Terminal Logs** (not browser console!) for:
```
LLM client initialized with endpoint: https://...
```

If you see errors about missing OpenAI keys, LLM mode won't work.

**Fix**: Check `.env` file has:
```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

---

#### 3. **Cache from Old File**
**Check Terminal** for:
```
💾 LOADING FROM DISK CACHE (NO LLM CALL)
```

If you see this, it's loading a previously parsed version.

**Fix**:
```bash
# Clear cache
rm -rf ~/.data_analyzer_cache_$(whoami)/*.pkl

# Refresh browser page
```

---

### Where to Look for Logs:

#### ❌ WRONG: Browser Dev Console (F12)
The `print()` statements go to Python, not JavaScript!

#### ✅ CORRECT: Terminal Where You Ran `./run_app.sh`

You should see lines like:
```
🔍 AUTO-DETECT: Checking file 'LTCPROMISE8314QC.pdf'
   Extension check: .pdf=True, .csv=False
   ✅ Detected as PDF, will use LLM
```

Or for CSV:
```
🔍 AUTO-DETECT: Checking file 'dictionary.csv'
   Extension check: .pdf=False, .csv=True
📊 Auto-detected: Structured CSV, using manual parsing
```

---

### Debug Checklist:

1. [ ] Check terminal output (not browser console)
2. [ ] Verify filename shown in UI (📎 Uploaded file: ...)
3. [ ] Check if using cached version
4. [ ] Verify LLM is available (check .env file)
5. [ ] Clear cache and try again
6. [ ] Check if file actually ends with .pdf (not .pdf.csv)

---

### Expected Flow for PDF with LLM:

**Terminal Output:**
```
🔍 AUTO-DETECT: Checking file 'LTCPROMISE8314QC.pdf'
   Extension check: .pdf=True, .csv=False
   ✅ Detected as PDF, will use LLM

================================================================================
🤖 LLM DICTIONARY PARSER INVOKED
   File: LTCPROMISE8314QC.pdf
   Size: 257119 bytes
================================================================================

[LLM] Starting dictionary parsing - 131020 characters at 17:18:21
[LLM] Estimated tokens: 35000
[LLM] ⚡ Using SINGLE-CALL mode (no chunking) - more reliable!
[LLM] Sending chunk (131020 chars) to Azure OpenAI...
[LLM] Received response from Azure OpenAI
[LLM] Single-call parsing complete - extracted 124 fields in 45.2 seconds
```

**UI Messages:**
```
🤖 Auto-detected: PDF requires AI parsing
🤖 LLM ACTIVE: Sending data to Azure OpenAI GPT-4...
✅ AI extracted 124 field definitions from 1 chunks in 45.2 seconds
💾 Dictionary cached - future loads will be instant (no API calls)
```

---

### If Still Having Issues:

**Share This Info:**
1. Exact filename shown in UI (📎 line)
2. Terminal output from auto-detect section
3. Whether you see "LLM ACTIVE" or "CSV PARSING"
4. Contents of `.env` file (AZURE_* variables, hide keys)
5. Output of: `ls -lah ~/.data_analyzer_cache_$(whoami)/`
