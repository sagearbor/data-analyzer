# Console Notes

## Browser Console Warnings

When running the Streamlit app, you may see warnings in the browser console like:
- `Unrecognized feature: 'legacy-image-formats'`
- `Unrecognized feature: 'oversized-images'`
- `Unrecognized feature: 'vr'`
- `Unrecognized feature: 'wake-lock'`
- `An iframe which has both allow-scripts and allow-same-origin...`

These warnings are **normal and expected**. They come from Streamlit's iframe sandboxing security features and do not affect the functionality of the application. These are browser security warnings related to how Streamlit embeds components and cannot be completely eliminated without compromising security.

## LLM Processing Logs

When using AI-powered dictionary parsing, you'll see console logs with the `[LLM]` prefix:
- `[LLM] Starting dictionary parsing...`
- `[LLM] Split text into X chunks for processing`
- `[LLM] Sending chunk to Azure OpenAI...`
- `[LLM] Processed chunk X/Y, extracted Z fields`
- `[LLM] Parsing complete - extracted X fields in Y seconds`

These logs help track the progress of LLM processing, especially for large PDF dictionaries.

## Development Tips

1. **Save API costs**: The AI parsing checkbox is now OFF by default. Use the small test dictionaries in `test_dictionaries/` folder for development.

2. **Test files**: Use `small_test_dict.json` or `small_test_dict.txt` for quick testing without API calls.

3. **Performance**: Large PDFs (>100 pages) may take 30-60 seconds to process with AI. The console logs will show progress.