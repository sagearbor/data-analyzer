"""Standalone PDF text extraction worker, run in a SEPARATE PROCESS by web_app.py.

Reads the PDF bytes from stdin and writes ONE JSON object to stdout:
  {"status": "ok", "pages": N, "text": "..."}
  {"status": "too_many_pages", "pages": N}
  {"status": "too_much_text", "pages": N}
  {"status": "error", "reason": "memory" | "unreadable"}
Stops as soon as the page limit or the accumulated character limit is exceeded,
so its output is bounded. Usage: pdf_extract_worker.py MAX_PAGES MAX_CHARS MAX_MEMORY_MB

Must stay standalone: it imports only the standard library and pypdf (never
web_app, streamlit or any other app module), and runs with a minimal environment.
"""
import io
import json
import sys


def apply_memory_limit(max_memory_mb: int) -> bool:
    """Best-effort address-space ceiling for THIS process (baseline + max_memory_mb).

    Enforced on Linux (the container). On macOS RLIMIT_AS is unsupported or not
    enforced, and on other platforms `resource` may be missing: in every such case
    this returns False and the caller carries on (the wall-clock timeout in the
    parent is then the remaining protection). Never raises.
    """
    try:
        import resource

        baseline = 0
        try:
            with open("/proc/self/statm") as f:  # pages of virtual size (Linux)
                baseline = int(f.read().split()[0]) * resource.getpagesize()
        except (OSError, ValueError, IndexError):
            pass
        limit = baseline + max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return True
    except Exception:
        return False


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()


def main(argv) -> int:
    try:
        max_pages, max_chars, max_mem_mb = (int(a) for a in argv[1:4])
        import pypdf  # imported BEFORE the memory ceiling so the baseline is accounted for

        data = sys.stdin.buffer.read()
        apply_memory_limit(max_mem_mb)
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = len(reader.pages)
        if pages > max_pages:
            _emit({"status": "too_many_pages", "pages": pages})
            return 0
        parts, total = [], 0
        for page in reader.pages:
            text = (page.extract_text() or "") + "\n"
            total += len(text)
            if total > max_chars:
                _emit({"status": "too_much_text", "pages": pages})
                return 0
            parts.append(text)
        _emit({"status": "ok", "pages": pages, "text": "".join(parts)})
        return 0
    except MemoryError:
        _emit({"status": "error", "reason": "memory"})
        return 0
    except Exception:
        _emit({"status": "error", "reason": "unreadable"})
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
