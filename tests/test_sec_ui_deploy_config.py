"""ADP-81 / ADP-73 / ADP-86 deployment-file parts for the UI: XSRF/CORS on by
default, upload limit set, loopback-only published ports, no whole-.env
pass-through, and no outside services for the About page diagrams."""
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _entrypoint_flags(env, tmp_path):
    """Run entrypoint.sh up to the `exec streamlit` line with fakes and return
    the streamlit command line it would run."""
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "python").write_text("#!/bin/sh\nexit 0\n")
    (fake / "streamlit").write_text("#!/bin/sh\necho \"$@\"\n")
    for f in fake.iterdir():
        f.chmod(0o755)
    base = {"PATH": f"{fake}:/usr/bin:/bin"}
    proc = subprocess.run(["bash", str(ROOT / "entrypoint.sh")], env={**base, **env},
                          capture_output=True, text=True, timeout=30, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip(), proc.stderr


def test_xsrf_and_cors_protection_are_on_by_default(tmp_path):
    out, _ = _entrypoint_flags({}, tmp_path)
    assert "--server.enableXsrfProtection=true" in out and "--server.enableCORS=true" in out


def test_a_deployed_container_cannot_turn_the_protection_off(tmp_path):
    for env in ({"CONTAINER_APP_NAME": "ui"}, {"APP_ENV": "prod"}):
        out, err = _entrypoint_flags({"STREAMLIT_INSECURE_LOCAL_ONLY": "1", **env}, tmp_path)
        assert "enableXsrfProtection=true" in out and "enableCORS=true" in out
        assert "ignored" in err


def test_the_local_only_override_is_explicit_and_loud(tmp_path):
    out, err = _entrypoint_flags({"STREAMLIT_INSECURE_LOCAL_ONLY": "1", "APP_ENV": "dev"}, tmp_path)
    assert "enableXsrfProtection=false" in out and "enableCORS=false" in out
    assert "OFF" in err


def test_the_old_secure_flag_is_not_needed_and_does_not_disable_anything(tmp_path):
    out, _ = _entrypoint_flags({"STREAMLIT_SECURE": "0"}, tmp_path)
    assert "enableXsrfProtection=true" in out


def test_streamlit_upload_limit_defaults_to_25_mb_and_is_configurable(tmp_path):
    assert "--server.maxUploadSize=25" in _entrypoint_flags({}, tmp_path)[0]
    assert "--server.maxUploadSize=10" in _entrypoint_flags({"UI_MAX_UPLOAD_MB": "10"}, tmp_path)[0]
    for bad in ("abc", "0", "-3", "1.5"):
        assert "--server.maxUploadSize=25" in _entrypoint_flags({"UI_MAX_UPLOAD_MB": bad}, tmp_path)[0]


def test_streamlit_config_file_has_protection_on_and_a_small_upload_limit():
    import tomllib
    server = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())["server"]
    assert server["enableXsrfProtection"] is True and server["enableCORS"] is True
    assert 0 < server["maxUploadSize"] <= 100


def test_run_app_sh_does_not_disable_protection_and_binds_to_loopback_by_default():
    text = (ROOT / "run_app.sh").read_text()
    assert "enableXsrfProtection=false" not in text and "enableCORS=false" not in text
    assert "127.0.0.1" in text


@pytest.mark.parametrize("name", ["docker-compose.yml", "docker-compose.fullstack.yml",
                                  "docker-compose.api.yml", "docker-compose.dev.yml"])
def test_published_ports_are_bound_to_loopback(name):
    spec = yaml.safe_load((ROOT / name).read_text())
    for svc, cfg in spec.get("services", {}).items():
        for port in cfg.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name}:{svc} publishes {port}"


def test_the_ui_container_gets_an_explicit_variable_list_not_the_whole_env_file():
    spec = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    ui = spec["services"]["data-analyzer"]
    assert "env_file" not in ui
    names = {e.split("=", 1)[0] for e in ui["environment"]}
    assert {"DATA_ANALYZER_API_URL", "DATA_ANALYZER_API_KEY", "APP_ENV"} <= names
    assert "DATA_ANALYZER_ADMIN_PASSWORD" not in names


def test_fullstack_rig_refuses_to_start_with_a_blank_api_key():
    text = (ROOT / "docker-compose.fullstack.yml").read_text()
    assert text.count("${DATA_ANALYZER_API_KEY:?") == 2
    assert "${DATA_ANALYZER_API_KEY}" not in text


def test_legacy_nginx_8003_listener_is_loopback_only():
    assert re.search(r"listen\s+127\.0\.0\.1:8003;", (ROOT / "nginx-prefix-fix.conf").read_text())


# --- About page diagrams: no outside services ---------------------------------
def test_diagram_renderer_contains_no_external_urls_or_scripts():
    text = (ROOT / "mermaid_renderer.py").read_text()
    assert "http://" not in text and "https://" not in text
    assert "components.html" not in text and "<script" not in text
    assert "mermaid.ink" not in text and "jsdelivr" not in text


def test_diagram_renderer_shows_the_local_image_then_the_source_as_text(monkeypatch):
    import mermaid_renderer
    shown = []
    monkeypatch.setattr(mermaid_renderer.st, "image", lambda *a, **k: shown.append(("image", a[0])))
    monkeypatch.setattr(mermaid_renderer.st, "text", lambda *a, **k: shown.append(("text", a[0])))
    monkeypatch.setattr(mermaid_renderer.st, "caption", lambda *a, **k: shown.append(("caption", a[0])))

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(mermaid_renderer.st, "expander", lambda *a, **k: _Ctx())

    png = ROOT / "assets/data_flow_simple.png"
    assert png.is_file()
    mermaid_renderer.render_mermaid("graph LR; A-->B", image_path=png)
    assert shown[0][0] == "image"
    shown.clear()
    mermaid_renderer.render_mermaid("graph LR; A-->B", image_path=ROOT / "assets/missing.png")
    assert ("text", "graph LR; A-->B") in shown and not any(k == "image" for k, _ in shown)


def test_about_page_diagram_png_files_exist_for_both_views():
    for name in ("data_flow_simple.png", "data_flow_diagram.png"):
        assert (ROOT / "assets" / name).stat().st_size > 1000


def test_inspect_cache_script_never_loads_pickle_files():
    text = (ROOT / "scripts/inspect_cache.py").read_text()
    assert "import pickle" not in text and "pickle.load" not in text
    assert ".pkl" not in text.replace("*.pkl` left", "")


def test_rendered_about_page_loads_nothing_from_outside_in_either_diagram_view():
    """Render the real About page and check what the browser would be told to load."""
    import re
    from streamlit.testing.v1 import AppTest
    for view in ("Simple View", "Detailed View"):
        at = AppTest.from_file(str(ROOT / "web_app.py"), default_timeout=90)
        at.session_state["nav_choice"] = "About"
        at.run()
        assert not at.exception
        if view != "Simple View":
            at.radio[0].set_value(view).run()
            assert not at.exception
        imgs = at.get("image")
        assert len(imgs) == 1, f"{view}: expected the pre-built diagram image, got {len(imgs)}"
        for url in (i.url for img in imgs for i in img.proto.imgs):
            assert not re.match(r"(?i)(https?:)?//", url), url   # served by the app itself
        kinds = {e.type for e in at.get("iframe") + at.get("html") + at.get("component_instance")}
        assert not kinds, kinds
        page_text = " ".join(str(getattr(e, "value", "")) for e in at.markdown)
        assert not re.search(r"(?i)<\s*(script|iframe|img|link)\b|!\[[^\]]*\]\(\s*https?:", page_text)


# --- ADP-90: selected top-bar tab label must be readable (WCAG AA, 4.5:1) -------
def _contrast(fg_hex, bg_hex):
    def lum(h):
        h = h.lstrip("#")
        ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    hi, lo = sorted((lum(fg_hex), lum(bg_hex)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_top_bar_tab_label_colours_meet_wcag_aa_contrast():
    import re
    css = (ROOT / "web_app.py").read_text()
    bar_bg = re.search(r"\.st-key-navbar \{[^}]*background-color:\s*(#[0-9a-f]{6})", css).group(1)

    def label_colour(selector_tail):
        m = re.search(r"\[data-testid=\"stButtonGroup\"\] button" + re.escape(selector_tail)
                      + r" p,[^{]*\{\s*color:\s*(#[0-9a-f]{6}) !important", css)
        assert m, f"no explicit label colour rule for {selector_tail!r}"
        return m.group(1)

    selected_bg = re.search(r"button\[aria-checked=\"true\"\] \{\s*background-color:\s*(#[0-9a-f]{6}) !important", css).group(1)
    assert _contrast(label_colour('[aria-checked="true"]'), selected_bg) >= 4.5
    assert _contrast(label_colour(""), bar_bg) >= 4.5
