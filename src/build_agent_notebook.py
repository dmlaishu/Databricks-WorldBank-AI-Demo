"""Regenerates the notebooks whose %%writefile cells embed source files, so they always match the repo:

    notebooks/08_agent.py  ← src/08_agent.template.py + src/worldbank_agent.py
    notebooks/12_app.py    ← src/12_app.template.py   + app/app.py, app/app.yaml, app/requirements.txt

    python src/build_agent_notebook.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def writefile_cell(target: str, source: Path) -> str:
    body = source.read_text(encoding="utf-8").rstrip("\n")
    return f"# MAGIC %%writefile {target}\n" + "\n".join(f"# MAGIC {line}".rstrip() for line in body.splitlines())


BUILDS = {
    "08_agent.py": ("08_agent.template.py", {
        "# {{AGENT_WRITEFILE_CELL}}": ("worldbank_agent.py", ROOT / "src" / "worldbank_agent.py"),
    }),
    "12_app.py": ("12_app.template.py", {
        "# {{APP_PY}}": ("app/app.py", ROOT / "app" / "app.py"),
        "# {{APP_YAML}}": ("app/app.yaml", ROOT / "app" / "app.yaml"),
        "# {{APP_REQUIREMENTS}}": ("app/requirements.txt", ROOT / "app" / "requirements.txt"),
    }),
}

for notebook, (template, placeholders) in BUILDS.items():
    out = (ROOT / "src" / template).read_text(encoding="utf-8")
    for placeholder, (target, source) in placeholders.items():
        out = out.replace(placeholder, writefile_cell(target, source))
    (ROOT / "notebooks" / notebook).write_text(out, encoding="utf-8")
    print("wrote notebooks/" + notebook)
