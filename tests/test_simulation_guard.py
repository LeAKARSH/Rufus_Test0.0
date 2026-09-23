"""Architectural guard (spec Section 7.5): the paper-trading core and the
read-only reporting layer must never touch the network or rate limiters.

The simulation/valuation/sizing/scorecard modules are pure of I/O by
construction; ``rufus.paper`` is the single allowed boundary (it fetches live
prices to fill/settle simulated positions), and ``rufus.report`` is read-only
over SQLite. These tests prove it in a fresh interpreter so imports from *any*
dependency (not just our own modules) would be caught. Banned are the HTTP
client transports this project could actually use to reach trading/news/LLM
endpoints. Stdlib ``socket``/``asyncio`` are *not* banned: they arrive
transitively via pandas/pydantic and cannot be used to make HTTP calls on
their own.
"""

import subprocess
import sys

GUARDED_MODULES = (
    "rufus.paper",
    "rufus.simulation",
    "rufus.portfolio",
    "rufus.sizing",
    "rufus.valuation",
    "rufus.report",
    "rufus.scorecard",
)

# Provider transports the guarded modules must never import.
TRANSPORT_MODULES = ("rufus.yahoo", "rufus.currents", "rufus.news", "rufus.ollama")

# Network-capable HTTP tx that must not be reachable from the guarded core.
BANNED = ("requests", "urllib.request", "http.client", "yfinance")

_PROG = """
import sys
for m in sys.argv[1:]:
    __import__(m)
loaded = set(sys.modules)
banned = [
    b
    for b in ('requests', 'urllib.request', 'http.client', 'yfinance')
    if any(m == b or m.startswith(b + '.') for m in loaded)
]
print('BANNED:' + ','.join(sorted(banned)))
"""


def test_guarded_core_has_no_network_imports():
    result = subprocess.run(
        [sys.executable, "-c", _PROG, *GUARDED_MODULES],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout.strip()
    assert output == "BANNED:", f"network modules leaked into guarded core: {output}"


def _module_imports(name: str) -> set[str]:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "rufus"
    source = (root / f"{name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_pure_modules_do_not_import_each_others_transports():
    # Only rufus.paper is the allowed boundary; every other guarded module
    # must stay free of the provider clients (and of paper itself).
    for name in ("portfolio", "sizing", "simulation", "valuation", "scorecard", "report"):
        imports = _module_imports(name)
        assert not (imports & set(TRANSPORT_MODULES)), (
            f"rufus.{name} imports provider transport {imports & set(TRANSPORT_MODULES)}"
        )
        assert "rufus.paper" not in imports, f"rufus.{name} unexpectedly imports rufus.paper"


def test_paper_is_the_only_boundary_module():
    # rufus.paper may legitimately import the Yahoo client (live fills), but
    # no OTHER guarded module is allowed to. This pins the boundary in place
    # so nobody can wire the simulation core to a provider behind paper's back.
    imports = _module_imports("paper")
    assert imports & set(TRANSPORT_MODULES) == {"rufus.yahoo"}


def test_boundary_preserved_from_sibling_imports():
    # If this ever changes, the boundary moved. Assert explicitly that the
    # guarded modules reach a provider transport ONLY via rufus.paper -> yahoo.
    root = ("rufus.yahoo", "rufus.currents", "rufus.news", "rufus.ollama")
    for name in ("portfolio", "sizing", "simulation", "valuation", "scorecard", "report"):
        imports = _module_imports(name)
        for entry in imports:
            assert not entry.startswith(root), f"rufus.{name} leaked {entry}"