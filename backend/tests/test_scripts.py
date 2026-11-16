"""Structural checks on the CLI scripts.

These exist because of a real bug: renaming `--ragas` to `--judge` left one
`args.ragas` reference behind, and nothing caught it until a 40-minute
evaluation crashed at stage 2 — after the expensive part had already run.
A unit test cannot run these scripts (they need Azure and Qdrant), but it can
check that every `args.X` they reference is actually an argument the parser
defines, which is exactly the class of mistake a rename produces.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPTS = sorted(p for p in SCRIPTS_DIR.glob("*.py") if not p.name.startswith("_"))


def _parser_dests(tree: ast.AST) -> set[str]:
    """Every attribute argparse will put on the Namespace."""
    dests: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        explicit = next((kw.value.value for kw in node.keywords
                         if kw.arg == "dest" and isinstance(kw.value, ast.Constant)), None)
        if explicit:
            dests.add(explicit)
            continue
        # Otherwise argparse derives the dest from the first long option, or the positional name.
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        long_flags = [f for f in flags if f.startswith("--")]
        name = (long_flags[0] if long_flags else flags[0] if flags else None)
        if name:
            dests.add(name.lstrip("-").replace("-", "_"))
    return dests


def _args_references(source: str) -> set[str]:
    return set(re.findall(r"\bargs\.([a-zA-Z_][a-zA-Z0-9_]*)", source))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_args_attribute_is_a_defined_argument(script: Path):
    source = script.read_text(encoding="utf-8")
    referenced = _args_references(source)
    if not referenced:
        pytest.skip(f"{script.name} takes no arguments")
    defined = _parser_dests(ast.parse(source))
    missing = referenced - defined
    assert not missing, (
        f"{script.name} reads args.{', args.'.join(sorted(missing))} but the parser never defines "
        f"{'it' if len(missing) == 1 else 'them'}. Defined: {sorted(defined)}"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_parse_as_valid_python(script: Path):
    ast.parse(script.read_text(encoding="utf-8"))


def test_evaluation_script_offers_both_judges():
    """The judge choice is the one flag whose values are referenced as string
    literals in the branching, so a typo in either place is silent."""
    source = (SCRIPTS_DIR / "run_evaluation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    choices: list[str] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and any(isinstance(a, ast.Constant) and a.value == "--judge" for a in node.args)):
            for kw in node.keywords:
                if kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    choices = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
    assert set(choices) == {"ragas", "builtin"}
    for choice in choices:
        assert f'"{choice}"' in source, f"--judge accepts {choice!r} but nothing branches on it"
