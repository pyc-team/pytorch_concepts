#!/usr/bin/env python3
import argparse
import ast
import doctest
import os
import textwrap
import traceback
from pathlib import Path


def iter_docstring_nodes(tree):
    """
    Yield (expr_node, docstring_text) for all docstrings in the AST:
    - module docstring
    - class docstrings
    - function / async function docstrings
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not getattr(node, "body", None):
                continue
            first = node.body[0]
            if isinstance(first, ast.Expr):
                value = first.value
                # Python 3.8+: Constant; older: Str
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    yield first, value.value
                elif isinstance(value, ast.Str):  # pragma: no cover (older Python)
                    yield first, value.s


def run_docstring_examples(docstring, file_path, doc_start_lineno, file_globals):
    """
    Execute all doctest-style examples in a docstring.

    - docstring: the string content of the docstring
    - file_path: absolute path to the file (for clickable tracebacks)
    - doc_start_lineno: 1-based line number where the docstring literal starts in the file
    - file_globals: globals dict shared for all examples in this file
    """
    parser = doctest.DocTestParser()
    parts = parser.parse(docstring)

    # Collect only actual doctest examples
    examples = [p for p in parts if isinstance(p, doctest.Example)]
    if not examples:
        # No code examples in this docstring -> do nothing
        return

    abs_path = os.path.abspath(file_path)

    for example in examples:
        code = example.source
        if not code.strip():
            continue

        # example.lineno is the 0-based line index within the docstring
        # The docstring itself starts at doc_start_lineno in the file.
        file_start_line = doc_start_lineno + example.lineno

        # Pad with newlines so that the first line of the example appears
        # at the correct line number in the traceback.
        padded_code = "\n" * (file_start_line - 1) + code

        try:
            compiled = compile(padded_code, abs_path, "exec")
            exec(compiled, file_globals)
        except Exception:
            print("\n" + "=" * 79)
            print(f"Error while executing docstring example in: {abs_path}:{file_start_line}")
            print("-" * 79)
            print("Code that failed:\n")
            print(textwrap.indent(code.rstrip(), "    "))
            print("\nStack trace:\n")
            traceback.print_exc()
            print("=" * 79)


def process_file(path: Path):
    """
    Parse a single Python file, extract docstrings, and run doctest-style examples.
    """
    if not path.is_file() or not path.suffix == ".py":
        return

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Non-text or weird encoding; skip
        return

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # Invalid Python; skip
        return

    # Shared globals per file: examples in the same file share state
    file_globals = {
        "__name__": "__doctest__",
        "__file__": str(path.resolve()),
        "__package__": None,
    }

    for expr_node, docstring in iter_docstring_nodes(tree):
        # expr_node.lineno is the line where the string literal starts
        run_docstring_examples(
            docstring=docstring,
            file_path=str(path.resolve()),
            doc_start_lineno=expr_node.lineno,
            file_globals=file_globals,
        )


def walk_directory(root: Path):
    """
    Recursively walk a directory and process all .py files.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                process_file(Path(dirpath) / filename)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively execute doctest-style examples found in docstrings."
    )
    parser.add_argument("root", help="Root directory (or single .py file) to scan")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        parser.error(f"{root} does not exist")

    if root.is_file():
        process_file(root)
    else:
        walk_directory(root)


if __name__ == "__main__":
    main()
