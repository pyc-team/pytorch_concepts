"""Pytest reporting helpers for diagnostic test files."""

from __future__ import annotations

import sys


_GRAPH_GENERATOR_RESULTS = []
_PYTEST_CONFIG = None


def _status_label(ok: bool) -> str:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "utf" in encoding:
        return "🟢 PASSED" if ok else "🔴 NOT PASSED"
    color = "\033[32m" if ok else "\033[31m"
    label = "[PASSED]" if ok else "[NOT PASSED]"
    return f"{color}{label}\033[0m" if sys.stdout.isatty() else label


def pytest_configure(config):
    global _PYTEST_CONFIG
    _PYTEST_CONFIG = config


def pytest_runtest_logreport(report):
    if "test_graph_generator.py" not in report.nodeid or report.when != "call":
        return

    ok = report.passed
    _GRAPH_GENERATOR_RESULTS.append(
        {
            "nodeid": report.nodeid,
            "name": report.nodeid.split("::", 1)[-1],
            "outcome": report.outcome,
            "duration": report.duration,
            "ok": ok,
        }
    )
    terminal = (
        _PYTEST_CONFIG.pluginmanager.get_plugin("terminalreporter")
        if _PYTEST_CONFIG is not None
        else None
    )
    if terminal is not None:
        terminal.write_line(
            f"{_status_label(ok)} | {report.nodeid} | "
            f"duration={report.duration:.3f}s"
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _GRAPH_GENERATOR_RESULTS:
        return

    passed = [result for result in _GRAPH_GENERATOR_RESULTS if result["ok"]]
    failed = [
        result
        for result in _GRAPH_GENERATOR_RESULTS
        if result["outcome"] == "failed"
    ]
    skipped = [
        result
        for result in _GRAPH_GENERATOR_RESULTS
        if result["outcome"] == "skipped"
    ]

    terminalreporter.write_line("")
    terminalreporter.write_line("GRAPH GENERATOR SUMMARY")
    terminalreporter.write_line(f"Total checks: {len(_GRAPH_GENERATOR_RESULTS)}")
    terminalreporter.write_line(f"PASSED: {len(passed)}")
    terminalreporter.write_line(f"NOT PASSED: {len(failed)}")
    if skipped:
        terminalreporter.write_line(f"SKIPPED: {len(skipped)}")

    if failed:
        terminalreporter.write_line("")
        terminalreporter.write_line("NOT PASSED details:")
        for result in failed:
            terminalreporter.write_line(f"  - {result['nodeid']}")
    else:
        terminalreporter.write_line("")
        terminalreporter.write_line("NOT PASSED details: none")
