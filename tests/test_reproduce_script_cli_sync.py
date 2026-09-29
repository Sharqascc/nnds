"""Assert that every CLI flag reproduce_pipeline.sh passes is accepted.

Regression guard for the class of bug that made the script unrunnable:
it drifted from the CLI's argparse block after #10's bug sweep, and
nothing caught it until a reviewer tried to run it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reproduce_pipeline.sh"
CLI = REPO / "src" / "pipeline" / "traffic_analyzer.py"


def _flags_in_script():
    """Every --flag the script passes on the command line."""
    text = SCRIPT.read_text()
    # match tokens like --something (word chars and dashes) not inside comments
    flags = set()
    for line in text.splitlines():
        # skip comments
        code = line.split("#", 1)[0]
        # also skip lines that are clearly descriptive
        for m in re.finditer(r"--[a-z][a-z0-9-]+", code):
            flags.add(m.group(0))
    return flags


def _flags_accepted_by_cli():
    """Every --flag declared via parser.add_argument in the CLI."""
    text = CLI.read_text()
    # find parser.add_argument("--flag", ...) or parser.add_argument(\n  "--flag"
    flags = set(re.findall(r'add_argument\(\s*"--([a-z][a-z0-9-]+)"', text))
    return {"--" + f for f in flags}


def test_every_script_flag_is_accepted_by_cli():
    script_flags = _flags_in_script()
    cli_flags = _flags_accepted_by_cli()
    # flags that appear inside the script but are shell flags, not CLI
    SHELL_ONLY = {
        "--config-file",
        "--help",
        "--version",  # not passed by this script but guardrails
    }
    missing = sorted(f for f in script_flags if f not in cli_flags and f not in SHELL_ONLY)
    assert not missing, (
        f"reproduce_pipeline.sh passes CLI flags the argparse block does not "
        f"accept: {missing}. Either add them to the CLI or remove them from "
        f"the script."
    )


def test_key_flags_are_present():
    cli_flags = _flags_accepted_by_cli()
    for f in (
        "--video",
        "--out-csv",
        "--detector",
        "--bev-config",
        "--grid-config",
        "--max-gap",
        "--max-frames",
        "--pet-threshold",
        "--device",
        "--video-source",
        "--gate-config",
        "--max-jump",
        "--no-progress",
        "--uvh-model",
        "--coco-person-model",
    ):
        assert f in cli_flags, f"{f} not in argparse block"
