#!/usr/bin/env bash
# Find every test file that touches a given source module.
#
# Usage:
#   scripts/find_covering_tests.sh <source-module-path>
#
# Why: mutation testing tools (cosmic-ray, mutmut) do not infer which
# tests exercise the module under mutation. You supply the list. If a
# covering test file is omitted, its assertions cannot kill mutants,
# and the tool will falsely report survivors.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <source-module-path>"
    echo "  e.g.  $0 src/analysis/ssm_error.py"
    exit 2
fi

MODULE="$1"
if [ ! -f "$MODULE" ]; then
    echo "error: not a file: $MODULE" >&2
    exit 1
fi

IMPORT_PATH=$(echo "$MODULE" | sed 's|/|.|g; s|\.py$||')
BARE_NAME=$(basename "$MODULE" .py)

echo "# Module:      $MODULE"
echo "# Import path: $IMPORT_PATH"
echo "# Bare name:   $BARE_NAME"
echo

direct=$(grep -rl "$IMPORT_PATH" tests/ --include="*.py" 2>/dev/null || true)
indirect=$(grep -rl "\\b${BARE_NAME}\\b" tests/ --include="*.py" 2>/dev/null || true)

covering=$(printf "%s\\n%s\\n" "$direct" "$indirect" | grep -v '^$' | sort -u)

if [ -z "$covering" ]; then
    echo "# No covering tests found." >&2
    exit 1
fi

count=$(echo "$covering" | wc -l | tr -d ' ')
echo "# Covering tests ($count):"
echo "$covering" | while read -r f; do echo "#   $f"; done
echo
echo "# Suggested mutation test command:"
echo -n "python -m pytest"
echo "$covering" | while read -r f; do echo -n " $f"; done
echo " -x -q -o addopts= -p no:cacheprovider --timeout=30"
