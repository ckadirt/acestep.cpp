#!/bin/bash
# check-no-fatal.sh: fail if engine code can kill the host process.
#
# Everything under src/ is compiled into the shared library the node loads.
# A single exit(1) or abort() there takes down a long-lived host process on a
# condition that should have been an error return: an OOM allocating a VAE
# tile, a truncated GGUF, an unavailable backend. Part 5 of the engine plan
# removed all 26 of them; this check is what keeps them gone.
#
# GGML_ASSERT is in the same family: it aborts inside ggml. Engine code must
# not introduce new ones (ggml's own internal uses are out of scope here).
#
# Run from the repo root. Exits non-zero on any hit.

set -u
cd "$(dirname "$0")/.." || exit 2

status=0

# error.h documents the policy in prose and legitimately names exit(1).
hits=$(grep -rn 'exit(1)\|exit(EXIT_FAILURE)\|abort()\|GGML_ASSERT' src/ \
       | grep -v '^src/error.h:' \
       | grep -v '^\s*//' \
       | grep -v ':\s*//')

if [ -n "$hits" ]; then
    echo "FAIL: fatal exit reachable from library code."
    echo "      Return a failure and record it with ace_set_error() instead."
    echo
    echo "$hits"
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "OK: no fatal exits under src/"
fi
exit $status
