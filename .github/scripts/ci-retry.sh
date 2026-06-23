#!/usr/bin/env bash
# Retry a command with exponential backoff.
#
# CI sometimes fails for reasons that have nothing to do with our code:
# transient PyPI/network hiccups when installing third-party packages
# (matplotlib, scipy, ...), apt mirror flakiness, or download timeouts.
# Wrapping those steps in this helper absorbs the transient failures so a
# blip doesn't turn a green change red.
#
# Usage:
#   bash .github/scripts/ci-retry.sh <command> [args...]
#   bash .github/scripts/ci-retry.sh bash -c "cmd1 && cmd2"   # for pipelines
#
# Tunables (environment variables):
#   CI_RETRY_ATTEMPTS  total attempts before giving up (default 5)
#   CI_RETRY_DELAY     initial backoff in seconds, doubled each retry (default 5)
set -uo pipefail

max_attempts="${CI_RETRY_ATTEMPTS:-5}"
delay="${CI_RETRY_DELAY:-5}"

if [ "$#" -eq 0 ]; then
  echo "ci-retry: no command given" >&2
  exit 2
fi

attempt=1
while true; do
  "$@" && exit 0
  status=$?
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "ci-retry: '$*' failed after ${attempt} attempt(s) (exit ${status}); giving up" >&2
    exit "$status"
  fi
  echo "ci-retry: attempt ${attempt}/${max_attempts} of '$*' failed (exit ${status}); retrying in ${delay}s..." >&2
  sleep "$delay"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
done
