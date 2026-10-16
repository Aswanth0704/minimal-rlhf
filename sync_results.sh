#!/usr/bin/env bash
# Pull checkpoints and plots produced on a remote machine back into ./out/.
# Usage:  REMOTE=myhost:/path/to/minimal-rlhf ./sync_results.sh
set -euo pipefail
cd "$(dirname "$0")"
REMOTE="${REMOTE:-gilbreth:/scratch/gilbreth/$USER/minimal-rlhf}"
rsync -avz --progress "$REMOTE/out/" ./out/
