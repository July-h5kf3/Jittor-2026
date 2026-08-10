#!/usr/bin/env bash

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0-beta.1
if [ ! -r "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    printf 'error: CANN environment script not found: %s\n' "$ASCEND_TOOLKIT_HOME/set_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi
if ! source "$ASCEND_TOOLKIT_HOME/set_env.sh"; then
    printf 'error: failed to source CANN environment script: %s\n' "$ASCEND_TOOLKIT_HOME/set_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi
export tikcc_path="$ASCEND_TOOLKIT_HOME/bin/ccec"
export JITTOR_HOME=/data/ldc/cache/jittor-track2-ascend
export NKAI_JITTOR_COMMIT=06f5d3d271555682c95aa3505518f47eeab2bd9c
export PATH="/data/ldc/envs/track2-ascend/bin:$PATH"
