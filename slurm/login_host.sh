# Sourced by the job scripts: set C3_LOGIN_HOST, the login node that jobs rsync
# /share1 through (it is not mounted on compute nodes).
#
# The old login node `ada` stopped accepting SSH in October 2026 (its banner
# points at the gateways ada-gw1 / ada-gw2), and a job that cannot reach /share1
# cannot fetch its environment, stage a model or persist a single trace part.
# So the host is not hard-coded: C3_LOGIN_HOST wins if set, otherwise the first
# candidate that accepts a non-interactive login is used.
if [ -z "${C3_LOGIN_HOST:-}" ]; then
    for _c3_h in ${C3_LOGIN_CANDIDATES:-ada-gw1 ada-gw2 ada}; do
        if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
                "$USER@$_c3_h" true </dev/null >/dev/null 2>&1; then
            C3_LOGIN_HOST="$_c3_h"
            break
        fi
    done
    unset _c3_h
fi
if [ -z "${C3_LOGIN_HOST:-}" ]; then
    echo "[login] WARNING: no login host accepted ssh (tried ${C3_LOGIN_CANDIDATES:-ada-gw1 ada-gw2 ada}); /share1 is unreachable"
    C3_LOGIN_HOST="${C3_LOGIN_CANDIDATES%% *}"
    C3_LOGIN_HOST="${C3_LOGIN_HOST:-ada-gw1}"
fi
export C3_LOGIN_HOST
