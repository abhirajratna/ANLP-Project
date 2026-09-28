# Sourced by the GPU job scripts: stage model weights onto node-local SSD and set
# MODEL_PATH_ARG (a bash array: empty, or --model-path <dir>).
#
# Inputs: MODEL (registry key), PRESET, C3_MODEL_REMOTE (a path on the login
# node to pull), C3_MODEL_PATH (a path already on this node), C3_SCRATCH,
# SSH_OPTS, HF_HOME.
#
# LLaDA-8B sits on /share1, but which variant it holds was never recorded, so it
# is used only when it exists and its tokenizer config says it is the variant
# requested (<|eot_id|> in the vocabulary means Instruct).  Otherwise the pinned revision is
# downloaded from the hub into HF_HOME on node-local SSD, never /home2.
SHARED_LLADA="${C3_SHARED_LLADA:-/share1/ishita/llada_model}"
MODEL_PATH_ARG=()
if [ -n "${C3_MODEL_REMOTE:-}" ]; then
    LOCAL_COPY="$C3_SCRATCH/models/$(basename "$C3_MODEL_REMOTE")"
    mkdir -p "$(dirname "$LOCAL_COPY")"
    echo "[stage] pulling $C3_MODEL_REMOTE"
    rsync -a --partial-dir=.rsync-partial -e "ssh $SSH_OPTS" \
        "$USER@ada:$C3_MODEL_REMOTE" "$LOCAL_COPY" \
        || { echo "[stage] FATAL: model transfer failed"; exit 1; }
    MODEL_PATH_ARG=(--model-path "$LOCAL_COPY")
elif [ -n "${C3_MODEL_PATH:-}" ]; then
    MODEL_PATH_ARG=(--model-path "$C3_MODEL_PATH")
else
    WANT="${MODEL:-}"
    [ -z "$WANT" ] && [ "${PRESET:-}" = "t2-pilot" ] && WANT="llada-8b-base"
    case "$WANT" in
        llada-8b|llada-8b-instruct) WANT_VARIANT=instruct ;;
        llada-8b-base) WANT_VARIANT=base ;;
        *) WANT_VARIANT="" ;;
    esac
    if [ -n "$WANT_VARIANT" ]; then
        SHARED_VARIANT=absent
        if ssh $SSH_OPTS "$USER@ada" "test -r $SHARED_LLADA/config.json" 2>/dev/null; then
            # Not the chat template (Base ships the same one, and it mentions
            # <|eot_id|>): Instruct's vocabulary has <|eot_id|> as an added-token
            # entry, Base's does not (c3.traces.models.detect_variant).
            if ssh $SSH_OPTS "$USER@ada" "grep -qE '\"content\": *\"<\\|eot_id\\|>\"' $SHARED_LLADA/tokenizer_config.json" 2>/dev/null; then
                SHARED_VARIANT=instruct
            else
                SHARED_VARIANT=base
            fi
        fi
        echo "[stage] $SHARED_LLADA: $SHARED_VARIANT; requested LLaDA-8B-$WANT_VARIANT"
        if [ "$SHARED_VARIANT" = "$WANT_VARIANT" ]; then
            LOCAL_LLADA="$C3_SCRATCH/llada_model_$SHARED_VARIANT"
            if [ ! -r "$LOCAL_LLADA/config.json" ]; then
                mkdir -p "$LOCAL_LLADA"
                rsync -a --inplace --partial -e "ssh $SSH_OPTS" \
                    "$USER@ada:$SHARED_LLADA/" "$LOCAL_LLADA/" \
                    || { echo "[stage] FATAL: LLaDA transfer failed"; exit 1; }
            fi
            MODEL_PATH_ARG=(--model-path "$LOCAL_LLADA")
        else
            echo "[stage] downloading the pinned hub revision into $HF_HOME"
        fi
    fi
fi
