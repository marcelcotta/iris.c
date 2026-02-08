#!/bin/bash
set -e

# Default: distilled 4B model. Use --base for base model, --9b for 9B model.
REPO="FLUX.2-klein-4B"
OUT="./flux-klein-model"
TOKEN=""

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --base)
            REPO="FLUX.2-klein-base-4B"
            OUT="./flux-klein-base-model"
            ;;
        --9b)
            REPO="FLUX.2-klein-9B"
            OUT="./flux-klein-9b-model"
            ;;
        --token)
            TOKEN="$2"
            shift
            ;;
        *)
            echo "Usage: $0 [--base] [--9b] [--token TOKEN]"
            exit 1
            ;;
    esac
    shift
done

# Try to find token from environment or file
if [ -z "$TOKEN" ] && [ -n "$HF_TOKEN" ]; then
    TOKEN="$HF_TOKEN"
fi
if [ -z "$TOKEN" ] && [ -f "hf_token.txt" ]; then
    TOKEN=$(cat hf_token.txt | tr -d '[:space:]')
fi

# Set up auth header if token is available
AUTH_HEADER=""
if [ -n "$TOKEN" ]; then
    AUTH_HEADER="-H \"Authorization: Bearer $TOKEN\""
    echo "Using authentication token"
fi

echo "Downloading $REPO..."

BASE="https://huggingface.co/black-forest-labs/$REPO/resolve/main"

# Helper function to download with optional auth
dl() {
    if [ -n "$TOKEN" ]; then
        curl -L -H "Authorization: Bearer $TOKEN" -o "$1" "$2"
    else
        curl -L -o "$1" "$2"
    fi
    # Check for auth errors
    if [ $? -ne 0 ]; then
        echo "Error downloading $2"
        echo "If this is a gated model, you need a HuggingFace token:"
        echo "  1. Accept the license at https://huggingface.co/black-forest-labs/$REPO"
        echo "  2. Get your token from https://huggingface.co/settings/tokens"
        echo "  3. Run: $0 --token YOUR_TOKEN"
        echo "  Or set HF_TOKEN env var, or save token to hf_token.txt"
        exit 1
    fi
}

mkdir -p "$OUT"/{text_encoder,tokenizer,transformer,vae}

# model_index.json (needed for autodetection)
dl "$OUT/model_index.json" "$BASE/model_index.json"

# text_encoder (Qwen3 - ~8GB for 4B, ~16GB for 9B)
dl "$OUT/text_encoder/config.json" "$BASE/text_encoder/config.json"
dl "$OUT/text_encoder/generation_config.json" "$BASE/text_encoder/generation_config.json"
dl "$OUT/text_encoder/model.safetensors.index.json" "$BASE/text_encoder/model.safetensors.index.json"

# Discover and download all safetensors shards from the index
SHARDS=$(python3 -c "
import json, sys
try:
    with open('$OUT/text_encoder/model.safetensors.index.json') as f:
        idx = json.load(f)
    shards = sorted(set(idx['weight_map'].values()))
    for s in shards:
        print(s)
except:
    # Fallback: assume 2 shards
    print('model-00001-of-00002.safetensors')
    print('model-00002-of-00002.safetensors')
" 2>/dev/null)

for shard in $SHARDS; do
    dl "$OUT/text_encoder/$shard" "$BASE/text_encoder/$shard"
done

# tokenizer
dl "$OUT/tokenizer/added_tokens.json" "$BASE/tokenizer/added_tokens.json"
dl "$OUT/tokenizer/chat_template.jinja" "$BASE/tokenizer/chat_template.jinja"
dl "$OUT/tokenizer/merges.txt" "$BASE/tokenizer/merges.txt"
dl "$OUT/tokenizer/special_tokens_map.json" "$BASE/tokenizer/special_tokens_map.json"
dl "$OUT/tokenizer/tokenizer.json" "$BASE/tokenizer/tokenizer.json"
dl "$OUT/tokenizer/tokenizer_config.json" "$BASE/tokenizer/tokenizer_config.json"
dl "$OUT/tokenizer/vocab.json" "$BASE/tokenizer/vocab.json"

# transformer
dl "$OUT/transformer/config.json" "$BASE/transformer/config.json"
dl "$OUT/transformer/diffusion_pytorch_model.safetensors" "$BASE/transformer/diffusion_pytorch_model.safetensors"

# vae (~168 MB)
dl "$OUT/vae/config.json" "$BASE/vae/config.json"
dl "$OUT/vae/diffusion_pytorch_model.safetensors" "$BASE/vae/diffusion_pytorch_model.safetensors"

echo "Done. -> $OUT"
