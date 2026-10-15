#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# launch_h3_audio.sh <ARM: A|B>  -- A: no determinism wrap (baseline behaviour)
#                                  B: decode_latent wrapped in _AudioVAEDeterminismContext
set -e
# Archived as it ran on 2026-10-14; the body is deliberately left byte-identical to the
# version that produced the numbers in the write-up. The unquoted $NAME/$W/$ARM below are
# harmless here (simple values, no spaces) but are real sloppiness -- fix them if this is
# ever revived, not before, or the archive stops matching the record.
IMG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-sync-20261012-test
W=/nfs-output/qfix-20261014
ARM=$1
case "$ARM" in
  A) DET=0 ;;
  B) DET=1 ;;
  *) echo "usage: $0 A|B"; exit 2 ;;
esac
NAME=h3aud$ARM
docker rm -f $NAME 2>/dev/null || true
rm -f $W/aud$ARM.log
docker run -d --name $NAME --gpus all --network host --shm-size=32g \
  --env-file $W/base_env.txt \
  -e H3_AUDIO_PROBE_LOG=$W/aud$ARM.log \
  -e H3_AUDIO_DECODE_DETERMINISM=$DET \
  -v $W/ws_audio/vllm_omni:/usr/local/lib/python3.12/dist-packages/vllm_omni \
  -v /nfs-output:/nfs-output -v /nfs-models:/nfs-models -v /nfs-data:/nfs-data \
  "$IMG" \
  vllm serve /nfs-models/wuhanjisuan894/models/MiniMax-H3-Singularity-VDN-INT8 \
    --omni \
    --deploy-config /nfs-models/wuhanjisuan894/deploy-configs/minimax_h3_singularity_vdn8_fl2va_a100_40g.yaml \
    --allowed-local-media-path /nfs-output \
    --init-timeout 2400 --stage-init-timeout 2400 \
    --lora-path /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250-curve-fl2va-baked2 \
    --host 0.0.0.0 --port 40002
echo "arm=$ARM determinism=$DET container=$NAME started"
