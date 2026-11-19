#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# 在 worker 上从 NFS tar 载入镜像，并把 <repo>:<tag> 指向它（覆盖本地同名旧 tag）。
# 用法: ./load_image_from_nfs.sh <tar 路径> [期望的 image_id 全量或 12 位短 ID]
set -euo pipefail
TAR=$1; WANT=${2:-}
REPO=${REPO:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni}
TAG=${TAG:-arm64-a100-latest}
H=$(hostname)
echo "[$H] == load $TAR"
docker load -i "$TAR" 2>&1 | tail -2
GOT=$(docker images --format '{{.ID}}' "$REPO:$TAG" | head -1)
short(){ local s=${1#sha256:}; echo "${s:0:12}"; }
GOTS=$(short "$GOT"); WANTS=$(short "$WANT")
echo "[$H] $REPO:$TAG -> $GOTS (want ${WANTS:-any})"
if [ -n "$WANT" ] && [ "$GOTS" != "$WANTS" ]; then echo "[$H] MISMATCH"; exit 1; fi
echo "[$H] OK"
