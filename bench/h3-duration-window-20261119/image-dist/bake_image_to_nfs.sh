#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# 把 GPUStack 用的 vLLM-Omni 镜像落成 NFS 上的 tar，供各 worker docker load。
# 用法: ./bake_image_to_nfs.sh [image:tag]
# 产物: /nfs-models/images/vllm-omni-<tag>-<image_id8>.tar{,.sha256,.manifest}
set -euo pipefail
REG=${REG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com}
REPO=${REPO:-$REG/reputationly/vllm-omni}
IMG=${1:-$REPO:arm64-a100-latest}
DIR=${DIR:-/nfs-models/images}
TAG=${IMG##*:}

echo "== pull $IMG"
docker pull -q "$IMG" >/dev/null
IMAGE_ID=$(docker inspect -f '{{.Id}}' "$IMG")
DIGEST=$(docker inspect -f '{{range .RepoDigests}}{{println .}}{{end}}' "$IMG" | grep "^$REPO@" | head -1 | sed 's/.*@//')
SHORT=${IMAGE_ID#sha256:}; SHORT=${SHORT:0:12}
TAR="$DIR/vllm-omni-$TAG-$SHORT.tar"

echo "== save -> $TAR  (image_id=$IMAGE_ID digest=${DIGEST:-unknown})"
docker save -o "$TAR.tmp" "$IMG"
mv "$TAR.tmp" "$TAR"
sha256sum "$TAR" | tee "$TAR.sha256"
cat > "$TAR.manifest" <<M
image=$IMG
tag=$TAG
image_id=$IMAGE_ID
repo_digest=${DIGEST:-unknown}
tar=$TAR
created=$(date -Is)
baked_on=$(hostname)
M
cat "$TAR.manifest"
ls -lh "$TAR"
