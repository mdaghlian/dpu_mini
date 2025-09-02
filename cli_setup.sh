#!/usr/bin/env bash

DPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# make scripts executable
chmod -R 775 ${DPU_DIR}/bin
export PATH=${PATH}:${DPU_DIR}/bin
