#!/bin/bash
set -euo pipefail

TRAIN_EPOCHS=(2 1)
POST_EPOCHS=(1 3 5 10)
ARCHS=(moe_subspace moe)

for TRAIN_EPOCH in "${TRAIN_EPOCHS[@]}"; do
  for ARCH in "${ARCHS[@]}"; do
    for POST_EPOCH in "${POST_EPOCHS[@]}"; do

      LOG_FILE="encode_moe_train${TRAIN_EPOCH}_arch_${ARCH}_post${POST_EPOCH}.log"

      echo "================================================"
      echo "Running:"
      echo "  num_train_epochs      = ${TRAIN_EPOCH}"
      echo "  num_post_train_epochs = ${POST_EPOCH}"
      echo "  lora_architecture     = ${ARCH}"
      echo "  log = ${LOG_FILE}"
      echo "================================================"

      python src/encode_moe.py \
          --inference_method prag \
          --num_train_epochs ${TRAIN_EPOCH} \
          --num_post_train_epochs ${POST_EPOCH} \
          --lora_architecture ${ARCH} \
          > "${LOG_FILE}" 2>&1

      echo "Finished: train=${TRAIN_EPOCH}, arch=${ARCH}, post=${POST_EPOCH}"
      echo
    done
  done
done

echo "All runs completed successfully."

