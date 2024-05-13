#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

# Warm HF cache
mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
popd
export HF_DATASETS_OFFLINE=1

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=12776 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
  configs/mitchish7-s3.yaml \
    --run_name=mitchish7-fp32 \
    --wandb.name=mitchish7-fp32 \
    --wandb.group=mitchish7-fp32 \
    --model.flash_attention=false \
    --fused_loss=false \
    --fsdp.wrapping_strategy=by_block_and_size \
    --fsdp.sharding_strategy=FULL_SHARD \
    --save_folder=runs/ \
    --activation_checkpointing=fine_grained \
    --device_train_microbatch_size=2 \
    --global_train_batch_size=1024 \
    --save_interval=50 \
    --eval_interval=50 \
    --precision=fp32 \
    --fsdp.precision=pure \
    --save_overwrite \
    '--load_path=${path.last_checkpoint:s3://ai2-llm/checkpoints/OLMo-medium/mitchish7-fp32/}' \
    --load_path=s3://ai2-llm/checkpoints/OLMo-medium/mitchish7/step614000/