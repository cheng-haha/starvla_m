#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Single-node 4xH100 setup
# -------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL needs a valid socket interface for bootstrap.
# In your container, only "eth0" and "lo" exist; "bond0" does NOT.
export NCCL_SOCKET_IFNAME=eth0

# For single-node training, IB is not required. Disable it to avoid HCA mismatch issues.
export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA

# Optional: NCCL logs (useful for debugging network/bootstrap)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Optional: robustness settings
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # seconds

# -------------------------
# W&B settings
# -------------------------
export WANDB_CONSOLE=off
export WANDB_MODE=offline

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/inspire/hdd/global_user/chengdongzhou-240108390137/ai_models/Qwen/Qwen3-VL-4B-Instruct
config_yaml=./examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
oxe_data_root=/inspire/hdd/global_user/chengdongzhou-240108390137/datasets/IPEC-COMMUNITY
data_mix=bridge_rt_1
run_root_dir=./results/Checkpoints
run_id=1221_${data_mix}_${Framework_name}
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/



accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_simplerEnv \
  --wandb_entity chenghaha \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
