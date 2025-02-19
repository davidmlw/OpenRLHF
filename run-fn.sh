
DATASETDIR=/lustre/fsw/portfolios/coreai/users/liweim/dataset


set -x

##ray job submit --address="http://127.0.0.1:8265" \
##   --runtime-env-json='{"working_dir": "/openrlhf"}' \
##   --

train_args=(
   --ref_num_nodes 1
   --ref_num_gpus_per_node 2
   --critic_num_nodes 1
   --critic_num_gpus_per_node 2
   --actor_num_nodes 1
   --actor_num_gpus_per_node 2
   --vllm_num_engines 2
   --vllm_tensor_parallel_size 1
#   --colocate_actor_ref
#   --pretrain OpenRLHF/Llama-3-8b-sft-mixture
   --pretrain $DATASETDIR/Qwen2.5-3B
   --remote_rm_url /openrlhf/examples/scripts/reward_func.py
#   --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf
   --micro_train_batch_size 1
#   --micro_train_batch_size 8
   --train_batch_size 16
   --micro_rollout_batch_size 1
   --rollout_batch_size 64
   --max_samples 1024
   --max_epochs 1
   --prompt_max_len 256
   --generate_max_len 1024
#   --zero_stage 3
   --zero_stage 1
   --bf16
   --actor_learning_rate 5e-7
   --critic_learning_rate 9e-6
   --init_kl_coef 0.01
   --prompt_data OpenRLHF/prompt-collection-v0.1
   --input_key context_messages
   --apply_chat_template
   --normalize_reward
   --packing_samples
   --adam_offload
   --flash_attn
   --gradient_checkpointing
)

python3 -m openrlhf.cli.train_ppo_ray ${train_args[@]}


#   --use_wandb eb69adb0d7a2262ca4584b1785fcd114cc009585
