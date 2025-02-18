#!/bin/bash

WORKDIR=$PWD
len_prompt_data=1024
#prompt_data=/lustre/fsw/portfolios/coreai/users/liweim/dataset/gsm8k
#prompt_data=/lustre/fsw/portfolios/coreai/users/liweim/dataset/prompt-collection-v0.1/data
prompt_data=OpenRLHF/prompt-collection-v0.1


train_args=(
   --ref_num_nodes 1 
   --ref_num_gpus_per_node 4 
   --critic_num_nodes 1 
   --critic_num_gpus_per_node 4 
   --actor_num_nodes 1 
   --actor_num_gpus_per_node 4 
   --vllm_num_engines 4 
   --vllm_tensor_parallel_size 1 
   --vllm_sync_backend nccl 
   --colocate_actor_ref 
#   --pretrain /lustre/fsw/portfolios/coreai/users/liweim/dataset/Qwen2.5-3B 
#   --critic_pretrain /lustre/fsw/portfolios/coreai/users/liweim/dataset/Qwen2.5-3B
#   --remote_rm_url $WORKDIR/examples/scripts/reward_func.py 
   
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture 
   --critic_pretrain OpenRLHF/Llama-3-8b-rm-mixture 
   --remote_rm_url $WORKDIR/examples/scripts/reward_func.py 

   
#  --ref_num_nodes 1 
#  --ref_num_gpus_per_node 2 
#  --reward_num_nodes 1 
#  --reward_num_gpus_per_node 2 
#  --critic_num_nodes 1 
#  --critic_num_gpus_per_node 2 
#  --actor_num_nodes 1 
#  --actor_num_gpus_per_node 2 
#  --vllm_num_engines 2 
#  --vllm_tensor_parallel_size 2 
#  --colocate_critic_reward 
#  --colocate_actor_ref 
#  --pretrain OpenRLHF/Llama-3-8b-sft-mixture 
#  --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture 
  --save_path examples/checkpoint/llama3-8b-rlhf 
  --micro_train_batch_size 8 
  --train_batch_size 128 
  --micro_rollout_batch_size 16 
  --rollout_batch_size 1024 
  --max_samples 100000 
  --max_epochs 1 
  --prompt_max_len 1024 
  --generate_max_len 1024 
  --zero_stage 3 
  --bf16 
  --actor_learning_rate 5e-7 
  --critic_learning_rate 9e-6 
  --init_kl_coef 0.01 
  --prompt_data OpenRLHF/prompt-collection-v0.1 
  --input_key context_messages 
  --disable_fast_tokenizer 
  --apply_chat_template 
  --normalize_reward 
  --packing_samples 
  --adam_offload 
  --flash_attn 
  --gradient_checkpointing
)

python3 -m openrlhf.cli.train_ppo_ray ${train_args[@]}

## ray job submit --working-dir . --address="http://127.0.0.1:8265" 
##    --runtime-env-json='{"working_dir": "."}' 
##    --
