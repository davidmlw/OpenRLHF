#!/bin/bash

WORKDIR=$PWD
DATASETDIR=/lustre/fsw/portfolios/coreai/users/liweim/dataset
len_prompt_data=1024
#prompt_data=/lustre/fsw/portfolios/coreai/users/liweim/dataset/gsm8k
#prompt_data=/lustre/fsw/portfolios/coreai/users/liweim/dataset/prompt-collection-v0.1/data
prompt_data=OpenRLHF/prompt-collection-v0.1
#prompt_data=/lustre/fsw/portfolios/coreai/users/liweim/dataset/apps

## ray job submit --working-dir . --address="http://127.0.0.1:8265" \
##    --runtime-env-json='{"working_dir": "."}' \
##    --

ulimit -n 4096

function small {

    train_args=(
       --ref_num_nodes 1
       --ref_num_gpus_per_node 2
       --critic_num_nodes 1
       --critic_num_gpus_per_node 2
       --actor_num_nodes 1
       --actor_num_gpus_per_node 2
       --vllm_num_engines 2
       --vllm_tensor_parallel_size 1
       --vllm_sync_backend nccl
    #   --colocate_actor_ref
       --pretrain Qwen/Qwen2.5-3B
       --critic_pretrain Qwen/Qwen2.5-3B
    #   --pretrain OpenRLHF/Llama-3-8b-sft-mixture
    #   --critic_pretrain OpenRLHF/Llama-3-8b-rm-mixture
       --remote_rm_url $WORKDIR/examples/scripts/reward_func.py
       --micro_train_batch_size 2
    #   --train_batch_size 256
       --train_batch_size 16
       --micro_rollout_batch_size 1
    #   --rollout_batch_size 1024
       --rollout_batch_size 32
       --max_samples 32
       --max_epochs 1
    #   --prompt_max_len 2048
       --prompt_max_len 256
    #   --generate_max_len 14336
       --generate_max_len 1024
       --zero_stage 1
       --bf16
       --actor_learning_rate 5e-7
       --critic_learning_rate 9e-6
       --init_kl_coef 0.01
       --prompt_data $prompt_data
       --input_key context_messages
       --apply_chat_template
    #   --disable_fast_tokenizer
       --flash_attn
       --adam_offload
       --gradient_checkpointing
       --load_checkpoint
       --value_head_prefix score
       --eval_steps 200
       --save_steps 500
       --freezing_actor_steps 0
       --load_checkpoint
    )

    python3 -m openrlhf.cli.train_ppo_ray ${train_args[@]}
}


function large {

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
#       --colocate_all_models
       --pretrain $DATASETDIR/Qwen2.5-3B
       --critic_pretrain $DATASETDIR/Qwen2.5-3B
       --remote_rm_url $WORKDIR/examples/scripts/reward_func.py
       --micro_train_batch_size 1
       --train_batch_size 256
       --micro_rollout_batch_size 1
       --rollout_batch_size 1024
       --max_samples $len_prompt_data
       --num_episodes 1
       --max_epochs 1
       --prompt_max_len 2048
       --generate_max_len 14336
#       --generate_max_len 4096
#       --generate_max_len 8192
#       --generate_max_len 2048
       --zero_stage 1
       --bf16
       --actor_learning_rate 5e-7
       --critic_learning_rate 9e-6
       --init_kl_coef 0.01
       --prompt_data $prompt_data
       --input_key context_messages
       --apply_chat_template
    #   --disable_fast_tokenizer
       --flash_attn
       --adam_offload
       --gradient_checkpointing
       --load_checkpoint
       --value_head_prefix score
       --eval_steps 200
       --save_steps 500
       --freezing_actor_steps 0
       --load_checkpoint
    )

    #python3 -m nvtx openrlhf/cli/train_ppo_ray.py ${train_args[@]}
    python3 -m openrlhf.cli.train_ppo_ray ${train_args[@]}
}



#   --input_key input
#   --input_template '<s>{}[|AI|]:'
#   --generate_min_len 14336
#

if declare -f $1 > /dev/null
then
    $@
else
    echo "'$1' is not a known function name" >&2
fi


# git@hf.co:Qwen/Qwen2.5-3B
