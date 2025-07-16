set -x

taco_train_path=$DATA_DIR/train.parquet
taco_test_path=$DATA_DIR/test.parquet
HF_TOKEN=$HF_TOKEN


# train_files="['$taco_train_path']"
# test_files="['$taco_test_path']"

# docker run -it -p 5000:5000 volcengine/sandbox-fusion:server-20241204


taco_train_path=/home/ubuntu/neslaai/tinycodezeroq/tinyzerocodeq/data_preprocess/code_dataset_taco_wrapped.json
taco_test_path=/home/ubuntu/neslaai/tinycodezeroq/tinyzerocodeq/data_preprocess/code_dataset_taco_wrapped.json

train_files="['$taco_train_path']"
test_files="['$taco_test_path']"

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct\
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.rollout.hf_token=$HF_TOKEN \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='tinyzerocodeq' \
    trainer.experiment_name='qwen_2.5_3b_leetcode' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@
