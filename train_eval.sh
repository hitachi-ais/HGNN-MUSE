OUTPUT_DIR=output
TRAIN_OUTPUT_DIR=$OUTPUT_DIR/train
TEST_OUTPUT_DIR=$OUTPUT_DIR/test

SEED=42
DEVICE=cuda

TRAIN_N_VARIABLES=(5 20)
TRAIN_MAX_ITN=5000
TRAIN_UPDATE_EPOCHS=4
TRAIN_PPO_EPOCHS=200
TRAIN_PPO_CLIP=0.2
TRAIN_PPO_VALUE_LOSS_COEF=0.5
TRAIN_PPO_ENTROPY_COEF=0.001
TRAIN_PPO_GAMMA=0.99
TRAIN_PPO_LAM=0.95
TRAIN_LR=2e-05
TRAIN_WEIGHT_DECAY=0.0
TRAIN_MAX_BATCH_SIZE=1024
TRAIN_NUM_EPISODE_PROBLEMS=4

TEST_NUM_PROBLEMS=500
TEST_MAX_ITN=10000
TEST_N_VARIABLES=(5 20)


python src/train.py \
    --output_dir $TRAIN_OUTPUT_DIR \
    --n_variables ${TRAIN_N_VARIABLES[@]} \
    --max_itn $TRAIN_MAX_ITN \
    --device $DEVICE \
    --update_epochs $TRAIN_UPDATE_EPOCHS \
    --ppo_epochs $TRAIN_PPO_EPOCHS \
    --ppo_clip $TRAIN_PPO_CLIP \
    --ppo_value_loss_coef $TRAIN_PPO_VALUE_LOSS_COEF \
    --ppo_entropy_coef $TRAIN_PPO_ENTROPY_COEF \
    --ppo_gamma $TRAIN_PPO_GAMMA \
    --ppo_lam $TRAIN_PPO_LAM \
    --lr $TRAIN_LR \
    --weight_decay $TRAIN_WEIGHT_DECAY \
    --max_batch_size $TRAIN_MAX_BATCH_SIZE \
    --seed $SEED \
    --num_episode_problems $TRAIN_NUM_EPISODE_PROBLEMS

python src/evaluate.py \
    --model_dir $TRAIN_OUTPUT_DIR \
    --output_dir $TEST_OUTPUT_DIR \
    --num_problems $TEST_NUM_PROBLEMS \
    --max_itn $TEST_MAX_ITN \
    --n_variables ${TEST_N_VARIABLES[@]} \
    --device $DEVICE \
    --itn_limits 500 1000 5000 10000