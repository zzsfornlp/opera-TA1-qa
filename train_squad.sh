
# --
DATA_DIR=${DATA_DIR:-../data/}
PY_OPTS=${PY_OPTS:-../run_squad.py}
python ${PY_OPTS} \
    --model_type bert \
    --model_name_or_path bert-large-cased \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --data_dir $DATA_DIR \
    --train_file train-v2.0.json \
    --predict_file dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./finetuned_squad/ \
    --per_gpu_eval_batch_size 3  \
    --per_gpu_train_batch_size 3  \
    --gradient_accumulation_steps 8 \
    --save_steps 1000000 \
    --logging_steps 1000 \
    --evaluate_during_training ${EXTRA_ARGS}

# --
# CUDA_VISIBLE_DEVICES=0 bash ../train_squad.sh |& tee _log
# CUDA_VISIBLE_DEVICES=0 DATA_DIR=?? PY_OPTS='-m pdb ../run_squad.py' bash ../train_squad.sh
