
# --
SQUAD_DIR=${SQUAD_DIR:-../events/data/squad/}
PY_OPTS=${PY_OPTS:-run_squad.py}
python ${PY_OPTS} \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./finetuned_squad/ \
    --per_gpu_eval_batch_size=4  \
    --per_gpu_train_batch_size=4   \
    --save_steps 5000 \
    --evaluate_during_training

# --
# CUDA_VISIBLE_DEVICES=0 SQUAD_DIR=?? PY_OPTS='-m pdb run_squad.py' bash train_squad.sh
