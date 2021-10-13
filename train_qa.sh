
# --
DATA_DIR=${DATA_DIR:-../data/}
PY_OPTS=${PY_OPTS:-../run_qa.py}
python ${PY_OPTS} \
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
    --output_dir ./ \
    --per_gpu_eval_batch_size 8  \
    --per_gpu_train_batch_size 3  \
    --gradient_accumulation_steps 8 \
    --logging_steps 1000 \
    --evaluate_during_training ${EXTRA_ARGS}

# --
# train:
# CUDA_VISIBLE_DEVICES=0 bash ../train_qa.sh |& tee _log
# CUDA_VISIBLE_DEVICES=0 EXTRA_ARGS='--qa_head_type ptr' bash ../train_qa.sh |& tee _log
# CUDA_VISIBLE_DEVICES=0 DATA_DIR=?? PY_OPTS='-m pdb ../run_qa.py' bash ../train_qa.sh
# --
# decode:
# CUDA_VISIBLE_DEVICES=0 python3 ../run_qa.py --version_2_with_negative --do_eval --data_dir ../data/ --predict_file dev-v2.0.json --qa_load_name zmodel.best
# --qa_load_kwargs "{'qa_label_pthr':0.}"
# --
# results:
# 'ptr' seems reasonable: {'exact_C0': 78.37109407900277, 'f1_C0': 81.15048470470018, 'total_C0': 11873, 'HasAns_exact_C0': 73.36369770580296, 'HasAns_f1_C0': 78.93044954434984, 'HasAns_total_C0': 5928, 'NoAns_exact_C0': 83.36417157275021, 'NoAns_f1_C0': 83.36417157275021, 'NoAns_total_C0': 5945, 'best_exact_C0': 78.37109407900277, 'best_exact_thresh_C0': 0.0, 'best_f1_C0': 81.15048470470008, 'best_f1_thresh_C0': 0.0}
# 'label' is worse (pthr=3): {'exact_C0': 72.9470226564474, 'f1_C0': 76.75539541535737, 'total_C0': 11873, 'HasAns_exact_C0': 73.41430499325236, 'HasAns_f1_C0': 81.04197195791781, 'HasAns_total_C0': 5928, 'NoAns_exact_C0': 72.48107653490328, 'NoAns_f1_C0': 72.48107653490328, 'NoAns_total_C0': 5945, 'best_exact_C0': 72.91333277183526, 'best_exact_thresh_C0': 0.0, 'best_f1_C0': 76.72170553074513, 'best_f1_thresh_C0': 0.0}
# --
