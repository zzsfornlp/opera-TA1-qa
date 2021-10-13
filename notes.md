### Some notes

# --
# Using original 'run_squad.py'
Use train_squad.sh to train on squad v2.0.
- `data_dir`: input data dir as well as cache dir (let it be as the default as '.')
- The processing of `SquadExample` seems not quite careful (tokenized by spaces), but currently let it be ...
- Then with `squad_convert_example_to_features`, `SquadExample` is converted to `SquadFeatures`
- The inputs to the model are simply: input_ids/attention_mask/token_type_ids/start_positions/end_positions
# --
# decoding example:
SQUAD_DIR=../events/data/squad/
CUDA_VISIBLE_DEVICES=0 python -m pdb run_squad.py --model_type bert --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --per_gpu_eval_batch_size 8 --max_seq_length 384 --doc_stride 128 --output_dir ./ --model_name_or_path ./finetuned_squad/ --do_eval

# --
# Using modified 'run_qa.py'
Use train_qa.sh to train it, we can have the models of 'ptr' (pointer) or 'label' (labeling).
