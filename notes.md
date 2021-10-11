### Some notes

Use scripts/train_squad.sh to train on squad v2.0.
- `data_dir`: input data dir as well as cache dir (let it be as the default as '.')
- The processing of `SquadExample` seems not quite careful (tokenized by spaces), but currently let it be ...
- Then with `squad_convert_example_to_features`, `SquadExample` is converted to `SquadFeatures`
- The inputs to the model are simply: input_ids/attention_mask/token_type_ids/start_positions/end_positions
