## Some brief notes

### Part 0 (training with squad):
### Using original `run_squad.py` (note: this is deprecated by `run_qa.py`)
Use `train_squad.sh` to train on squad v2.0.
- `data_dir`: input data dir as well as cache dir (let it be as the default as '.')
- The processing of `SquadExample` seems not quite careful (tokenized by spaces), but currently let it be ...
- Then with `squad_convert_example_to_features`, `SquadExample` is converted to `SquadFeatures`
- The inputs to the model are simply: input_ids/attention_mask/token_type_ids/start_positions/end_positions
### decoding example:
SQUAD_DIR=../events/data/squad/
CUDA_VISIBLE_DEVICES=0 python -m pdb run_squad.py --model_type bert --version_2_with_negative --predict_file $SQUAD_DIR/dev-v2.0.json --per_gpu_eval_batch_size 8 --max_seq_length 384 --doc_stride 128 --output_dir ./ --model_name_or_path ./finetuned_squad/ --do_eval

### Part 1 (training with squad):
### Using modified `run_qa.py`
Use `run_qa.sh` to train with squad v2.0.
- `run_qa.py` is adopted from `run_squad.py` and it keeps the main training framework and data processing parts. But the model is replaced by the ones specified in `qa_model.py`.
- See `run_qa.sh` for an example running of training.
- `qa_model.py` specifies a `QaModel` where there are two modes: 1) `--qa_head_type ptr`: the commonly utilized two-pointer decoder where two pointers are stacked upon the encoder to select the start and end subtoken of the answer span; 2) `--qa_head_type label`: a simply binary classifier (scorer) satcked to indicate whether the current subtoken is inside the answer span (this can be viewed as an "IO" encoding.)
- Notice that currently the decoding method in `run_qa.py` still follows the `ptr` version and thus the results for `label` are much worse. If we use `qa_main.py` for testing, the results can be better (only slightly worse than or similar if tuned if compared to `ptr`) for `label` with specific decoding method.
- For our final decoding, we will use the `label` mode since we want to outputs answers from our extracted entities/events.
- For the labeling mode, there are some hyper-parameter that may be tuned, please refer to `qa_model.py:QaModel.add_args`.
- `run_qa.py` is only utilized for training and we will have separate scripts for final decoding. It follows the original `run_squad.py` and accepts input files in the squad format. If we later want to further fine-tune with our own datasets and still want to re-use this script, the datasets will need to be converted to the squad format.

### Part 2 (testing):
### Using `qa_main.py`
Use `qa_main.py` for testing, and see `qa_main.sh` for an example.
- In `qa_main.py`, we have two testing mode: 1) `--mode squad`, where we test on squad files, in this mode, `--input_path` should indicate an input squad json file and the `--output_path` should indicate the output squad prediction-json path. 2) `--mode csr`, where we test on csr files, in this mode, `--input_path` indicates input dir where there are `*.csr.json` input csr files, and the `--output_path` indicates the output dir where we write `*.csr.json` output csr files with query answer (initial claim frames) added.
- For the csr mode, we further need two other inputs: 1) `--input_pct` is the parent-children tab files, where we read the querying topic/subtopic of each document, 2) `--input_topic` is the parsed topic files, where we read the topics/subtopics/templates and our converted questions (using `opera-TA1-event:msp2.tasks.opera.tools.parse_topics`).
- See `qa_main.sh` for an example of decoding csr files (note that currently this is also the script that we are calling in our text pipeline).
- Another major file here is `qa_data.py` where we handle the data related aspects for testing, here are some brief descriptions for it:
- (1) `NTokenizer` and `SubToker` are the word-tokenizer and subword-tokenizer, the former is a wrapper for nltk's word-tokenizer and the latter is a wrapper for the `transformers` ones. 
- (2) `GlobalResources` is places to store some global variables like the original subword tokenizer and global device and `GR` is the one global variable that stores things, use `set_gr` to initialize things.
- (3) `TextPiece` is a class for a piece of text (like question or context), it is intialized with a text str and then do word-tokenization and subword-tokenization, and stores all these information, especially including the token character-span and subtoken to token information which we need to use later. Check it for more details, I guess things are straight-forward there.
- (3.5) Notice that `TextPiece.merge_pieces` creates a "fake" `TextPiece` where we only want to merge several pieces of sub-tokens into a larger one but do not care about more details (like char/subtoken mappings). This is just for convenience and only utilized in `qa_main`'s forwardings of chunks of sentences.
- (4) `TextSpan` denotes a span over a `TextPiece`, this is simply a helping class.
- (5) `QaInstance` is a pair of a question `TextPiece` and a context `TextPiece`, these two are further concated to form the input to bert (CLS ... question ... SEP ... context ... SEP), notice that we truncate things with limits set in `GR`.
- (6) `CsrDoc` is a representation of csr document, where we read a csr-json input and parse its information. Go through its `__init__` for more details, which will also give a rough idea of the csr formats.
- Now back to `qa_main.py`, in the `csr` mode, for each csr doc, we first get its query subtopics/questions (by default, we query all the subtopics under its topic, which is controlled by `--csr_query_topic`). For each doc (see function `decode_one_csr`), we split the doc sentences into chunks and query the model for each pair of question and chunck. Then for each candidate entity (doc.cand_entities) or event (doc.cand_events), looking at their head token span, if any token's prob is larger than certain threshold (--csr_prob_thresh), we view it as an answer candidate. This candidate's score will be the average of the probs of all head tokens. Then we further rank the candidates and do some simple heuristic based pruning and output the answers.
- See `qa_main.py:decode_one_csr` for more details of the above procedure. Of course there can be other (and better) ways to do this, here is simply an initial trying with some simple heuristics.

