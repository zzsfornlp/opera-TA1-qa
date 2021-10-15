#

# main decoding

import argparse
import logging
import json

import torch
import tqdm
from qa_model import QaModel
from qa_data import QaInstance, TextPiece, set_gr, GR
from qa_eval import main_eval

def parse_args():
    parser = argparse.ArgumentParser('Main Decoding')
    # input & output
    parser.add_argument('--mode', type=str, default='csr', choices=['squad', 'csr'])
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_kwargs', type=str, default='None')  # for example: '{"qa_label_pthr":1.}'
    # more
    parser.add_argument('--device', type=int, default=0)  # gpuid, <0 means cpu
    parser.add_argument('--batch_size', type=int, default=16)
    # --
    args = parser.parse_args()
    logging.info(f"Start decoding with: {args}")
    return args

def batched_forward(args, model, insts):
    # --
    # sort by length
    sorted_insts = [(ii, zz) for ii, zz in enumerate(insts)]
    sorted_insts.sort(key=lambda x: len(x[-1]))
    # --
    bs = args.batch_size
    tmp_logits = []
    for ii in range(0, len(sorted_insts), bs):
        cur_insts = [z[-1] for z in sorted_insts[ii:ii+bs]]
        input_ids, attention_mask, token_type_ids = QaInstance.batch_insts(cur_insts)  # [bs, len]
        res = model.forward(input_ids, attention_mask, token_type_ids, ret_dict=True)
        res_logits = list(res['logits'].detach().cpu().numpy())  # List[len, ??]
        tmp_logits.extend(res_logits)
    # --
    # re-sort back
    sorted_logits = [(xx[0], zz) for xx, zz in zip(sorted_insts, tmp_logits)]
    sorted_logits.sort(key=lambda x: x[0])  # resort back
    all_logits = [z[-1] for z in sorted_logits]
    return all_logits  # List(arr)

def decode_squad(args, model):
    # load dataset
    with open(args.input_path) as fd:
        dataset_json = json.load(fd)
        dataset = dataset_json['data']
    # decode them
    all_preds = {}
    for article in tqdm.tqdm(dataset):
        # load the qa insts
        cur_insts = []
        for p in article['paragraphs']:
            context = TextPiece(p['context'])
            for qa in p['qas']:
                qid = qa['id']
                question = TextPiece(qa['question'])
                cur_insts.append(QaInstance(context, question, qid))
        # forward to get logits
        cur_logits = batched_forward(args, model, cur_insts)
        # decode them
        for one_inst, one_logit in zip(cur_insts, cur_logits):
            ans_left, ans_right = model.decode(one_inst, one_logit)
            ans_span = one_inst.get_answer_span(ans_left, ans_right)
            all_preds[one_inst.qid] = ans_span.get_orig_str()
        # --
    # --
    # output and eval
    logging.info("Decoding finished, output and eval:")
    if args.output_path:
        with open(args.output_path, 'w') as fd:
            json.dump(all_preds, fd)
    # eval
    eval_res = main_eval(dataset, all_preds)
    logging.info(f"Eval results: {eval_res}")
    return all_preds

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    args = parse_args()
    # load model
    model = QaModel.load_model(args.model, eval(args.model_kwargs))
    set_gr(model.tokenizer, args.device)
    model.eval()
    model.to(GR.device)
    # --
    logging.info("#--\nStart decoding:\n")
    with torch.no_grad():
        if args.mode == 'csr':
            raise NotImplementedError()
        elif args.mode == 'squad':
            decode_squad(args, model)
        else:
            raise NotImplementedError()
    # --

if __name__ == '__main__':
    main()

"""
# examples runs
# decode with ptr-models
python3 qa_main.py --mode squad --input_path data/dev-v2.0.json --output_path '' --model try0/zmodel.best
# -> OrderedDict([('exact', 77.50357955024005), ('f1', 80.42150998666928), ('total', 11873), ('HasAns_exact', 72.62145748987854), ('HasAns_f1', 78.46568624691706), ('HasAns_total', 5928), ('NoAns_exact', 82.3717409587889), ('NoAns_f1', 82.3717409587889), ('NoAns_total', 5945)])
# decode with label-models
python3 qa_main.py --mode squad --input_path data/dev-v2.0.json --output_path '' --model try1/zmodel.best --model_kwargs "{'qa_label_pthr':3.}"
# -> OrderedDict([('exact', 75.76012802156153), ('f1', 79.63949665083895), ('total', 11873), ('HasAns_exact', 72.80701754385964), ('HasAns_f1', 80.57687984740397), ('HasAns_total', 5928), ('NoAns_exact', 78.70479394449117), ('NoAns_f1', 78.70479394449117), ('NoAns_total', 5945)])
# --
# search for the tuned models
for ii in {0..5}; do
for tt in 0. 1. 2. 3. 4.; do
echo "RUN model${ii} pthr=${tt}"
python3 qa_main.py --mode squad --input_path data/dev-v2.0.json --output_path '' --model tune1013/zmodel${ii}.best --model_kwargs "{'qa_label_pthr':${tt}}"
done
done |& tee tune1013/_log_dec
cat tune1013/_log_dec | grep -E "RUN|Eval"
# -> the best is: rate3e-5,neg10 -> pthr=3 -> 76.299/79.989
"""
