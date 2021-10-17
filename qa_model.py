#

# qa models
import torch
import logging
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import logging
import numpy as np
import math
from transformers import AutoTokenizer, AutoModel, AutoConfig

# --
# binary cross-entropy loss
# [*], [*](0 or 1) -> [*]
def loss_binary(score_expr, gold_idxes, label_smoothing=0.):
    v0 = F.binary_cross_entropy_with_logits(score_expr, gold_idxes, reduction='none')
    if label_smoothing > 0.:  # todo(+N): can we directly put into targets?
        v1 = F.binary_cross_entropy_with_logits(score_expr, 1.-gold_idxes, reduction='none')
        ret = v1 * label_smoothing + v0 * (1.-label_smoothing)
        # note: substract baseline
        ret += ((1.-label_smoothing) * math.log(1.-label_smoothing) + label_smoothing * math.log(label_smoothing))
    else:
        ret = v0
    return ret  # [*]
# --

# --
class QaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # --
        self.args = args  # note: could save/load more here, but nevermind, just ignore others ...
        # --
        self.tokenizer, self.bert = QaModel.get_bert(args)
        _model_dim = self.bert.config.hidden_size
        if args.qa_head_type == 'label':
            self.qa = nn.Linear(_model_dim, 1)
        elif args.qa_head_type == 'ptr':  # start and end
            self.qa = nn.Linear(_model_dim, 2)
        else:
            raise NotImplementedError()

    def save(self, name: str):
        logging.info(f"Save model&args to {name} and {name}.args")
        d = self.state_dict()
        torch.save(d, f'{name}.m')
        torch.save(self.args, f'{name}.args')

    def load(self, name: str):
        logging.info(f"Load model from {name}")
        d = torch.load(f'{name}.m')
        self.load_state_dict(d)

    @staticmethod
    def create_model(args, try_load_name: str):
        model = QaModel(args)  # first create one!
        if try_load_name:
            model.load(try_load_name)
        else:
            logging.info("No model to load when create_model, skip loading!")
        return model

    @staticmethod
    def load_model(load_name: str, extra_args=None):
        # first load args
        args = torch.load(load_name+".args")
        if extra_args is not None:
            if not isinstance(extra_args, dict):
                extra_args = extra_args.__dict__
            for k, v in extra_args.items():
                # if hasattr(args, k):
                setattr(args, k, v)
                logging.info(f"Change loaded args: {k} -> {v}")
        # then create model (and load inside)
        args.qa_load_name = load_name
        model = QaModel.create_model(args, args.qa_load_name)
        return model

    @staticmethod
    def add_args(parser):
        # add arguments specific to the qa_model!
        # --
        parser.add_argument("--bert_model", default='bert-large-cased', type=str)
        parser.add_argument("--bert_cache", default=None, type=str)
        # --
        parser.add_argument("--qa_head_type", default='label', type=str, choices=['label', 'ptr'])
        parser.add_argument("--qa_label_ls", default=0., type=float)  # label-smoothing
        parser.add_argument("--qa_label_negratio", default=5., type=float)  # neg ratio to positive
        parser.add_argument("--qa_label_pthr", default=0., type=float)  # predicting threshold, >=this?keep:_NEG
        # --
        parser.add_argument("--qa_load_name0", default=None, type=str)  # loading name (mainly for training)
        parser.add_argument("--qa_load_name", default=None, type=str)  # loading name (mainly for testing)
        parser.add_argument("--qa_save_name", default='zmodel', type=str)  # saving name
        # --

    @staticmethod
    def get_bert(args):
        bert_name, bert_cache = args.bert_model, args.bert_cache
        logging.info(f"Loading pre-trained bert model for ZBert of {bert_name} from {bert_cache}")
        # --
        tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=bert_cache)
        if args.qa_load_name:  # if we later want to load model
            bert_config = AutoConfig.from_pretrained(bert_name)
            model = AutoModel.from_config(bert_config)
            logging.info("No pretrain-loading for bert, really want this?")
        else:
            model = AutoModel.from_pretrained(bert_name, cache_dir=bert_cache)
        # --
        if hasattr(model, "pooler"):  # note: delete unused part!
            model.__delattr__("pooler")
            model.__setattr__('pooler', (lambda x: x))
        # --
        model.eval()  # note: by default set eval!!
        # --
        return tokenizer, model

    # --
    # input conventions: [cls] Question [sep] Context [sep] [pad...]; att_mask>0 for non-pad; ...
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None, ret_dict=False):
        args = self.args
        # --
        # we can totally shrink inputs if there are extra paddings!
        # note: this requires that attention_mask is left-aligned, which seems to be True!!
        if np.prod(attention_mask.shape) > 0:  # not zero
            max_len = (attention_mask>0).float().sum(-1).long().max().item()  # []
            max_len = max(max_len, 10)  # do not truncate too much?
            if max_len < attention_mask.shape[-1]:
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]
                if token_type_ids is not None:
                    token_type_ids = token_type_ids[:, :max_len]
        # --
        # anyway, first pass though bert
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_hid = bert_outputs[0]  # [bs, slen, D]
        logits = self.qa(bert_hid)  # [bs, slen, ??]
        # --
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
        # --
        # which method?
        _NEG = -10000.
        if args.qa_head_type == 'label':
            logits = logits.squeeze(-1)  # [bs, slen]
            if start_positions is not None and end_positions is not None:
                _slen = input_ids.shape[-1]
                _arange_t = torch.arange(_slen).to(start_positions.device)  # [slen]
                _trg_t = ((start_positions.unsqueeze(-1) <= _arange_t)
                          & (_arange_t <= end_positions.unsqueeze(-1))).float()  # [bs, slen]
                _trg_t[:, 0] = 0  # make sure idx0 is 0!!
                _trg_t[(input_ids==self.tokenizer.cls_token_id) | (input_ids==self.tokenizer.sep_token_id)] = 0
                _trg_t[attention_mask<=0.] = 0  # make sure invalud is 0
                # --
                # calculate neg rate
                _pos_count = _trg_t.sum(-1, keepdims=True).clamp(min=1)  # [bs, 1]
                _neg_count = ((_trg_t==0) & (attention_mask>0.)).float().sum(-1, keepdims=True).clamp(min=1)  # [bs, 1]
                _neg_dw = args.qa_label_negratio * _pos_count / _neg_count  # [bs, 1]
                # --
                loss0 = loss_binary(logits, _trg_t, args.qa_label_ls)  # [bs, slen]
                weight0 = torch.where(_trg_t<=0., _neg_dw, torch.ones_like(loss0))  # [bs, slen]
                weight0 *= attention_mask  # mask out invalid ones
                total_loss = (loss0 * weight0).sum() / weight0.sum()  # []
                ret = {'total_loss': total_loss, 'start_logits': None, 'end_logits': None, 'logits': logits}
            else:
                logits[attention_mask<=0.] = _NEG  # mask out invalid ones!
                # --
                t_logits = logits.clone()  # [bs, slen]
                t_logits[t_logits < args.qa_label_pthr] = _NEG  # simply cut them off!
                # --
                log_prob1 = F.logsigmoid(t_logits)  # [bs, slen]
                log_prob0 = F.logsigmoid(-t_logits)  # [bs, slen]
                # note: prob of NULL is the min(log_prob0): if the min-log_prob0 is large, then high chance no-ans
                # -- here '*2' means that for other logits we are adding two things
                start_null = end_null = (log_prob0 * attention_mask).min(-1)[0].unsqueeze(-1) * 2  # [bs, 1]
                start_logits1 = log_prob0[:,:-1] + log_prob1[:,1:]  # [bs, slen-1]
                end_logits1 = log_prob1[:,1:-1] + log_prob0[:,2:]  # [bs, slen-2]
                # --
                start_logits = torch.cat([start_null, start_logits1], dim=-1)  # [bs, slen]
                end_logits = torch.cat([end_null, end_logits1, torch.full_like(end_null, _NEG)], dim=-1)  # [bs, slen]
                ret = {'total_loss': None, 'start_logits': start_logits, 'end_logits': end_logits, 'logits': logits}
        elif args.qa_head_type == 'ptr':  # start and end
            start_logits, end_logits = [z.squeeze(-1) for z in logits.split(1, dim=-1)]  # [bs, slen]
            ret = {'total_loss': None, 'start_logits': start_logits, 'end_logits': end_logits, 'logits': logits}
            if start_positions is not None and end_positions is not None:
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = input_ids.size(-1)  # slen
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)
                # --
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                ret['total_loss'] = total_loss
        else:
            raise NotImplementedError()
        # --
        if ret_dict:
            return ret
        else:
            return (() if ret['total_loss'] is None else (ret['total_loss'], )) + (ret['start_logits'], ret['end_logits'])
        # --

    # decode one instance
    def decode(self, inst, arr_logits):
        t_logits = torch.tensor(arr_logits[:len(inst)])  # [len, ??], just on cpu is fine!
        # --
        args = self.args
        if args.qa_head_type == 'label':
            pthr = args.qa_label_pthr
            # get all valid spans
            _ctx_offset = inst.context_offset
            t_logits = t_logits.squeeze(-1)  # [len]
            t_probs = t_logits.sigmoid().tolist()  # [len]
            # --
            _hit = (t_logits>pthr).tolist()
            valid_spans = []  # [left, right, sum(prob)]
            last_hit = False
            for ii, vv in enumerate(_hit[_ctx_offset:], _ctx_offset):
                if vv:
                    if not last_hit:
                        valid_spans.append([ii, ii, t_probs[ii]])
                    else:  # add it
                        valid_spans[-1][1] = ii
                        valid_spans[-1][-1] += t_probs[ii]
                    last_hit = True
                else:
                    last_hit = False
            # --
            # get the best one!
            if len(valid_spans) == 0:
                return (0, 0)
            else:
                sorted_spans = sorted(valid_spans, key=(lambda x: x[-1]/(x[1]-x[0]+1)), reverse=True)
                best_one = sorted_spans[0]
                return (best_one[0], best_one[1])
            # --
        elif args.qa_head_type == 'ptr':
            _NEG = -1000.
            _ctx_offset = inst.context_offset
            t_logits[1:_ctx_offset] = _NEG  # mask out bad ones
            t_probs = t_logits.softmax(0)  # [len, 2]
            best_left_prob, best_left_idx = [z.item() for z in t_probs[:,0].max(0)]
            t_probs[:best_left_idx, 1] = _NEG
            best_right_prob, best_right_idx = [z.item() for z in t_probs[:,1].max(0)]
            return (best_left_idx, best_right_idx)
        else:
            raise NotImplementedError()
        # --
