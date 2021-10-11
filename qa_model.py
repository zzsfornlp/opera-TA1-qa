#

# qa models
import torch
import logging
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertModel, RobertaModel, XLMRobertaModel

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
        # --

    def save(self, name: str):
        logging.info(f"Save model&args to {name} and {name}.args")
        d = self.state_dict()
        torch.save(d, name)
        torch.save(self.args, f'{name}.args')

    def load(self, name: str):
        logging.info(f"Load model from {name}")
        d = torch.load(name)
        self.load_state_dict(d)

    @staticmethod
    def create_model(args):
        model = QaModel(args)  # first create one!
        if args.qa_load_name:
            model.load(args.qa_load_name)
        return model

    @staticmethod
    def load_model(load_name: str):
        # first load args
        args = torch.load(load_name+".args")
        # then create model (and load inside)
        args.qa_load_name = load_name
        model = QaModel.create_model(args)
        return model

    @staticmethod
    def add_args(parser):
        # add arguments specific to the qa_model!
        parser.add_argument("--bert_model", default='bert-large-cased', type=str)
        parser.add_argument("--bert_cache", default=None, type=str)
        parser.add_argument("--qa_head_type", default='ptr', type=str, choices=['label', 'ptr'])
        parser.add_argument("--qa_load_name", default=None, type=str)  # loading name
        parser.add_argument("--qa_save_name", default='zmodel.m', type=str)  # saving name

    @staticmethod
    def get_bert(args):
        bert_name, bert_cache = args.bert_model, args.bert_cache
        logging.info(f"Loading pre-trained bert model for ZBert of {bert_name} from {bert_cache}")
        # --
        tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=bert_cache)
        mtype = {"bert": BertModel, "roberta": RobertaModel, "xlm": XLMRobertaModel}[
            bert_name.split("/")[-1].split("-")[0]]
        if args.qa_load_name:  # if we later want to load model
            from transformers import AutoConfig
            bert_config = AutoConfig.from_pretrained(bert_name)
            model = mtype(bert_config)
            logging.info("No pretrain-loading for bert, really want this?")
        else:
            model = mtype.from_pretrained(bert_name, cache_dir=bert_cache)
        # --
        if hasattr(model, "pooler"):  # note: delete unused part!
            model.__delattr__("pooler")
        # --
        model.eval()  # note: by default set eval!!
        # --
        return tokenizer, model

    # --
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        args = self.args
        # --
        # anyway, first pass though bert
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_hid = bert_outputs[0]  # [bs, slen, D]
        logits = self.qa(bert_hid)  # [bs, slen, ??]
        # --
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = input_ids.size(-1)  # slen
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)
        # --
        # which method?
        if args.qa_head_type == 'label':
            # TODO(!)
            logits = logits.squeeze(-1)  # [bs, slen]
            if start_positions is not None and end_positions is not None:
                pass
            else:
                log_prob1 = F.logsigmoid(logits)  # [bs, slen]
                log_prob0 = F.logsigmoid(-logits)  # [bs, slen]
                # prob of NULL is the prob of all 0 (inside valid)
                start_null = end_null = (log_prob0 * attention_mask).sum(-1, keepdims=True)  # [bs, 1]
                start_logits = None
        elif args.qa_head_type == 'ptr':  # start and end
            start_logits, end_logits = [z.squeeze(-1) for z in logits.split(1, dim=-1)]  # [bs, slen]
            total_loss = None
            if start_positions is not None and end_positions is not None:
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
            ret = (start_logits, end_logits)
            if total_loss is not None:
                ret = (total_loss, ) + ret
            return ret
        else:
            raise NotImplementedError()
