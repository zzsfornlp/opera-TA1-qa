#

# qa models

from torch import nn
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertModel, RobertaModel, XLMRobertaModel

# --
class QaHead(nn.Module):
    def __init__(self, args, enc_dim: int):
        super().__init__()
        # --
        if args.qa_head == 'label':
            output_dim = 1
        elif args.qa_head == 'ptr':
            output_dim = 2
        else:
            raise NotImplementedError()
        # --
        self.head = nn.Linear(enc_dim, output_dim)
        # --

# --
class QaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # --
        self.tokenizer, self.bert = QaModel.get_bert(args)
        self.qa = QaHead(args, self.bert.config.hidden_size)
        # --

    @staticmethod
    def get_bert(args):
        bert_name, bert_cache = args.bert_model, args.bert_cache
        logging.info(f"Loading pre-trained bert model for ZBert of {bert_name} from {bert_cache}")
        # --
        tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=bert_cache)
        mtype = {"bert": BertModel, "roberta": RobertaModel, "xlm": XLMRobertaModel}[
            bert_name.split("/")[-1].split("-")[0]]
        if args.load_name:  # if we later want to load model
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
