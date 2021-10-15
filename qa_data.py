#

import string
import logging
import torch
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, PunktSentenceTokenizer

# --
# word tokenizer
class NTokenizer:
    def __init__(self):
        self.word_toker = TreebankWordTokenizer()
        self.sent_toker = PunktSentenceTokenizer()
        # --

    def tokenize(self, text: str):
        # first split sent
        sent_spans = list(self.sent_toker.span_tokenize(text))
        sents = [text[a:b] for a,b in sent_spans]
        # then split tokens
        char2posi = [None] * len(text)  # [L_char]
        all_tokens = []  # [L_tok]
        all_token_spans = []  # [L_tok]
        mark_eos = []  # [L_tok]
        for sid, sent in enumerate(sents):
            if len(mark_eos) > 0:
                mark_eos[-1] = True
            # --
            tok_spans = list(self.word_toker.span_tokenize(sent))
            tokens = [sent[a:b] for a,b in tok_spans]
            for ii, (a, b) in enumerate(tok_spans):
                _offset = sent_spans[sid][0]
                _s0, _s1 = _offset+a, _offset+b
                char2posi[_s0:_s1] = [len(all_tokens)] * (b - a)
                all_tokens.append(tokens[ii])
                all_token_spans.append((_s0, _s1))
                mark_eos.append(False)
        if len(mark_eos) > 0:
            mark_eos[-1] = True
        # return all_tokens, all_token_spans, char2posi, mark_eos
        return all_tokens, all_token_spans

# --
class SubToker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # --
        self.is_roberta_like = any(z in tokenizer.__class__.__name__.lower() for z in ['roberta', 'gpt2', 'bart'])
        self.punct_set = set(string.punctuation)
        # --

    def sub_tokenize(self, tokens):
        subtokens, sub2tid = [], []
        for tid, tok in enumerate(tokens):
            add_space = self.is_roberta_like and not all((c in self.punct_set) for c in tok)
            cur_subtoks = self.tokenizer.tokenize((" "+tok) if add_space else tok)
            # delete special ones!!
            if len(cur_subtoks) > 0 and cur_subtoks[0] in ['▁', 'Ġ']:  # for xlmr and roberta
                cur_subtoks = cur_subtoks[1:]
            # in some cases, there can be empty strings -> put the original word
            if len(cur_subtoks) == 0:
                cur_subtoks = [tok]
            # add
            subtokens.extend(cur_subtoks)
            sub2tid.extend([tid] * len(cur_subtoks))
        # --
        return subtokens, sub2tid

# --
# note: global resources for convenience
class GlobalResources:
    def __init__(self):
        self.g_toker = NTokenizer()
        self.g_subtoker: SubToker = None
        self.args_max_seq_length = 384  # maximum full length to the model
        self.args_max_query_length = 64  # maximum length for the query

    @property
    def sub_tokenizer(self):
        return self.g_subtoker.tokenizer
# --
GR = GlobalResources()
def set_gr(tokenizer):
    GR.g_subtoker = SubToker(tokenizer)
# --

# --
# a piece of text that we can trace the char positions
class TextPiece:
    def __init__(self, text: str):
        self.orig_text = text
        # --
        # first split sent/word!
        self.tokens, self.token_spans = GR.g_toker.tokenize(text)
        # --
        # then do subword tokenizer
        self.subtokens, self.sub2tid = GR.g_subtoker.sub_tokenize(self.tokens)
        self.subtoken_ids = GR.sub_tokenizer.convert_tokens_to_ids(self.subtokens)
        # --

# --
# actual qa instances

# one pair of question and context
class QaInstance:
    def __init__(self, context: TextPiece, question: TextPiece, qid: str):
        self.qid = qid
        # construct the inputs
        self.input_ids, self.type_ids, self.context_offset = QaInstance.construct_qc_pair(context, question)
        # --

    # extract answer from subtok idxes
    def extract_answer(self, start: int, end: int):
        raise NotImplementedError()

    @staticmethod
    def construct_qc_pair(context: TextPiece, question: TextPiece):
        _sub_tokenizer = GR.sub_tokenizer
        _limit_q, _limit_full = GR.args_max_query_length, GR.args_max_seq_length
        input_ids, type_ids = [_sub_tokenizer.cls_token_id], [0]
        # add question
        if len(question.subtoken_ids) > _limit_q:
            logging.warning(f"Truncate question ({len(question.subtoken_ids)} > {_limit_q}): {question.orig_text}")
        q_ids = question.subtoken_ids[:_limit_q]
        input_ids.extend(q_ids + [_sub_tokenizer.sep_token_id])
        type_ids.extend([0] * (len(q_ids) + 1))
        context_offset = len(input_ids)
        # add context
        budget = _limit_full - 1 - len(input_ids)
        if len(context.subtoken_ids) > budget:
            logging.warning(f"Truncate context ({len(context.subtoken_ids)} > {budget}): {context.orig_text}")
        c_ids = context.subtoken_ids[:budget]
        input_ids.extend(c_ids + [_sub_tokenizer.sep_token_id])
        type_ids.extend([1] * (len(c_ids) + 1))
        return input_ids, type_ids, context_offset

    @staticmethod
    def batch_insts(insts, device=None):
        _shape = [len(insts), max(len(z.input_ids) for z in insts)]  # [bs, max-len]
        input_ids, attention_mask, token_type_ids = \
            torch.full(_shape, GR.sub_tokenizer.pad_token_id).long(), torch.zeros(_shape).float, torch.zeros(_shape).long()
        for bidx, inst in enumerate(insts):
            _len = len(inst.input_ids)
            input_ids[bidx, :_len] = inst.input_ids
            attention_mask[bidx, :_len] = 1.
            token_type_ids[bidx, :_len] = inst.type_ids
        # --
        if device is not None:
            input_ids, attention_mask, token_type_ids = [z.to(device) for z in [input_ids, attention_mask, token_type_ids]]
        return input_ids, attention_mask, token_type_ids
