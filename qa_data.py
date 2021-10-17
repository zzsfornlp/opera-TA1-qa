#

import os
from typing import List
from collections import defaultdict, Counter
import string
import logging
import torch
import json
import copy
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
        self.device = None

    @property
    def sub_tokenizer(self):
        return self.g_subtoker.tokenizer
# --
# note: for simplicity, just make it global ...
GR = GlobalResources()
def set_gr(tokenizer, gpuid):
    GR.g_subtoker = SubToker(tokenizer)
    if gpuid >= 0:
        GR.device = torch.device(gpuid)
    else:
        GR.device = torch.device('cpu')
# --

# --
# a piece of text that we can trace the char positions
class TextPiece:
    def __init__(self, text: str, **info_kwargs):
        self.orig_text = text
        # --
        # first split sent/word!
        self.tokens, self.token_spans = GR.g_toker.tokenize(text)
        self.cmap = [None] * len(text)
        for tid, (a,b) in enumerate(self.token_spans):
            self.cmap[a:b] = [tid] * (b-a)
        # --
        # then do subword tokenizer
        self.subtokens, self.sub2tid = GR.g_subtoker.sub_tokenize(self.tokens)
        self.subtoken_ids = GR.sub_tokenizer.convert_tokens_to_ids(self.subtokens)
        # --
        self.info = {}
        self.info.update(info_kwargs)
        # --

    def __repr__(self):
        return f"Text({self.orig_text[:100]} ...)"

    # char span to token span
    def cspan2tspan(self, char_start: int, char_length: int):
        tids = sorted(set([self.cmap[c] for c in range(char_start, char_start+char_length) if self.cmap[c] is not None]))
        if len(tids) == 0:
            return None
        else:
            return tids[0], tids[-1]-tids[0]+1
        # --

    @staticmethod
    def merge_pieces(pieces, **info_kwargs):
        ret = TextPiece("", **info_kwargs)
        ret.cmap = ret.tokens = ret.token_spans = None
        ret.subtokens = ret.sub2tid = None
        # --
        # make a fake one for forwarding
        ret.orig_text = ""
        ret.subtoken_ids = sum([z.subtoken_ids for z in pieces], [])
        return ret

class TextSpan:
    def __init__(self, text_piece: TextPiece, start: int, end: int):
        self.text_piece = text_piece
        self.start = start
        self.end = end

    @staticmethod
    def create_from_subspan(text_piece: TextPiece, sub_start: int, sub_end: int):
        tok_start, tok_end = text_piece.sub2tid[sub_start], text_piece.sub2tid[sub_end-1] + 1
        return TextSpan(text_piece, tok_start, tok_end)

    def get_orig_str(self):
        if self.start >= self.end:
            return ""
        else:
            span_left = self.text_piece.token_spans[self.start]
            span_right = self.text_piece.token_spans[self.end-1]
            return self.text_piece.orig_text[span_left[0]:span_right[1]]

# --
# actual qa instances

# one pair of question and context
class QaInstance:
    def __init__(self, context: TextPiece, question: TextPiece, qid: str):
        self.qid = qid
        # construct the inputs
        self.context = context
        self.question = question
        self.input_ids, self.type_ids, self.context_offset = QaInstance.construct_qc_pair(context, question)
        # --

    def __repr__(self):
        return f"QA({self.question}, {self.context})"

    def __len__(self):
        return len(self.input_ids)  # full length

    # norm answer from subtok idxes to token span
    def get_answer_span(self, p_left: int, p_right: int):
        if p_left == 0 and p_right == 0:  # no answer
            return TextSpan(self.context, 0, 0)
        else:
            c_left, c_right = max(0, p_left-self.context_offset), max(0, p_right-self.context_offset)
            return TextSpan.create_from_subspan(self.context, c_left, c_right+1)

    @staticmethod
    def construct_qc_pair(context: TextPiece, question: TextPiece):
        _sub_tokenizer = GR.sub_tokenizer
        _limit_q, _limit_full = GR.args_max_query_length, GR.args_max_seq_length
        input_ids, type_ids = [_sub_tokenizer.cls_token_id], [0]
        # add question
        if len(question.subtoken_ids) > _limit_q:
            logging.warning(f"Truncate question ({len(question.subtoken_ids)} > {_limit_q}): {question.orig_text[:80]} ...")
        q_ids = question.subtoken_ids[:_limit_q]
        input_ids.extend(q_ids + [_sub_tokenizer.sep_token_id])
        type_ids.extend([0] * (len(q_ids) + 1))
        context_offset = len(input_ids)
        # add context
        budget = _limit_full - 1 - len(input_ids)
        if len(context.subtoken_ids) > budget:
            logging.warning(f"Truncate context ({len(context.subtoken_ids)} > {budget}): {context.orig_text[:80]} ...")
        c_ids = context.subtoken_ids[:budget]
        input_ids.extend(c_ids + [_sub_tokenizer.sep_token_id])
        type_ids.extend([1] * (len(c_ids) + 1))
        return input_ids, type_ids, context_offset

    @staticmethod
    def batch_insts(insts, device=None):
        # --
        if device is None:
            device = GR.device
        # --
        _shape = [len(insts), max(len(z.input_ids) for z in insts)]  # [bs, max-len]
        input_ids, attention_mask, token_type_ids = \
            torch.zeros(_shape).long()+GR.sub_tokenizer.pad_token_id, torch.zeros(_shape).float(), torch.zeros(_shape).long()
        for bidx, inst in enumerate(insts):
            _len = len(inst.input_ids)
            input_ids[bidx, :_len] = torch.tensor(inst.input_ids)
            attention_mask[bidx, :_len] = 1.
            token_type_ids[bidx, :_len] = torch.tensor(inst.type_ids)
        # --
        input_ids, attention_mask, token_type_ids = [z.to(device) for z in [input_ids, attention_mask, token_type_ids]]
        return input_ids, attention_mask, token_type_ids

# --
# csr related

class CsrDoc:
    def __init__(self, csr_path: str):
        # read csr
        with open(csr_path) as fd:
            json_doc = json.load(fd)
        self.orig_doc = copy.deepcopy(json_doc)  # keep a deep-copy for output
        # --
        # parse it!
        self.doc_id = os.path.basename(csr_path)
        self.frame_id_infix = ""
        _csr_suffix = ".csr.json"
        if self.doc_id.endswith(_csr_suffix):
            self.doc_id = self.doc_id[:-len(_csr_suffix)]
        # --
        # record all frames
        _id2frame = {}  # @id -> one
        for ff in json_doc['frames']:
            if ff['@id'] in _id2frame:
                logging.warning(f"Ignoring repeated @id frame: {ff}")  # this should not happen!
                continue
            if ff['@type'] == 'document' and ff['@id'] != f"data:{self.doc_id}":  # check docid
                logging.warning(f"Mismatched doc-id: {self.doc_id} vs {ff}")
            if ff['@type'] == 'sentence' and not self.frame_id_infix:  # find an infix
                self.frame_id_infix = '-'.join(ff['@id'].split('-')[1:-1])  # "{doc_id}-text-cmu-{time_stamp}"
            _id2frame[ff['@id']] = ff
        # --
        # parse sentences
        _sents = []  # List[TextPiece]
        _id2sent = {}  # @id -> TextPiece
        _cur_tok_offset = 0
        for ff in json_doc['frames']:
            if ff['@type'] == 'sentence':
                s = TextPiece(text=ff['provenance']['text'])
                s.info.update({'id': ff['@id'], 'tok_offset': _cur_tok_offset})
                _sents.append(s)
                _id2sent[ff['@id']] = s
                _cur_tok_offset += len(s.tokens)
        # --
        # prepare useful entities and events
        self.claim_events = defaultdict(list)  # sid -> claim events
        self.cand_events = defaultdict(list)  # sid -> other candidate events
        self.cand_entities = defaultdict(list)  # sid -> candidate entities
        failed_spans = Counter()
        for ff in json_doc['frames']:
            if ff['@type'] in ['entity_evidence', 'event_evidence']:
                _provenance = ff['provenance']
                _char_posi = self.get_provenance_span(ff)
                _tok_posi = None
                if _char_posi is not None:
                    _tok_posi = _id2sent[_provenance['parent_scope']].cspan2tspan(*_char_posi)
                if _tok_posi is None:
                    # logging.warning(f"Cannot find head tok_posi for frame: {ff['@id']} {_provenance}")
                    failed_spans[ff['@type']] += 1
                    continue
                ff['tok_posi'] = _tok_posi  # token-span inside sentence!
                # --
                if ff['@type'] == 'entity_evidence':
                    self.cand_entities[_provenance['parent_scope']].append(ff)
                else:
                    if ff['interp'].get('info', {}).get('sip', False):
                        self.claim_events[_provenance['parent_scope']].append(ff)
                    else:
                        self.cand_events[_provenance['parent_scope']].append(ff)
                # --
        if len(failed_spans) > 0:
            logging.warning(f"Cannot find head tok_posi for {self.doc_id}: {failed_spans}")
        # --
        # remember these
        self.id2frame = _id2frame
        self.sents = _sents
        self.id2sent = _id2sent
        self.cf_frames = []  # cf-frames
        # --

    def get_provenance_span(self, ff, try_base=True, try_head=True):
        _provenance = ff['provenance']
        if try_base and 'base_provenance' in _provenance:
            _provenance = _provenance['base_provenance']
        if try_head:
            if 'head_span_start' in _provenance and 'head_span_length' in _provenance:  # use head if possible
                return _provenance['head_span_start'], _provenance['head_span_length']
            else:
                return None
        else:
            return _provenance['start'], _provenance['length']

    def add_cf(self, subtopic: dict, x: str, x_score: float, claim_evt: str):
        # here simply add id
        _id = f"data:cf-{self.frame_id_infix}-{len(self.cf_frames)}"
        # some checking
        # assert self.id2frame[x]['@type'] in ['entity_evidence', 'event_evidence']
        # assert claim_evt is None or self.id2frame[claim_evt]['@type']=='event_evidence' \
        #        and self.id2frame[claim_evt]['interp']['info']['sip']
        # --
        # also include text for easier checking
        x_text = self.id2frame[x]['provenance']['text']
        claim_evt_text = None if claim_evt is None else self.id2frame[claim_evt]['provenance']['text']
        res = {
            '@id': _id, '@type': 'claim_frame_evidence', 'component': 'opera.cf.qa',
            'subtopic': subtopic, 'x': x, 'x_score': x_score, 'x_text': x_text,
            'claim_evt': claim_evt, 'claim_evt_text': claim_evt_text,
        }
        self.cf_frames.append(res)
        # --

    def write_output(self, output_path: str):
        with open(output_path, 'w') as fd:
            res = copy.deepcopy(self.orig_doc)
            res['frames'].extend(self.cf_frames)
            json.dump(res, fd, indent=2, ensure_ascii=False)
        # --
