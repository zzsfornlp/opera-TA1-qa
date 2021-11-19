#

# parse the "topic_list.txt" file

from typing import List
import sys
import string
import logging
import json
import stanza
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

class StanzaParser:
    def __init__(self, stanza_dir: str):
        common_kwargs = {"lang": 'en', "use_gpu": False, 'dir': stanza_dir}
        self.parser = stanza.Pipeline(
            processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True, **common_kwargs)

    def parse(self, tokens: List[str]):
        res = self.parser([tokens])
        words = res.sentences[0].words
        if len(words) != len(tokens):
            logging.warning(f"Sent length mismatch: {len(words)} vs {len(tokens)}")
        ret = {
            'text': [w.text for w in words],
            'lemma': [w.lemma for w in words], 'upos': [w.upos for w in words],
            'head': [w.head for w in words], 'deprel': [w.deprel for w in words],
        }
        return ret

def read_tab_file(file: str):
    with open(file) as fd:
        headline = fd.readline()
        head_fields = [z.lower() for z in headline.rstrip().split("\t")]
        ret = []
        for line in fd:
            fields = line.rstrip().split("\t")
            # assert len(fields) == len(head_fields)
            ret.append({k: v for k,v in zip(head_fields, fields)})
        return ret

class TemplateParser:
    def __init__(self, stanza_dir: str):
        self.word_toker = TreebankWordTokenizer()
        self.parser = StanzaParser(stanza_dir)

    def parse_template(self, template: str, hint: str = None, quite=False):
        # step 1: normalize text
        raw_tokens = self.word_toker.tokenize(template)
        x_widx = None
        x_mod = None
        normed_tokens = []
        for tok in raw_tokens:
            if tok.lower() == 'x' or tok.lower().endswith("-x"):
                if tok.lower() == 'x':
                    _toks = ['X']
                else:
                    _toks = [tok[:-2].split("/")[0], 'X']
                # --
                if x_widx is not None:
                    logging.warning(f"Hit multiple Xs, ignore the later one: {raw_tokens}")
                else:
                    x_widx = len(normed_tokens) + len(_toks) - 1
                    if tok.lower().endswith("-x"):
                        x_mod = _toks[0]
            elif tok == '/':  # by itself
                _toks = ['or']
            else:
                _toks = [tok]
            normed_tokens.extend(_toks)
        # step 2: parse it!
        if normed_tokens[-1] != '.':  # add last dot if there are not
            normed_tokens = normed_tokens + ["."]
        sent = self.parser.parse(normed_tokens)  # parse this one!
        # step 3: template to question
        q_toks = ['what']
        if hint is not None:
            if 'who/what' in hint.lower():
                q_toks = ['who', 'or', 'what']
            elif 'who' in hint.lower():
                q_toks = ['who']
        question_tokens = self.template2question(sent, x_widx, q_toks)
        # step 4: final change!
        final_tokens = list(question_tokens)
        if len(question_tokens)>3 and question_tokens[0].lower() in ['on', 'in', 'at'] and question_tokens[1].lower() == 'what':
            if question_tokens[2].lower() in ['location', 'place']:
                final_tokens = ["Where"] + question_tokens[3:]
            elif question_tokens[2].lower() in ['time', 'date']:
                final_tokens = ["When"] + question_tokens[3:]
        # step 5: more templates with simple negation
        more_templates = self.create_more_templates(sent, final_tokens)
        # --
        if not quite:
            logging.info(f"#-- Parse template: {template} ||| {hint}\n"
                 f"=>raw={raw_tokens}\n=>norm={normed_tokens}\n=>q={question_tokens}\n=>ret={final_tokens}")
        # if debug:
        #     breakpoint()
        return sent, more_templates

    def simple_negation(self, tokens):
        # todo: really bad replacements here ...
        REPL_MAP0 = {
            "can": "can not", "may": "may not",
            "is": "isn't", "was": "wasn't", "are": "aren't", "were": "weren't", "will": "won't",
            "did": "didn't", "does": "doesn't", "do": "don't",
        }
        REPL_MAPS = {
            z: z[:-1] for z in ["transmits", "transfers", "destroys", "prevents", "cures", "shortens", "reduces"]}  # does not
        REPL_MAPD = {"created": "create", "funded": "fund", "enacted": "enact", "received": "receive"}  # did not
        # --
        ret = []
        hit_neg = False
        for t in tokens:
            if not hit_neg:
                if t in REPL_MAP0:
                    ret.extend(REPL_MAP0[t].split())
                    hit_neg = True
                    continue  # skip this token!
                elif t in REPL_MAPS:
                    hit_neg = True
                    ret.extend("does not".split())
                elif t in REPL_MAPD:
                    hit_neg = True
                    ret.extend("did not".split())
            # add token
            if t in REPL_MAPS:
                ret.extend(REPL_MAPS[t].split())
            elif t in REPL_MAPD:
                ret.extend(REPL_MAPD[t].split())
            else:
                ret.append(t)
        # --
        return ret

    def create_more_templates(self, sent, question_tokens):
        ret = {
            "template_pos": sent['text'],
            "question_pos": question_tokens,
            "template_neg": self.simple_negation(sent['text']),
            "question_neg": self.simple_negation(question_tokens),
        }
        return ret

    def get_chs_lists(self, cur_heads):
        chs = [[] for _ in range(len(cur_heads) + 1)]
        for m, h in enumerate(cur_heads):  # note: already sorted in left-to-right
            chs[h].append(m)  # note: key is hidx, value is midx
        return chs

    def get_ranges(self, cur_heads):
        ranges = [[z, z] for z in range(len(cur_heads))]
        for m in range(len(cur_heads)):
            cur_idx = m
            while cur_idx >= 0:
                ranges[cur_idx][0] = min(m, ranges[cur_idx][0])
                ranges[cur_idx][1] = max(m, ranges[cur_idx][1])
                cur_idx = cur_heads[cur_idx] - 1  # offset -1
        return ranges

    def template2question(self, sent, q_widx: int, q_toks: List[str]):
        sent_toks = list(sent['text'])  # copy it to modify!
        if not str.isupper(sent_toks[0][:2]):  # probably not PROPN
            sent_toks[0] = sent_toks[0][0].lower() + sent_toks[0][1:]  # todo(+N): lowercase anyway ...
        dep_labels = [z.split(":")[0] for z in sent['deprel']]
        dep_heads = sent['head']
        dep_chs_lists = self.get_chs_lists(dep_heads)  # [1+m]: list of chidren
        dep_ranges = self.get_ranges(dep_heads)  # [m]: left&right span boundary
        sent_upos, sent_lemma = sent['upos'], sent['lemma']
        # --
        # note: some (4 in prac) may suffer from parsing errors (center verb not detected), but can be fixed simply:
        if q_widx == 0:  # X ... -> W?? ...
            final_toks = q_toks + sent_toks[1:]
        elif q_widx == 1:  # ?? X
            final_toks = q_toks + [sent_toks[0]] + sent_toks[2:]
        else:
            # --
            # construct Q-NP.
            NP_DEPS = {'nmod', 'nummod', 'acl', 'amod', 'case', 'compound'}
            NP_IG_DEPS = {'det', 'clf', 'fixed', 'flat', 'goeswith', 'orphan', 'reparandum', 'dep'}
            NP_DEPS.update(NP_IG_DEPS)
            q0_toks = []
            q0_put = False
            orig_q_widx = q_widx
            if dep_labels[q_widx] == 'compound':  # up one layer!
                q_widx = dep_heads[q_widx]-1
            for q0_ch in sorted(dep_chs_lists[1+q_widx] + [q_widx]):
                if q0_ch == orig_q_widx:
                    continue  # ignore X
                if q0_ch == q_widx:
                    if not q0_put:
                        q0_toks.extend(q_toks)
                        q0_put = True
                    q0_toks.append(sent_toks[q_widx])
                else:
                    _ch_lab = dep_labels[q0_ch]
                    if _ch_lab not in NP_DEPS:
                        continue
                    if _ch_lab in NP_IG_DEPS:
                        continue  # ignore things
                    if dep_labels[q0_ch] not in ['case']:
                        if not q0_put:
                            q0_toks.extend(q_toks)
                            q0_put = True
                    _range = dep_ranges[q0_ch]
                    q0_toks.extend(sent_toks[_range[0]:_range[1]+1])
            if not q0_put:
                q0_toks.extend(q_toks)
            # ===
            def _get_toks(_tmp_i: int, _repl, _ch_set=None):
                if _tmp_i == q_widx: return _repl
                _tmp_chs = [z for z in dep_chs_lists[1+_tmp_i] if _ch_set is None or dep_labels[z] in _ch_set]
                _left = sum([_get_toks(_c, _repl) for _c in _tmp_chs if _c<_tmp_i], [])
                _right = sum([_get_toks(_c, _repl) for _c in _tmp_chs if _c>_tmp_i], [])
                return _left + [sent_toks[_tmp_i]] + _right
            # ===
            # --
            # look at root
            root_widx = dep_chs_lists[0][0]
            # check subj, obj, obl, head and decide whether reverse ...
            root_chs = [c for c in sorted(dep_chs_lists[1+root_widx]) if dep_labels[c] not in NP_DEPS]
            if q_widx < root_widx:  # note: no need to change too much
                root_ch_toks = [_get_toks(c, q0_toks) for c in root_chs]
                final_toks = sum([ts for c,ts in zip(root_chs,root_ch_toks) if c<root_widx],[]) \
                             + _get_toks(root_widx, [], NP_DEPS) + sum([ts for c,ts in zip(root_chs,root_ch_toks) if c>root_widx],[])
            else:  # need to change
                root_ch_toks = [_get_toks(c, []) for c in root_chs]
                root_self_toks = _get_toks(root_widx, [], NP_DEPS)  # note: exclude self as q!
                subj_idxes = [ii for ii,ii2 in enumerate(root_chs) if dep_labels[ii2] in ['nsubj', 'csubj']]
                auxcop_idxes = [ii for ii,ii2 in enumerate(root_chs) if dep_labels[ii2] in ['aux', 'cop']]
                if len(subj_idxes) > 0:
                    subj_i0 = subj_idxes[0]
                    if len(auxcop_idxes) > 0:  # simply move that one to the front
                        ac_i0 = auxcop_idxes[0]
                        root_ch_toks[subj_i0] = root_ch_toks[ac_i0] + root_ch_toks[subj_i0]
                        root_ch_toks[ac_i0] = []
                    elif sent_upos[root_widx] == 'VERB' and sent_lemma[root_widx] is not None:
                        root_lemma, root_word = sent_lemma[root_widx], sent_toks[root_widx]
                        if root_word == root_lemma:
                            _extra = 'do'
                        elif root_word == root_lemma + 's' or root_word == root_lemma + 'es' \
                                or root_word in ['has']:  # special ones!
                            _extra = 'does'
                        else:
                            _extra = 'did'
                        root_ch_toks[subj_i0] = [_extra] + root_ch_toks[subj_i0]
                        root_self_toks = [root_lemma]
                    else:
                        pass  # note: strange pattern, simply no change!
                final_toks = q0_toks + sum([ts for c,ts in zip(root_chs,root_ch_toks) if c<root_widx],[]) \
                             + root_self_toks + sum([ts for c,ts in zip(root_chs,root_ch_toks) if c>root_widx],[])
        # --
        if len(final_toks) > 0 and len(final_toks[0]) > 0:  # uppercase!
            final_toks[0] = final_toks[0][0].upper() + final_toks[0][1:]
        if len(final_toks) > 0 and final_toks[-1] == '.':
            final_toks = final_toks[:-1]
        final_toks = final_toks + ["?"]
        return final_toks

# --
SHORTCUTS = {
    # first 31 from the practice
    'C303': 'Who can catch COVID-19',
    'C304': 'What transmits or transfers COVID-19',
    'C306': 'What destroys COVID-19',
    'C307': 'What prevents COVID-19',
    'C308': 'What cures COVID-19',
    'C311': 'What is SARS-CoV-2',
    'C312': 'What is immunity against COVID-19',
    'C313': 'What is the potency of SARS-CoV-2',
    'C314': 'Where is wearing masks required',
    'C315': 'Where is wearing masks recommended',
    'C317': 'Where is wearing masks necessary',
    'C318': 'What negative effect does wearing masks have',
    'C319': 'What animal is associated/involved with the origin of COVID-19',
    'C322': 'Who created SARS-CoV-2',
    'C323': 'Who funded SARS-CoV-2 development',
    'C324': 'Where did SARS-CoV-2 originate',
    'C325': 'Where was SARS-CoV-2 created',
    'C326': 'Where did the first case of COVID-19 occur',
    'C327': 'When did the first case of COVID-19 occur',
    'C328': 'When was SARS-CoV-2 created',
    'C329': 'Who or what was the target of the virus',
    'C331': 'What government enacted population restrictions related to COVID-19',
    'C332': 'What government enacted contact tracing related to COVID-19',
    'C333': 'What government enacted patient containment actions related to COVID-19',
    'C335': 'What government is withholding cures for COVID-19',
    'C343': 'What is an effective treatment for COVID-19',
    'C344': 'What treatment shortens the length of COVID-19 infection',
    'C346': 'What treatment prevents death or reduces the chance of death from COVID-19 infection',
    'C348': 'What treatment received emergency use approval',
    'C349': 'What treatment is a safe treatment for COVID-19',
    'C350': 'What generally safe medication is unsafe for COVID-19 patients',
}
# --

# note: special post-processing!
def postprocess_tokens(toks):
    ts = []
    for t in toks:
        if t.lower() in [z.lower() for z in ["SARS-CoV-2", "COVID-19", "virus"]]:
            ts.extend(f"{t} or coronavirus".split())
        else:
            ts.append(t)
    return ts

def main(input_file='', output_file='', stanza_dir=''):
    # --
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    # --
    parser = TemplateParser(stanza_dir)
    final_res = {"topics": {}, "subtopics": {}}
    tabs = read_tab_file(input_file)
    detoker = TreebankWordDetokenizer()
    for v in tabs:
        sent_parse, seqs = parser.parse_template(v['template'], hint=v['subtopic'], quite=True)
        if v['id'] in SHORTCUTS:
            sc_toks = SHORTCUTS[v['id']].split()
            if seqs['question_pos'] != sc_toks + ['?']:
                logging.warning(f"Unmatched question: {SHORTCUTS[v['id']]} <-> {seqs['question_pos']}")
                seqs['question_pos'] = sc_toks  # replace it anyway!
        # postprocessing
        seqs = {k: postprocess_tokens(v) for k,v in seqs.items()}
        # --
        v['question'] = detoker.detokenize(seqs['question_pos'])
        v['parse'] = sent_parse
        v['seqs'] = seqs
        # --
        final_res['subtopics'][v['id']] = v
        if v['topic'] not in final_res['topics']:
            final_res['topics'][v['topic']] = []
        final_res['topics'][v['topic']].append(v['id'])
    if output_file:
        with open(output_file, 'w') as fd:
            json.dump(final_res, fd, indent=4)
    # --

# python3 parse_topics.py
if __name__ == '__main__':
    main(*sys.argv[1:])
