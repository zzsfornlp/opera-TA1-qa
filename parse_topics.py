#

# parse the "topic_list.txt" file

from typing import List
import sys
import string
from msp2.data.inst import Sent, MyPrettyPrinter, QuestionAnalyzer
from msp2.utils import zlog, zopen, zwarn, Conf, default_json_serializer
from msp2.tools.annotate import AnnotatorStanzaConf, AnnotatorStanza
from nltk.tokenize import TreebankWordTokenizer

def read_tab_file(file: str):
    with zopen(file) as fd:
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
        stanza_conf = AnnotatorStanzaConf.direct_conf(
            stanza_lang='en', stanza_dir=stanza_dir, stanza_processors='tokenize,pos,lemma,depparse'.split(','),
            stanza_use_gpu=False, stanza_input_mode='tokenized')
        self.stanza = AnnotatorStanza(stanza_conf)

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
                    zwarn(f"Hit multiple Xs, ignore the later one: {raw_tokens}")
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
        sent = Sent.create(normed_tokens)
        self.stanza.annotate([sent])  # parse this one!
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
        # --
        if not quite:
            zlog(f"#-- Parse template: {template} ||| {hint}\n"
                 f"=>raw={raw_tokens}\n=>norm={normed_tokens}\n=>q={question_tokens}\n=>ret={final_tokens}")
        # if debug:
        #     breakpoint()
        return sent, final_tokens

    def template2question(self, tsent: Sent, q_widx: int, q_toks: List[str]):
        # note: directly use dep-tree, which seems easier ...
        tree = tsent.tree_dep
        sent_toks = list(tsent.seq_word.vals)  # copy it
        if not str.isupper(sent_toks[0][:2]):  # probably not PROPN
            sent_toks[0] = sent_toks[0][0].lower() + sent_toks[0][1:]  # todo(+N): lowercase anyway ...
        dep_labels = [z.split(":")[0] for z in tree.seq_label.vals]
        dep_chs_lists = tree.chs_lists
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
                q_widx = tree.seq_head.vals[q_widx]-1
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
                    _range = tree.ranges[q0_ch]
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
                    elif tsent.seq_upos.vals[root_widx] == 'VERB' and tsent.seq_lemma.vals[root_widx] is not None:
                        root_lemma, root_word = tsent.seq_lemma.vals[root_widx], sent_toks[root_widx]
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
def postprocess_question(q_toks):
    ts = []
    for t in q_toks:
        if t.lower() in [z.lower() for z in ["SARS-CoV-2", "COVID-19", "virus"]]:
            t = f"{t} or coronavirus"
        ts.append(t)
    return " ".join(ts)

def main(input_file='', output_file='', stanza_dir=''):
    parser = TemplateParser(stanza_dir)
    if input_file == '':
        while True:
            in_line = input(">> ")
            in_line = in_line.strip()
            if len(in_line) == 0:
                break
            template, subtopic = (in_line.split("|||") + [''])[:2]
            if 'X' in template:
                sent, res_toks = parser.parse_template(template, hint=subtopic)
            else:
                tokens = parser.word_toker.tokenize(template)
                sent = Sent.create(tokens)
                parser.stanza.annotate([sent])  # parse this one!
                res_toks = QuestionAnalyzer().question2template(sent)
            # --
            zlog(MyPrettyPrinter.str_fnode(sent, sent.tree_dep.fnode))
            zlog(f"{sent.get_text()}")
            zlog(f"{' '.join(res_toks)}")
    else:
        final_res = {"topics": {}, "subtopics": {}}
        tabs = read_tab_file(input_file)
        for v in tabs:
            _, _q_toks = parser.parse_template(v['template'], hint=v['subtopic'], quite=True)
            if v['id'] in SHORTCUTS:
                sc_toks = SHORTCUTS[v['id']].split()
                zlog(f"{'Hit' if _q_toks == sc_toks else 'Miss'}: {SHORTCUTS[v['id']]} <-> {_q_toks}")
                _q_toks = sc_toks  # replace it anyway!
            _q_toks = _q_toks + ["?"]  # add question mark!
            v['question'] = postprocess_question(_q_toks)
            # --
            final_res['subtopics'][v['id']] = v
            if v['topic'] not in final_res['topics']:
                final_res['topics'][v['topic']] = []
            final_res['topics'][v['topic']].append(v['id'])
        if output_file:
            default_json_serializer.to_file(final_res, output_file, indent=4)
    # --

# python3 parse_topics.py
if __name__ == '__main__':
    main(*sys.argv[1:])
