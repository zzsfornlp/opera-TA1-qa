#

# make a topic-list from all the queries

import sys
import os
from collections import Counter

# --
def printing(s: str): print(s, file=sys.stderr)

# --
# from "parse_cond7.py"
def topic2template(topic: str):
    tokens = topic.split(' ')
    if tokens[0].lower() in ['what', 'who']:
        tokens[0] = 'X'
    elif tokens[0].lower() in ['when', 'where']:
        if tokens[1] in ['is', 'are']:
            tokens = tokens[2:] + [tokens[1], 'at', 'X']
        else:
            tokens = tokens[1:] + ['at', 'X']
    elif tokens[0].lower() == 'why':
        if tokens[1] in ['is', 'are']:
            tokens = tokens[2:] + [tokens[1], 'because', 'X']
        else:
            tokens = tokens[1:] + ['because', 'X']
    elif tokens[0].lower() == 'how':
        if tokens[1] in ['is', 'are']:
            tokens = tokens[2:] + [tokens[1], 'is', 'by', 'X']
        else:
            tokens = tokens[1:] + ['by', 'X']
    else:
        tokens[0] = tokens[0].lower()
        tokens = ['X', 'are'] + tokens
    template = ' '.join(tokens)
    return template
# --

def read_query(file: str):
    cc = Counter()
    with open(file) as fd:
        ret = []
        for line in fd:
            if line.strip() != "":
                fields = [z.strip() for z in line.rstrip().split("\t")]
                if len(fields) >= 4:
                    ret.append({'id': fields[0], 'topic': fields[1], 'subtopic': fields[2], 'template': fields[3]})
                    cc['query_full'] += 1
                else:  # we have a cond7 query (put topic & subtopic as the same!)
                    ret.append({'id': fields[0], 'topic': fields[1], 'subtopic': fields[1], 'template': topic2template(fields[1])})
                    cc['query_cond7'] += 1
    # --
    printing(f"Read queries from {file}: {cc}")
    return ret

# --
def main(query_dir: str, output_file: str):
    # read all the queries
    all_queries = []
    for cond in [5, 6, 7]:
        ff = os.path.join(query_dir, f'Condition{cond}', 'topics.tsv')
        if os.path.exists(ff):
            all_queries.extend(read_query(ff))
        else:
            printing(f"Potential error? Cannot find query file: {ff}")
    # change to subtopic-id for later processing
    cc = Counter()
    added_queries = []
    topic_counts = Counter()
    with open(output_file, 'w') as fd:
        fd.write('\t'.join(['ID', 'subtopic', 'topic', 'Template']) + '\n')  # headline
        for one in all_queries:
            cc['all'] += 1
            if any(z==one for z in added_queries):  # duplicate! note: simply check each one!
                cc['all_duplicate'] += 1
                continue
            cc['all_ok'] += 1
            added_queries.append(one)
            _id = one["id"]
            if topic_counts[one['topic']] > 0:
                _id = one["id"] + "." + str(topic_counts[one['topic']])
            topic_counts[one['topic']] += 1
            fd.write("\t".join([_id, one['subtopic'], one['topic'], one['template']]) + "\n")
    printing(f"Write to {output_file}: {cc}")
    # --

# --
# python3 csr/event/io/query2topic.py QDIR OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
