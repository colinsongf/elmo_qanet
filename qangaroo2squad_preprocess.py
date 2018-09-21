'''
Created on 27 Aug, 2018

@author: luotianyi
'''

import codecs, json, sys, nltk, copy, re
from tqdm import tqdm

IDX, QUERY, ANSWER, CAND, DOCS, ANN = ['id', 'query', 'answer', 'candidates', 'supports', 'annotations']

train_in = 'train.json'
train_out ='20180828_true_periond_wikihop.train.qplusa.withcand.squad.json'
dev_in = 'dev.json'
dev_out = '20180828_true_periond_wikihop.dev.cand.withcand.squad.new.json'

def filter_docs(docs, query_ent, candidates, answer_str, is_dev=False):
    
    def check_candidates(candidates, x):
        max_idx = 0
        flag = False
        for c in candidates:
            if c in x:
                flag = True
                cur_idx = x.find(c)
                if cur_idx > max_idx:
                    max_idx = cur_idx
        return flag, max_idx
    
    def find_substr_idx(d, query_ent_sub):
        inverse_idx = lambda x: 1.0/(x+0.5)
        idx = d.find(query_ent_sub)
        if idx > 0: return inverse_idx(idx)
        else:
            meta_chars = ['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '\\', '|', '(', ')']
            
            bef = query_ent_sub
            for c in meta_chars: query_ent_sub = query_ent_sub.replace(c, '')
            query_ent_tokens = query_ent_sub.strip().split()
            if len(query_ent_tokens) == 0:
                return inverse_idx(-1)
            if len(query_ent_tokens) == 1:
                return inverse_idx(d.find(query_ent_sub))
            else:
                try:
                    query_ent_re = re.compile('%s (.*?)?%s'%(query_ent_tokens[0], query_ent_tokens[-1]))
                    match_str = re.search(query_ent_re, d)
                    if match_str: return inverse_idx(match_str.start())
                    else: return inverse_idx(-1)
                except:
                    print(bef)
                    print(query_ent_sub)
                    print(d)
                    sys.exit(0)   
    
    #filter the documents and keep the sentences containing the candidates
    #print(type(docs))
    temp_docs = []
    for temp_doc in docs:
        temp_str_period_level = ""
        list_temp_doc_split_by_period = temp_doc.split(".")
        #len_list_split_by_period = len(list_temp_doc_split_by_period)
        len_has_candidate_sentence_period = 0
        temp_len_split_by_period = 0
        for temp_doc_split_by_period in list_temp_doc_split_by_period:
            if len(temp_doc_split_by_period) == 0:
                if len(temp_str_period_level) != 0:
                    temp_str_period_level += "."
                    continue
            temp_len_split_by_period += 1
            list_temp_doc_split_by_comma = temp_doc_split_by_period.split(",")
            temp_str_comma_level = ""
            #len_list_split_by_comma = len(list_temp_doc_split_by_comma)
            temp_len_split_by_comma = 0
            len_has_candidate_sentence = 0
            #is_or_not_candidate_in_sentence_period = False
            for temp_doc_split_by_comma in list_temp_doc_split_by_comma:
                if len(temp_doc_split_by_comma) == 0:
                    if len(temp_str_comma_level) != 0:
                        temp_str_comma_level += ","
                        continue
                temp_len_split_by_comma += 1
                is_or_not_candidate_in_sentence = False
                for temp_candidate in candidates:
                    #if is_or_not_candidate_in_sentence:
                    #    break
                    if temp_candidate.lower() in temp_doc_split_by_comma.lower():
                        is_or_not_candidate_in_sentence = True
                        len_has_candidate_sentence += 1
                        break
                if is_or_not_candidate_in_sentence:
                    if len_has_candidate_sentence == 1:
                        temp_str_comma_level += temp_doc_split_by_comma
                    else:
                        temp_str_comma_level += "," + temp_doc_split_by_comma
            if len(temp_str_comma_level) != 0:
                #is_or_not_candidate_in_sentence_period = True
                len_has_candidate_sentence_period += 1
                if len_has_candidate_sentence_period == 1:
                    temp_str_period_level += temp_doc_split_by_period
                else:
                    temp_str_period_level += "." + temp_doc_split_by_period
        if len(temp_str_period_level) > 0:
            temp_docs.append(temp_str_period_level)
        #else:
        #    print("None!!!")
    docs = temp_docs
    # ----- start from here ---- #
    docs_tk = [' '.join(nltk.word_tokenize(d.lower())) for d in docs]
    query_ent = ' '.join(nltk.word_tokenize(query_ent.lower()))
    # fix tokenization error
    if '-' in query_ent: query_ent = ' - '.join([x.strip() for x in query_ent.split('-')])

    query_ent_sub = copy.deepcopy(query_ent)
    if query_ent.startswith('list of'): 
        query_ent_sub = query_ent.split('list of')[1].strip()
    
    doc_with_labels = sorted([(d, dtk, find_substr_idx(dtk, query_ent_sub)) 
                              for d, dtk in zip(docs, docs_tk)],
                             key=lambda x:x[2], reverse=True)
    
    #doc_with_answers = []
    if doc_with_labels[0][2] > 0:
        samples = [doc_with_labels[0][0].strip()]
    else:
        if is_dev:
            samples = []
        else:
            return docs

    for d, dtk, l in doc_with_labels[len(samples):]:
        dtk = dtk.strip()
        if is_dev:
            has_can, max_idx = check_candidates(candidates, dtk)
            if has_can:
                samples.append(d)
        else:
            if answer_str in dtk:
                samples.append(d)

    #attach docs with candidates after the one with the answer for training set
    if not is_dev:
        for d, dtk, l in doc_with_labels[len(samples):]:
            has_can, max_idx = check_candidates(candidates, dtk)
            if has_can:
                samples.append(d)

    return samples

def main(in_file, out_file, is_dev=False):
    count = 0
    total_docs = 0
    filtered_docs = 0
    # cand_len = []

    squad_samples = {'version':'qangaroo_wikihop_v1.1', 'data':[]}

    with codecs.open(in_file,'r',encoding="utf-8") as fin, codecs.open(out_file,'w+',encoding="utf-8") as fout:

        all_items_raw = json.load(fin)
        for item in tqdm(all_items_raw):
            query_tuple = item[QUERY].strip().split()
            query_rel = ' '.join(query_tuple[0].split('_'))  # string with '_' between words
            query_ent = ' '.join(query_tuple[1:])  # multi-word string

            query_str = ' '.join([query_rel, query_ent]).lower()
            
            answer_str = item[ANSWER].strip().lower()  # string
            
            answer_str_tk = ' '.join(nltk.word_tokenize(answer_str)) # tokenize answer string

            candidates = [x.strip().lower() for x in item[CAND]] # string
            # candidates_tk = [' '.join(nltk.word_tokenize(x)) for x in candidates] # tokenize candidates
            # assert answer_str_tk in candidates_tk
            
            answer_position = candidates.index(answer_str)
            # cand_len.append((len(candidates), answer_position))

            total_docs += len(item[DOCS])
            docs = filter_docs(item[DOCS], query_ent, candidates, answer_str_tk, is_dev)
            filtered_docs += len(docs)
            
            docs_str = ' eos '.join([x.lower() for x in docs])
            
            tokens = [x for x in docs_str.split() if len(x.strip()) > 0]

            docs_str = ' '.join(tokens)

            id_str = item[IDX]
            
            answer_start = docs_str.find(answer_str)
            # answer_start = []
            # current = docs_str.find(answer_str)
            # while current > -1:
            #     answer_start.append(current)
            #     current += len(answer_str)
            #     current = docs_str.find(answer_str, current)
            
            # if len(answer_start) > 0:
            if is_dev:
                count +=1
                # answers = [{'text':answer_str, 'answer_start':int(x)} for x in answer_start]
                answers = [{'text':answer_str, 'answer_start':int(answer_start), 
                            'candidates': candidates, 'answer_position': int(answer_position)}]
                qas = [{'id':id_str, 'question':query_str, 'answers':answers}]
                paras = [{'context': docs_str, 'qas':qas}]
                title = ' '.join([id_str, query_str])
                squad_samples['data'].append({'title': title, 'paragraphs':paras})

            else:
                if answer_start > -1:
                    count +=1
                    # answers = [{'text':answer_str, 'answer_start':int(x)} for x in answer_start]
                    answers = [{'text':answer_str, 'answer_start':int(answer_start), 
                                'candidates': candidates, 'answer_position': int(answer_position)}]
                    qas = [{'id':id_str, 'question':query_str, 'answers':answers}]
                    paras = [{'context': docs_str, 'qas':qas}]
                    title = ' '.join([id_str, query_str])
                    squad_samples['data'].append({'title': title, 'paragraphs':paras})
        json.dump(squad_samples, fout)
        print(count)

    print(total_docs)
    print(filtered_docs)

if __name__ == "__main__":

    main(dev_in, dev_out, is_dev=True)

    main(train_in, train_out)
