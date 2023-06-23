import json
import random
from tqdm import tqdm


def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data


def save_file(data, path):
    with open(path, 'w', encoding='utf-8') as w:
        for unit in data:
            output = json.dumps(unit)
            w.write(output + "\n")
        w.close()


def build_claim_popQA(relation, subj, obj):
    if relation == "occupation":
        return subj + "'s occupation is " + obj + '.'
    elif relation == "place of birth":
        return subj + " was born in " + obj + '.'
    elif relation == "genre":
        return "The genre of " + subj + " is " + obj + '.'
    elif relation == "father":
        return obj + " is the father of " + subj + '.'
    elif relation == "country":
        return subj + " is in " + obj + '.'
    elif relation == "producer":
        return obj + ' is the producer of ' + subj + '.'
    elif relation == "director":
        return obj + ' is the director of ' + subj + '.'
    elif relation == "capital of":
        return subj + ' is the capital of ' + obj + '.'
    elif relation == "screenwriter":
        return obj + ' is the screenwriter for ' + subj + '.'
    elif relation == "composer":
        return obj + ' was the composer of ' + subj + '.'
    elif relation == "color":
        return "The color of " + subj + " is " + obj + '.'
    elif relation == "religion":
        return obj + " is the religion of " + subj + '.'
    elif relation == "sport":
        return subj + " plays " + obj + '.'
    elif relation == "author":
        return obj + " is the author of " + subj + '.'
    elif relation == "mother":
        return obj + " is the mother of " + subj + '.'
    elif relation == "capital":
        return obj + " is the capital of " + subj + '.'
    else:
        raise ValueError("Wrong Relation " + relation)


def build_zeroshot_prompt_popQA(filename, model_name='gpt-4'):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            question = unit["question"]
            if 'gpt-4' in model_name:
                prompt_text = "In the first paragraph, you are expected to answer the question. And in the second paragraph, you should give the evidence.\n\nQ:" + question + "\n" + "A:"
            else:
                prompt_text = "The first paragraph answers the question and the second paragraph gives the reason.\n\nQ:" + question + "\n" + "A:"
            data.append(prompt_text)
    return data


def build_doubleCheck_prompt_popQA(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            question = unit["question"]
            evidence = unit["parametric_memory"]
            # We only test the examples where both answers are supported for subsequent experiments.
            if 'parametric_entailment' in unit and unit['parametric_entailment'] and 'conflict_entailment' in unit and unit['conflict_entailment']:
                prompt_text = "According to the given information and your knowledge, answer the question.\n\nInformation:" + evidence + '\nQ:' + question + "\n" + "A:"
            else:
                prompt_text = ""
            data.append(prompt_text)
    return data


def build_zeroshot_prompt_strategyQA(filename, model_name='gpt-4'):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        file = json.load(f)['strategy_qa']['train']
    for unit in file:
        question = unit["question"]
        if 'gpt-4' in model_name:
            prompt_text = """In the first paragraph, you are expected to answer the question "True" or "False". And in the second paragraph, you should give the evidence.\n\nQ:""" + question + "\n" + "A:"
        else:
            prompt_text = """The first paragaph answers the question "True" or "False" and the second paragraph gives the reason.\n\nQ:""" + question + "\n" + "A:"
        data.append(prompt_text)
    return data


def build_doubleCheck_prompt_strategyQA(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            question = unit["question"]
            evidence = unit["parametric_memory"]
            # We only test the examples where both answers are supported for subsequent experiments.
            if 'parametric_entailment' in unit and unit['parametric_entailment'] and 'conflict_entailment' in unit and unit['conflict_entailment']:
                prompt_text = """According to the given information and your knowledge, answer the question "True" or "False".\n\nInformation: """ + evidence + '\nQ:' + question + "\n" + "A:"
            else:
                prompt_text = ""
            data.append(prompt_text)
    return data


def build_cliam_prompt_strategyQA(filename, contrary=False):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            question = unit["question"]
            ans = "True"
            if contrary:
                ans = "False"
            prompt_text = "Following the examples, please make a claim based on the given question and answer.\nQ: Is it more risky to use Tesla's autopilot than to drive drunk?\nA: False.\nClaim: It is not more risky to use Tesla's autopilot than to drive drunk.\nQ: Would it be harder for me to find a job in the USA in 2009 vs. 1932?\nA: True\nClaim: It would be harder for me to find a job in the USA in 2009 vs. 1932\n" \
                          + "Q: " + question + '\n' + "A: " + ans + '\n' + "Claim: "
            data.append(prompt_text)
    return data


def build_conflict_evidence_prompt_strategyQA(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            claim = unit["contrary_claim"]
            if claim is None:
                prompt_text = ""
            else:
                if unit['self-consistency'] == 'True2True':
                    claim = unit['contrary_claim']
                elif unit['self-consistency'] == 'False2False':
                    claim = unit['claim']
                else:
                    raise ValueError("Wrong self-consistency type")
                prompt_text = "Given a claim, please write a short piece of evidence to support it. You can make up fake content and supporting evidence but it should be as realistic as possible.\n\nClaim: " + claim + "\n\nPassage: "
            data.append(prompt_text)
    return data


def build_contrary_cliam_prompt_popQA(filename):
    popQA = []
    relation_set = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            relation = unit['prop']
            possible_ans = unit['possible_answers']
            if relation not in relation_set:
                relation_set[relation] = set()
                relation_set[relation].add(";".join(x for x in possible_ans))
            else:
                relation_set[relation].add(";".join(x for x in possible_ans))
            popQA.append(unit)
        for key in relation_set:
            relation_set[key] = list(relation_set[key])
            for idx, unit in enumerate(relation_set[key]):
                relation_set[key][idx] = unit.split(";")
        for unit in tqdm(popQA):
            double_correct = unit["double_correct"]
            relation = unit['prop']
            subj = unit['subj']
            parametric_memory = unit['parametric_memory']
            if unit["is-consistency"] != True:
                unit['contrary_ans'] = None
                prompt_text = ""
            else:
                if double_correct == True:
                    cont_ans = ""
                    while cont_ans == "":
                        flag = False
                        random_possible_ans = random.choice(relation_set[relation])
                        for ans in random_possible_ans:
                            if ans in parametric_memory:
                                flag = True
                                break
                        if not flag:
                            cont_ans = random.choice(random_possible_ans)
                            break

                elif double_correct == False:
                    cont_ans = random.choice(unit['possible_answers'])
                else:
                    print(unit)
                    raise ValueError("The answer must in {'True','False'}")
                claim = build_claim_popQA(relation, subj, cont_ans)
                unit['contrary_ans'] = cont_ans
    return popQA


def build_conflict_evidence_prompt_popQA(filename, reGen=False):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            relation = unit['prop']
            subj = unit['subj']
            cont_ans = unit["contrary_ans"]
            if unit["is-consistency"] != True:
                prompt_text = ""
            else:
                if not reGen:
                    try:
                        claim = build_claim_popQA(relation, subj, cont_ans)
                    except:
                        print(relation, subj, cont_ans)
                        assert False
                else:
                    if unit['reGen_plus']:
                        if unit['self-consistency'] == 'True2True':
                            claim = unit['contrary_claim']
                        elif unit['self-consistency'] == 'False2False':
                            claim = unit['first_pred_ans']
                        else:
                            raise ValueError("Wrong self-consistency type")
                prompt_text = "Given a claim, please write a short piece of evidence to support it. You can make up fake content and supporting evidence but it should be as realistic as possible.\n\nClaim: " \
                              + claim + "\n\nPassage: "
            data.append(prompt_text)
    return data


def build_zeroshot_prompt_strategyQA_with_singleSource_evidence(filename, mode='implicit', evidence_order="para_conf",
                                                                regenerate=False):
    # v1 A: True.\nB: False.\nC: Uncertain.
    # v2 A: Wrong.\nB: Unknown\nC: Right.
    # v3 A: Unanswerable.\nB: Yes.\nC: No.
    file = load_line_json_data(filename)
    data = []
    for unit in file:
        if unit['conflict_entailment'] != True:
            data.append("")
            continue
        if regenerate:
            if unit['reGen_flag'] != True:
                data.append("")
                continue
        question = unit["question"]
        conflict_evidence = unit["conflict_evidence"]
        parametric_evidence = unit['parametric_memory']
        assert conflict_evidence is not None and parametric_evidence is not None
        if mode == 'implicit':
            evidence = conflict_evidence
            dynamic_prompt = " and your knowledge"
        elif mode == 'explicit':
            if evidence_order == "conf_para":
                evidence = '\n1. ' + conflict_evidence + '\n2. ' + parametric_evidence
            elif evidence_order == "para_conf":
                evidence = '\n1. ' + parametric_evidence + '\n2. ' + conflict_evidence
            elif evidence_order == "random":
                tmp_evidence = [parametric_evidence] + [conflict_evidence]
                random.shuffle(tmp_evidence)
                evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1]
            dynamic_prompt = ""
        prompt_text = """According to the given information""" + dynamic_prompt + """, choose the best choice from the following options.\n\nInformation: """ + evidence + "\n\nQuestion: " + question + "\n\nOptions:\nA: True.\nB: False.\nC: Uncertain.\n\nAnswer: "
        data.append(prompt_text)
    return data


def build_zeroshot_prompt_popQA_with_singleSource_evidence(filename, mode='explicit', evidence_order="random",
                                                           option_order="conf_para", parametric_filter=True,
                                                           conflict_type="all", regenerate=False):
    file = load_line_json_data(filename)
    data = []
    for unit in file:
        if not parametric_filter:
            if unit['conflict_entailment'] != True:
                data.append("")
                continue
        else:
            if unit['conflict_entailment'] != True or unit['parametric_entailment'] != True or unit['reGen_plus']:
                data.append("")
                continue

        if regenerate:
            if unit['reGen_flag'] != True:
                data.append("")
                continue
        question = unit["question"]
        if conflict_type == "all":
            conflict_evidence = unit["conflict_evidence"]
        elif conflict_type == "sum":
            conflict_evidence = unit["conflict_evidence_summarization"]
        parametric_evidence = unit['parametric_memory']
        first_pred = unit["first_pred_ans"]
        contrary_claim = unit["contrary_claim"]
        assert conflict_evidence is not None and parametric_evidence is not None
        if mode == 'implicit':
            evidence = conflict_evidence
            dynamic_prompt = " and your knowledge"
            option = "Options:\nA: " + first_pred + "\nB: " + contrary_claim + "\nC: Uncertain."
        elif mode == 'explicit':
            if evidence_order == "conf_para":
                evidence = '\n1. ' + conflict_evidence + '\n2. ' + parametric_evidence
            elif evidence_order == "para_conf":
                evidence = '\n1. ' + parametric_evidence + '\n2. ' + conflict_evidence
            elif evidence_order == "random":
                tmp_evidence = [parametric_evidence] + [conflict_evidence]
                random.shuffle(tmp_evidence)
                evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1]
            dynamic_prompt = ""
            if option_order == "para_conf":
                option = "Options:\nA: " + first_pred + "\nB: " + contrary_claim + "\nC: Uncertain."
            elif option_order == "conf_para":
                option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: Uncertain."
        elif mode == "irrelevant":
            irrelevant_evidence = unit["irrelevant_evidence_sentBERT"]["text"]
            irrelevant_id = unit["irrelevant_evidence_sentBERT"]["idx"]
            contrary_claim = build_claim_popQA(unit['prop'], unit['subj'],
                                               file[irrelevant_id]['obj'])  # TODO no cover r
            evidence = irrelevant_evidence
            option = "Options:\nA: " + first_pred + "\nB: " + contrary_claim + "\nC: Uncertain."
            dynamic_prompt = " and your knowledge"
        elif mode == "2irr":
            irrelevant_evidence = []
            irrelevant_claim = []
            for i in range(2):
                irrelevant_evidence.append(unit["irrelevant_evidence_sentBERT"][i]["text"])
                irrelevant_id = unit["irrelevant_evidence_sentBERT"][i]["idx"]
                irrelevant_claim.append(build_claim_popQA(unit['prop'], unit['subj'], file[irrelevant_id]['obj']))
            tmp_evidence = irrelevant_evidence
            random.shuffle(irrelevant_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1]
            dynamic_prompt = " and your knowledge"
            option = "Options:\nA: " + first_pred + "\nB: " + irrelevant_claim[0] + "\nC: " + irrelevant_claim[
                1] + "\nD: Uncertain."
        elif mode == "3irr":
            irrelevant_evidence = []
            irrelevant_claim = []
            for i in range(3):
                irrelevant_evidence.append(unit["irrelevant_evidence_sentBERT"][i]["text"])
                irrelevant_id = unit["irrelevant_evidence_sentBERT"][i]["idx"]
                irrelevant_claim.append(build_claim_popQA(unit['prop'], unit['subj'], file[irrelevant_id]['obj']))
            tmp_evidence = irrelevant_evidence
            random.shuffle(irrelevant_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2]
            dynamic_prompt = " and your knowledge"
            option = "Options:\nA: " + first_pred + "\nB: " + irrelevant_claim[0] + "\nC: " + irrelevant_claim[
                1] + "\nD: " + irrelevant_claim[2] + "\nE: Uncertain."
        else:
            raise ValueError("Unknown mode.")
        prompt_text = """According to the given information""" + dynamic_prompt + """, choose the best choice from the following options.\n\nInformation: """ + evidence + "\n\nQuestion: " + question + "\n\n" + option + "\n\nAnswer: "
        data.append(prompt_text)
    return data


def build_zeroshot_prompt_popQA_with_singleSource_evidence_entSub(filename):
    # v1 A: First_pred.\nB: Contrary_claim.\nC: Uncertain.
    file = load_line_json_data(filename)
    data = []
    for unit in file:
        if unit['entity_substitution_parametric_memory'] is None:
            data.append("")
            continue
        question = unit["question"]
        evidence = unit["entity_substitution_parametric_memory"]
        parametric_evidence = unit['parametric_memory']
        first_pred = unit["first_pred_ans"]
        contrary_claim = unit["contrary_claim"]
        option = "Options:\nA: " + first_pred + "\nB: " + contrary_claim + "\nC: Uncertain."
        assert evidence is not None and parametric_evidence is not None
        prompt_text = """According to the given information and your knowledge, choose the best choice from the following options.\n\nInformation: """ + evidence + "\n\nQuestion: " + question + "\n\n" + option + "\n\nAnswer: "
        data.append(prompt_text)
    return data


def build_zeroshot_prompt_popQA_summarization(filename):
    # v1 A: First_pred.\nB: Contrary_claim.\nC: Uncertain.
    file = load_line_json_data(filename)
    data = []
    for unit in file:
        if unit['conflict_entailment'] != True:
            data.append("")
            continue
        conflict_evidence = unit["conflict_evidence"]
        contrary_claim = unit["contrary_claim"]
        assert conflict_evidence is not None and contrary_claim is not None
        prompt_text = """Summarize the given text. Note that the summarization should be as short as possible, and the given claim should be mentioned.\n\nText:\n""" + conflict_evidence + "\n\nClaim: " + contrary_claim + "\n\nSummarization:\n"
        data.append(prompt_text)
    return data


def build_zeroshot_prompt_popQA_triplets2nl(filename):
    # v1 A: First_pred.\nB: Contrary_claim.\nC: Uncertain.
    file = load_line_json_data(filename)
    data = []
    for unit in file:
        if 'ground_truth' in unit and type(unit['ground_truth']) == list:
            triplets = str(unit['ground_truth'])
            prompt_text = """Given a list of triplets (the schema is [subjective,relation,objective]), please help me to rewrite it in natural language form. All the objective should be included in the passage as its given value. Note that """ + \
                          unit['obj'] + " have to be mentioned.\n\nTriplets:" + triplets + "\n\nPassage: "
            data.append(prompt_text)
        else:
            data.append("")
            continue
    return data


def build_zeroshot_prompt_popQA_with_multiSource_evidence(filename, mode='all_true', parametric_filter=True,
                                                          regenerate=False):
    # v1 A: First_pred.\nB: Contrary_claim.\nC: Uncertain.
    file = load_line_json_data(filename)
    data = []
    for unit in file:
        if not parametric_filter:
            if unit['conflict_entailment'] != True:
                data.append("")
                continue
        else:
            if 'ground_truth' not in unit or unit['ground_truth'] == None:
                data.append("")
                continue
        if regenerate:
            if unit['reGen_flag'] != True:
                data.append("")
                continue
        if unit['self-consistency'] == 'True2True':
            true_evidence = [unit['parametric_memory'], unit['ground_truth']]
            false_evidence = [unit['conflict_evidence'], unit['evidence_plus']]
        elif unit['self-consistency'] == 'False2False':
            true_evidence = [unit['conflict_evidence'], unit['ground_truth']]
            false_evidence = [unit['parametric_memory'], unit['evidence_plus']]
        else:
            raise ValueError(unit['self-consistency'] + " is not supported.")

        question = unit["question"]
        first_pred = unit["first_pred_ans"]
        contrary_claim = unit["contrary_claim"]
        option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: Uncertain."
        conflict_evidence = unit["conflict_evidence"]
        parametric_evidence = unit['parametric_memory']
        if mode == 'all_true':
            evidence = '\n1. ' + true_evidence[0] + '\n2. ' + true_evidence[1]
            dynamic_prompt = ""
        elif mode == 'all_false':
            evidence = '\n1. ' + false_evidence[0] + '\n2. ' + false_evidence[1]
            dynamic_prompt = ""
        elif mode == '2t1f':
            tmp_evidence = true_evidence + random.sample(false_evidence, 1)
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2]
            dynamic_prompt = ""
        elif mode == '2f1t':
            tmp_evidence = false_evidence + random.sample(true_evidence, 1)
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2]
            dynamic_prompt = ""
        elif mode == '2t2f':
            tmp_evidence = false_evidence + true_evidence
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2] + '\n4. ' + \
                       tmp_evidence[3]
            dynamic_prompt = ""
        elif mode == "1t1f_gt":
            if unit['self-consistency'] == 'False2False':
                data.append("")
                continue
            tmp_evidence = [conflict_evidence] + [unit['ground_truth']]
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1]
            dynamic_prompt = ""
            option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: Uncertain."
        elif mode == "1t1f_gt_f2f":
            if unit['self-consistency'] == 'True2True':
                data.append("")
                continue
            tmp_evidence = [parametric_evidence] + [unit['ground_truth']]
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1]
            dynamic_prompt = ""
            option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: Uncertain."
        elif mode == "1t1f1i":
            irrelevant_evidence = unit["irrelevant_evidence_sentBERT"][0]["text"]
            irrelevant_id = unit["irrelevant_evidence_sentBERT"][0]["idx"]
            irrelevant_claim = build_claim_popQA(unit['prop'], unit['subj'], file[irrelevant_id]['obj'])
            tmp_evidence = [conflict_evidence] + [parametric_evidence] + [irrelevant_evidence]
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2]
            dynamic_prompt = ""
            option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: " + irrelevant_claim + "\nD: Uncertain."
        elif mode == "1t1f_conClaim":
            tmp_evidence = [contrary_claim] + [parametric_evidence]
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1]
            dynamic_prompt = ""
            option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: Uncertain."
        elif mode == "1t1f_memoryClaim":
            tmp_evidence = [conflict_evidence] + [first_pred]
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1]
            dynamic_prompt = ""
            option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: Uncertain."
        elif mode == "1t1f2i":
            irrelevant_evidence = []
            irrelevant_claim = []
            for i in range(2):
                irrelevant_evidence.append(unit["irrelevant_evidence_sentBERT"][i]["text"])
                irrelevant_id = unit["irrelevant_evidence_sentBERT"][i]["idx"]
                irrelevant_claim.append(build_claim_popQA(unit['prop'], unit['subj'], file[irrelevant_id]['obj']))
            tmp_evidence = [conflict_evidence] + [parametric_evidence] + irrelevant_evidence
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2] + '\n4. ' + \
                       tmp_evidence[3]
            dynamic_prompt = ""
            option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: " + irrelevant_claim[0] + "\nD: " + \
                     irrelevant_claim[1] + "\nE: Uncertain."
        elif mode == "1t1f3i":
            irrelevant_evidence = []
            irrelevant_claim = []
            for i in range(3):
                irrelevant_evidence.append(unit["irrelevant_evidence_sentBERT"][i]["text"])
                irrelevant_id = unit["irrelevant_evidence_sentBERT"][i]["idx"]
                irrelevant_claim.append(build_claim_popQA(unit['prop'], unit['subj'], file[irrelevant_id]['obj']))
            tmp_evidence = [conflict_evidence] + [parametric_evidence] + irrelevant_evidence
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2] + '\n4. ' + \
                       tmp_evidence[3] + '\n5. ' + tmp_evidence[4]
            dynamic_prompt = ""
            option = "Options:\nA: " + contrary_claim + "\nB: " + first_pred + "\nC: " + irrelevant_claim[0] + "\nD: " + \
                     irrelevant_claim[1] + "\nE: " + irrelevant_claim[2] + "\nF: Uncertain."
        else:
            raise ValueError("Unknown mode.")
        prompt_text = """According to the given information""" + dynamic_prompt + """, choose the best choice from the following options.\n\nInformation: """ + evidence + "\n\nQuestion: " + question + "\n\n" + option + "\n\nAnswer: "
        data.append(prompt_text)
    return data


def build_zeroshot_prompt_strategyQA_with_multiSource_evidence(filename, mode='all_true', parametric_filter=True,
                                                               regenerate=False):
    file = load_line_json_data(filename)
    data = []
    for unit in file:
        if not parametric_filter:
            if unit['conflict_entailment'] != True:
                data.append("")
                continue
        else:
            if unit['conflict_entailment'] != True or unit['parametric_entailment'] != True or unit['reGen_plus']:
                data.append("")
                continue

        if regenerate:
            if unit['reGen_flag'] != True:
                data.append("")
                continue

        if unit['self-consistency'] == 'True2True':
            true_evidence = [unit['parametric_memory'], unit['ground_truth']]
            false_evidence = [unit['conflict_evidence'], unit['evidence_plus']]
        elif unit['self-consistency'] == 'False2False':
            true_evidence = [unit['conflict_evidence'], unit['ground_truth']]
            false_evidence = [unit['parametric_memory'], unit['evidence_plus']]
        else:
            raise ValueError(unit['self-consistency'] + " is not supported.")

        question = unit["question"]
        option = "Options:\nA: True" + "\nB: False" + "\nC: Uncertain."

        if mode == 'all_true':
            evidence = '\n1. ' + true_evidence[0] + '\n2. ' + true_evidence[1]
            dynamic_prompt = ""
        elif mode == 'all_false':
            evidence = '\n1. ' + false_evidence[0] + '\n2. ' + false_evidence[1]
            dynamic_prompt = ""
        elif mode == '2t1f':
            tmp_evidence = true_evidence + random.sample(false_evidence, 1)
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2]
            dynamic_prompt = ""
        elif mode == '2f1t':
            tmp_evidence = false_evidence + random.sample(true_evidence, 1)
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2]
            dynamic_prompt = ""
        elif mode == '2t2f':
            tmp_evidence = false_evidence + true_evidence
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2] + '\n4. ' + \
                       tmp_evidence[3]
            dynamic_prompt = ""
        elif mode == '2t2f_gt_scatter':
            true_evidence = [unit['conflict_evidence']] + unit['cot']
            tmp_evidence = false_evidence + true_evidence
            random.shuffle(tmp_evidence)
            evidence = ""
            for idx in range(len(tmp_evidence)):
                evidence += '\n' + str(idx + 1) + '. ' + tmp_evidence[idx]
        elif mode == "1t1f1i":
            irrelevant_evidence = unit["irrelevant_evidence_sentBERT"]
            tmp_evidence = random.sample(false_evidence, 1) + random.sample(true_evidence, 1) + [irrelevant_evidence]
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1] + '\n3. ' + tmp_evidence[2]
            dynamic_prompt = ""
        elif mode == "1t1f_gt":
            tmp_evidence = [true_evidence[1]] + [false_evidence[0]]
            random.shuffle(tmp_evidence)
            evidence = '\n1. ' + tmp_evidence[0] + '\n2. ' + tmp_evidence[1]
            dynamic_prompt = ""
        else:
            raise ValueError("Unknown mode.")
        prompt_text = """According to the given information""" + dynamic_prompt + """, choose the best choice from the following options.\n\nInformation: """ + evidence + "\n\nQuestion: " + question + "\n\n" + option + "\n\nAnswer: "
        data.append(prompt_text)
    return data