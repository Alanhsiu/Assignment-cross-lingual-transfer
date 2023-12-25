from fairseq.models.transformer_lm import TransformerLanguageModel
import numpy as np
from tqdm import tqdm
import random

model_dir = '7.5B'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')
lm = lm.eval()
lm = lm.half()
lm = lm.cuda()
print ("model loaded")

# 3-shot
selected_examples = []
en_examples = []
fr_examples = []
ru_examples = []
zh_examples = []
hi_examples = []
ur_examples = []
bg_examples = []
vi_examples = []

with open('data-bin/flores101_dataset/dev/eng.dev', 'r', encoding='utf-8') as file:
    for line in file:
        en_examples.append(line.strip())
with open('data-bin/flores101_dataset/dev/fra.dev', 'r', encoding='utf-8') as file:
    for line in file:
        fr_examples.append(line.strip())
with open('data-bin/flores101_dataset/dev/zho_trad.dev', 'r', encoding='utf-8') as file:
    for line in file:
        zh_examples.append(line.strip())
with open('data-bin/flores101_dataset/dev/rus.dev', 'r', encoding='utf-8') as file:
    for line in file:
        ru_examples.append(line.strip())
with open('data-bin/flores101_dataset/dev/hin.dev', 'r', encoding='utf-8') as file:
    for line in file:
        hi_examples.append(line.strip())
with open('data-bin/flores101_dataset/dev/urd.dev', 'r', encoding='utf-8') as file:
    for line in file:
        ur_examples.append(line.strip())
with open('data-bin/flores101_dataset/dev/bul.dev', 'r', encoding='utf-8') as file:
    for line in file:
        bg_examples.append(line.strip())
with open('data-bin/flores101_dataset/dev/vie.dev', 'r', encoding='utf-8') as file:
    for line in file:
        vi_examples.append(line.strip())

# load xnli
lang_codes = list()
labels = list()
premises = list()
hypotheses = list()
for i, line in enumerate(open('data-bin/XNLI-1.0/xnli.test.tsv').readlines()):
    if i == 0:
        continue
    line = line.split('\t')
    lang_codes.append(line[0])
    labels.append(line[1])
    premises.append(line[6])
    hypotheses.append(line[7])
print ("data loaded")

lang_codes = np.array(lang_codes)
labels = np.array(labels)
premises = np.array(premises)
hypotheses = np.array(hypotheses)
print ("data converted to numpy")

languages = ['Arabic', 'Bulgarian', 'German', 'Greek', 'English', 'Spanish', 'French', 'Hindi', 'Russian', 'Swahili', 'Thai', 'Turkish', 'Urdu', 'Vietnamese', 'Chinese']
codes = ['en', 'fr', 'ru', 'zh', 'hi', 'ur', 'bg', 'vi']
code_to_lang = dict(zip(np.unique(lang_codes), languages))
print ("languages loaded")

with open('data-bin/XNLI-1.0/others_to_en_3.tsv', 'w') as f:

    for code in tqdm(codes):
        if code == 'en':
            continue

        print ("code: ", code)
        ind = np.where(lang_codes == code)#[0][:100]
        current_premises = premises[ind]
        current_hypotheses = hypotheses[ind]
        current_labels = labels[ind]

        premise_examples = list()
        hypotheses_examples = list()
        min_len_b_premises = list()
        min_len_b_hypotheses = list()

        prompt = 'translate from {} to english: \n'.format(code_to_lang[code])
        print (prompt)

        for premise, hypothesis, label in zip(current_premises, current_hypotheses, current_labels):
            
            selected_index = random.sample(range(len(en_examples)), 3)
            selected_en_examples = [en_examples[i] for i in selected_index]
            selected_target_examples = []
            
            if code == 'fr':
                selected_target_examples = [fr_examples[i] for i in selected_index]
            elif code == 'zh':
                selected_target_examples = [zh_examples[i] for i in selected_index]
            elif code == 'ru':
                selected_target_examples = [ru_examples[i] for i in selected_index]
            elif code == 'hi':
                selected_target_examples = [hi_examples[i] for i in selected_index]
            elif code == 'ur':
                selected_target_examples = [ur_examples[i] for i in selected_index]
            elif code == 'bg':
                selected_target_examples = [bg_examples[i] for i in selected_index]
            elif code == 'vi':
                selected_target_examples = [vi_examples[i] for i in selected_index]
            
            example = prompt + selected_target_examples[0] + ' => ' + selected_en_examples[0] + '\n' + selected_target_examples[1] + ' => ' + selected_en_examples[1] + '\n' + selected_target_examples[2] + ' => ' + selected_en_examples[2] + '\n' + premise + ' => '
            # example = prompt + premise + '=>'
            # print (example)
            
            if code == 'zh':
                max_len_b = len(premise) * 1.2 + 100
            else:
                max_len_b = len(premise.split()) * 1.2 + 100

            premise_examples.append(example)
            min_len_b_premises.append(max_len_b)

            example = prompt + selected_target_examples[0] + ' => ' + selected_en_examples[0] + '\n' + selected_target_examples[1] + ' => ' + selected_en_examples[1] + '\n' + selected_target_examples[2] + ' => ' + selected_en_examples[2] + '\n' + hypothesis + ' => '
            # example = prompt + hypothesis + ' => '
            # print (example)

            if code == 'zh':
                max_len_b = len(hypothesis) * 1.2 + 100
            else:
                max_len_b = len(hypothesis.split()) * 1.2 + 100

            hypotheses_examples.append(example)
            min_len_b_hypotheses.append(max_len_b)
            
        print (len(premise_examples))
        print (len(hypotheses_examples))
        print (len(current_labels))
        print (len(min_len_b_premises))
        print (len(min_len_b_hypotheses))
        print (premise_examples[0])
        print (hypotheses_examples[0])
        print('translating...')
        
        min_len_b = np.max(min_len_b_premises)
        print (min_len_b)
        pred_premises = lm.translate(premise_examples, beam=1, max_len_a=1.0,  max_len_b=max_len_b, replace_newlines_with_eos=True)
        print (pred_premises)
        min_len_b = np.max(min_len_b_hypotheses)
        print (min_len_b)
        pred_hypotheses = lm.translate(hypotheses_examples, beam=1, max_len_a=1.0,  max_len_b=max_len_b, replace_newlines_with_eos=True)
        print (pred_hypotheses)
        
        for pred_premise, pred_hypothesis, label in zip(pred_premises, pred_hypotheses, current_labels):
            pred_premise = pred_premise.split('=>')[-1]
            pred_hypothesis = pred_hypothesis.split('=>')[-1]
            f.write('{}\t{}\t{}\t{}\n'.format(code, label, pred_premise, pred_hypothesis))
