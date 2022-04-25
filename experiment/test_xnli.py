from fairseq.models.transformer_lm import TransformerLanguageModel
import numpy as np
from tqdm import tqdm


model_dir = '7.5B'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')
lm = lm.eval()
lm = lm.half()
lm = lm.cuda()

def get_logprobs(prompt):
    import re
    prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines, which indicate separate documents
    return lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']
    

# to 
pred_label = np.array(['entailment', 'contradiction', 'neutral'])
def XNLI_eval(premise, hypothesis):
    lprob1 = get_logprobs(premise + " , right? Yes, " + hypothesis).sum().cpu()
    lprob2 = get_logprobs(premise + " , right? No, " + hypothesis).sum().cpu()
    lprob3 = get_logprobs(premise + " , right? Also, " + hypothesis).sum().cpu()
    return pred_label[np.argmax([lprob1, lprob2, lprob3])]
    
def val_all(premise, hypothesis):
    lprob1 = get_logprobs(premise + " , right? Yes, " + hypothesis).sum().cpu()
    lprob2 = get_logprobs(premise + " , right? No, " + hypothesis).sum().cpu()
    lprob3 = get_logprobs(premise + " , right? Also, " + hypothesis).sum().cpu()
    return [lprob1, lprob2, lprob3]

# load xnli
lang = list()
label = list()
premise = list()
hypothesis = list()
for i, line in enumerate(open('data-bin/XNLI-1.0/xnli.test.tsv').readlines()):
    if i == 0:
        continue
    line = line.split('\t')
    lang.append(line[0])
    label.append(line[1])
    premise.append(line[6])
    hypothesis.append(line[7])


lang = np.array(lang)
label = np.array(label)
premise = np.array(premise)
hypothesis = np.array(hypothesis)

en_ind = np.where(lang == 'en')
en_label = label[en_ind]
en_premise = premise[en_ind]
en_hypothesis = hypothesis[en_ind]

acc = 0.0
for i in tqdm(range(len(en_label))):
    predict = XNLI_eval(en_premise[i], en_hypothesis[i])
    if predict == en_label[i]:
        acc += 1.0

print('accuracy of zero-shot on en: ', acc/float(len(en_label)))

