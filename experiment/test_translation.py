from fairseq.models.transformer_lm import TransformerLanguageModel

# image: pytorch-21.06-py3:lates



model_dir = '7.5B'
# model_dir = '564M'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')
lm = lm.eval()
lm = lm.half()
lm = lm.cuda()

def get_logprobs(prompt):
    import re
    prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines, which indicate separate documents
    return lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']
    
# Zero-shot evaluation for the Choice of Plausible Alternatives (COPA) task.
# A return value of 0 indicates that the first alternative is more plausible,
# while 1 indicates that the second alternative is more plausible.
def COPA_eval(prompt, alternative1, alternative2):
    lprob1 = get_logprobs(prompt + "\n" + alternative1).sum()
    lprob2 = get_logprobs(prompt + "\n" + alternative2).sum()
    return 0 if lprob1 > lprob2 else 1
    
    
# example = 'Sportsman Jhonathan Florez jumped from a helicopter above Bogota , the capital of Colombia , on Thursday . = Le sportif Jhonathan Florez a sauté jeudi d\' un hélicoptère au-dessus de Bogota , la capitale colombienne .\n' + \
#         'The American Civil Liberties Union is deeply concerned , too , raising a variety of privacy issues . = L\' American Civil Liberties Union est elle aussi très préoccupée et exprime son inquiétude concernant la protection de la vie privée .\n' + \
#         'They are exploring how , over the next decade , they can move to a system in which drivers pay per mile of road they roll over . = Ils cherchent comment , au cours de la prochaine décennie , ils pourront passer à un système permettant aux conducteurs de payer en fonction du nombre de miles parcourus .\n '

en_lines = [line[:-1] for line in open('../data-bin/test.en').readlines()]
fr_lines = [line[:-1].replace('& apos ;', '\'') for line in open('../data-bin/test.fr').readlines()]
enr_lines = [line[:-1] for line in open('../data-bin/test.en_reordered').readlines()]

# example = 'Translate English to French: \n'
# for i in range(60, 63):
#     example += en_lines[i] + ' = ' + fr_lines[i] + ' \n '

# example += 'Wonks call it a mileage-based user fee . = '

# example = 'Translate English to French: \n'
# example += 'sea otter => loutre de mer \n'
# example += 'peppermint => menthe poivrée \n'
# example += 'plush girafe => menthe peluche \n'
# example += 'cheese => '



# example = 'Reorder: \nI wanted to conserve energy. = I energy conserve to want.\n' + \
#             'I would like a cup of tea. = I a cup of tea like would.\n' + \
#             'I can do it better. = I it better do can.\n' + \
#             'I am a man of charm. = I a man of charm am.\n' + \
#             'I do not like it. = I it like not.\n' + \
#             'I have to fuck you. = '

# example = list()
# tmp = 'Translate from English to French: '
# for i in range(70, 80):
    
#     example.append(tmp + en_lines[i] + ' => ' + fr_lines[i])

# example.append(tmp + ' Wonks call it a mileage-based user fee . => ')


example = 'Reorder: \n'
for i in range(30):
    example += en_lines[i] + ' => ' + enr_lines[i] + ' \n '

example += 'Nevada is among several states now scrambling to find affordable technology that would allow the state to keep track of how many miles a car is being driven , but not exactly where and at what time . => '

# pred = lm.sample('\n'.join(example), replace_newlines_with_eos=True, max_len_a=1.0,  max_len_b=50)
pred = lm.sample(example, replace_newlines_with_eos=True, max_len_a=1.0,  max_len_b=50)
print(pred)



# for lang in ['en', 'zh', 'hi']:
#     for idx, example in enumerate(data_samples[lang]):
#         predict = COPA_eval(example["premise"], example["choice1"], example["choice2"])
#         print(f'{lang}-{idx}', predict, example['label'])