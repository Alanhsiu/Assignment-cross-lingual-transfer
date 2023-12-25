#!/bin/bash

languages=("en" "fr" "ru" "zh" "hi" "ur" "bg" "vi")
translated_languages=("fr" "ru" "zh" "hi" "ur" "bg" "vi")

python ./experiment/test_xnli.py en
for lang in "${languages[@]}"
do
    echo "Running for language: $lang"
    python ./experiment/test_xnli_zh_prompt.py $lang
    python ./experiment/test_xnli_12_zh_prompt.py $lang
done

# for lang in "${translated_languages[@]}"
# do
#     echo "Running for translated language: $lang"
#     # python ./experiment/test_xnli_12_others_to_en.py $lang 
#     python ./experiment/test_xnli_12_others_to_en_3.py $lang 
# done

# inference in english
for lang in "${translated_languages[@]}"
do
    echo "Running for language: $lang"
    python ./experiment/test_xnli_12_inference_in_en.py $lang
done

python ./experiment/test_xnli_12_bg_to_ru.py
python ./experiment/test_xnli_12_ru_to_bg.py
