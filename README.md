# Assignment: Cross-lingual Transfer

## Prerequisites
- Navigate to the file `fairseq/fairseq/modules/layer_norm.py` and comment out lines 31 and 32.
- Run the script `experiment/req.sh`.

## Script Descriptions
- `experiment/test_translate_xnli.py`: Zero-shot translation from other languages to English.
  - TODO: Modify the script to perform k-shot translation with k > 0 to improve the translation quality. Examples for k-shot translation can be drawn from the dev set of XNLI since the other languages’ sentences are translations of the English ones. One can also use the dev set of the Flores-101 benchmark by MetaAI (https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz).
- `experiment/test_xnli.py`: Zero-shot in-context learning on the English testing set of XNLI.
  - TODO: Modify the script to implement {0, 12}-shot in-context learning for part 1 and part 2.

