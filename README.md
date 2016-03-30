# golden-horse

The implementation of the paper:

**Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings**  
Nanyun Peng and Mark Dredze  
*Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2015  

If you use the code, please kindly cite the following bibtex:

@inproceedings{peng2015ner,  
title={Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings.},  
author={Peng, Nanyun and Dredze, Mark},  
booktitle={EMNLP},  
pages={548–-554},  
year={2015}  
}

## Dependencies:
This is an theano implementation; it requires installation of python module:  
Theano  
jieba (a Chinese word segmentor)  
Both of them can be simply installed by pip moduleName.

The lstm layer was adapted from http://deeplearning.net/tutorial/lstm.html and the feature extraction part was adapted from crfsuite: http://www.chokkan.org/software/crfsuite/

## A sample command for the training:
python theano_src/crf_ner.py --nepochs 30 --neval_epochs 1 --training_data data/weibo_ner/weiboNER.conll.train --valid_data data/weibo_ner/weiboNER.conll.dev --test_data data/weibo_ner/weiboNER.conll.test --emb_file embeddings/weibo_charpos_vectors --emb_type charpos --save_model_param weibo_best_parameters --eval_test false

## A sample command for running the test:
time python theano_src/crf_ner.py --test_data data/weibo_ner/weiboNER.conll.test --only_test true --output_dir data/ --save_model_param weibo_best-parameters

