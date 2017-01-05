# Important update of the data

We fixed some inconsistancies in the data, especially the annotations for the nominal mentions.
We thank Hangfeng He for his contribution to the major cleanup and revision of the annotations.

The original and revised annotated data are both made available in the data/ directory, with prefixes weiboNER.conll and weiboNER_2nd_conll, respectively.

We composed updated results of our models on the revised version of the data in the supplementary material: http://www.cs.jhu.edu/~npeng/papers/golden_horse_supplement.pdf. If you want to compare with our models on the revised data, please refer to this supplementary material. Thanks! 

If you use the revised version, please kindly add the citation of the following bibtex to the citation of our original paper:

@article{HeS16,  
author={Hangfeng He and Xu Sun},  
title={F-Score Driven Max Margin Neural Network for Named Entity Recognition in Chinese Social Media.},  
journal={CoRR},  
volume={abs/1611.04234},  
year={2016}  
}

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
python theano_src/crf_ner.py --nepochs 30 --neval_epochs 1 --training_data data/weiboNER.conll.train --valid_data data/weiboNER.conll.dev --test_data data/weiboNER.conll.test --emb_file embeddings/weibo_charpos_vectors --emb_type charpos --save_model_param weibo_best_parameters --eval_test false

## A sample command for running the test:
python theano_src/crf_ner.py --test_data data/weiboNER.conll.test --only_test true --output_dir data/ --save_model_param weibo_best_parameters

In the above example, the output will be written at output_dir/weiboNER.conll.test.prediction. If you also want to see the evaluation (you must have labeled test data), you can add flag --eval_test True.


## Data
We noticed that several factors could affect the replicatability of experiments:  
1. the segmentor for preprocessing: we used jieba 0.37   
2. the random number generator. Alghough we fixed the random seed, we noticed it will render slight different numbers on different machine.  
3. the traditional lexical feature used.  
4. the pre-trained embeddings.
To enhance the replicatability of our experiments, we provide the original data in conll format at data/weiboNER.conll.(train/dev/test). In addition, we also provide files including all the features and the char-positional transformation we used in our experiments in data/crfsuite.weiboNER.charpos.conll.(train/dev/test), as well as the pre-trained char and char-positional embeddings.

Note: the data we provide contains both named and nominal mentions, you can get the dataset with only named entities by simply filtering out the nominal mentions.
