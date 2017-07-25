# Important update of the data

We fixed some inconsistancies in the data, especially the annotations for the nominal mentions.
We thank Hangfeng He for his contribution to the major cleanup and revision of the annotations.

The original and revised annotated data are both made available in the data/ directory, with prefixes weiboNER.conll and weiboNER_2nd_conll, respectively.

We composed updated results of our models on the revised version of the data in the supplementary material: http://www.cs.jhu.edu/~npeng/papers/golden_horse_supplement.pdf. If you want to compare with our models on the revised data, please refer to this supplementary material. Thanks! 

Please note that the updated version provided

If you use the revised dataset, please kindly cite the following bibtex in addition to the citation of our papers:

@article{HeS16,  
author={Hangfeng He and Xu Sun},  
title={F-Score Driven Max Margin Neural Network for Named Entity Recognition in Chinese Social Media.},  
journal={CoRR},  
volume={abs/1611.04234},  
year={2016}  
}

# golden-horse

The implementation of the papers:

**Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings**  
Nanyun Peng and Mark Dredze  
*Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2015  

and  

**Improving Named Entity Recognition for Chinese Social Media  
with Word Segmentation Representation Learning**  
Nanyun Peng and Mark Dredze  
*Annual Meeting of the Association for Computational Linguistics (ACL)*, 2016  

If you use the code, please kindly cite the following bibtex:

@inproceedings{peng2015ner,  
title={Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings.},  
author={Peng, Nanyun and Dredze, Mark},  
booktitle={Processings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},  
pages={548–-554},  
year={2015}  
}  

@inproceedings{peng2016improving,  
title={Improving named entity recognition for Chinese social media with word segmentation representation learning},  
author={Peng, Nanyun and Dredze, Mark},  
booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)},  
volume={2},  
pages={149--155},  
year={2016}  
}  

## Dependencies:
This is an theano implementation; it requires installation of python module:  
Theano  
jieba (a Chinese word segmentor)  
Both of them can be simply installed by pip moduleName.

The lstm layer was adapted from http://deeplearning.net/tutorial/lstm.html and the feature extraction part was adapted from crfsuite: http://www.chokkan.org/software/crfsuite/

## running the EMNLP_15 experiments:
### Sample commands for the training:
python theano_src/crf_ner.py --nepochs 30 --neval_epochs 1 --training_data data/weiboNER.conll.train --valid_data data/weiboNER.conll.dev --test_data data/weiboNER.conll.test --emb_file embeddings/weibo_charpos_vectors --emb_type charpos --save_model_param weibo_best_parameters --emb_init true --eval_test False  

python theano_src/crf_ner.py --nepochs 30 --neval_epochs 1 --training_data data/weiboNER_2nd_conll.train --valid_data data/weiboNER_2nd_conll.dev --test_data data/weiboNER_2nd_conll.test --emb_file embeddings/weibo_charpos_vectors --emb_type char --save_model_param weibo_best_parameters --emb_init true --eval_test False  

In the above example, the output will be written at output_dir/weiboNER.conll.test.prediction. If you also want to see the evaluation (you must have labeled test data), you can add flag --eval_test True.  

### Sample commands for running the test:
python theano_src/crf_ner.py --test_data data/weiboNER.conll.test --only_test true --output_dir data/ --save_model_param weibo_best_parameters  

## running the ACL_16 experiments:
python theano_src/jointSegNER.py --cws_train_path data/pku_training.utf8 --cws_valid_path data/pku_test_gold.utf8 --cws_test_path data/pku_test_gold.utf8 --ner_train_path data/weiboNER_2nd_conll.train --ner_valid_path data/weiboNER_2nd_conll.dev --ner_test_path data/weiboNER_2nd_conll.test --emb_init file --emb_file embeddings/weibo_charpos_vectors --lr 0.05 --nepochs 30 --train_mode joint --cws_joint_weight 0.7 --m1_wemb1_dropout_rate 0.1

The last three parameters and the learning rate can be tuned. In our experiments, we found that for named mention, the best combination is (joint, 0.7, 0.1); for nonimal mention, the best combination is (alternative, 1.0, 0.1)

## Data
We noticed that several factors could affect the replicatability of experiments:  
1. the segmentor for preprocessing: we used jieba 0.37   
2. the random number generator. Alghough we fixed the random seed, we noticed it will render slight different numbers on different machine.  
3. the traditional lexical feature used.  
4. the pre-trained embeddings.
To enhance the replicatability of our experiments, we provide the original data in conll format at data/weiboNER.conll.(train/dev/test). In addition, we also provide files including all the features and the char-positional transformation we used in our experiments in data/crfsuite.weiboNER.charpos.conll.(train/dev/test), as well as the pre-trained char and char-positional embeddings.

Note: the data we provide contains both named and nominal mentions, you can get the dataset with only named entities by simply filtering out the nominal mentions.
