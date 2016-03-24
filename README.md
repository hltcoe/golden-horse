# golden-horse
## The command for training is:
time python theano_src/crf_ner.py --nepochs 30 --neval_epochs 2 --training_data /export/projects/npeng/weiboNER_data/concrete_sighan_train_phrase.tar.gz --valid_data /export/projects/npeng/weiboNER_data/concrete_sighan_valid_phrase.tar.gz --test_data /export/projects/npeng/weiboNER_data/concrete_sighan_test_phrase.tar.gz --emb_file /export/projects/npeng/weiboNER_data/gigaword_charpos_vectors --emb_type charpos --save_model_param /export/projects/npeng/weiboNER_data/sighan_best_parameters --eval_test true

## For running the test, you need to do this:
time python theano_src/crf_ner.py --test_data /home/hltcoe/npeng/weiboNER/test_dir/concrete_weiboNER_test.tar.gz --only_test true --output_dir /export/projects/npeng/weiboNER_data/ --save_model_param best-parameters

This will load your test_data, load model from the file you passed through --save_model_param, do the prediction and write the output into the --output_dir you assigned.
