python pilot2_flatten_channel_mul.py --data_dir ../data/F1_MIMO_train_processed --features_file f1train_features_ch_mul.csv --labels_file train_labels.csv

python pilot2_flatten_channel_mul.py --data_dir ../data/F1_MIMO_test_processed --features_file f1test_features_ch_mul.csv --labels_file test_labels.csv

python pilot2_flatten_channel_mul.py --data_dir ../data/F2_MIMO_train_processed --features_file f2train_features_ch_mul.csv --labels_file train_labels.csv

python pilot2_flatten_channel_mul.py --data_dir ../data/F2_MIMO_test_processed --features_file f2test_features_ch_mul.csv --labels_file test_labels.csv



python pilot2_flatten_rphi.py --mode polar --train_dir ../data/F1_MIMO_train_processed --test_dir ../data/F1_MIMO_test_processed --train_features f1train_features_rphi.csv --test_features f1test_features_rphi.csv

python pilot2_flatten_rphi.py --mode polar_relative --train_dir ../data/F1_MIMO_train_processed --test_dir .../data/F1_MIMO_test_processed --train_features f1train_features_rdphi.csv --test_features f1test_features_rdphi.csv

python pilot2_flatten_rphi.py --mode polar --train_dir ../data/F2_MIMO_train_processed --test_dir ../data/F2_MIMO_test_processed --train_features f2train_features_rphi.csv --test_features f2test_features_rphi.csv

python pilot2_flatten_rphi.py --mode polar_relative --train_dir ../data/F2_MIMO_train_processed --test_dir ../data/F2_MIMO_test_processed --train_features f2train_features_rdphi.csv --test_features f2test_features_rdphi.csv

