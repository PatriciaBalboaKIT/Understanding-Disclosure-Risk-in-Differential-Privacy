import sys
import os
import argparse
import pandas as pd
import multiprocessing as mp
import numpy as np
import warnings
import random
import csv

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logs
mp.set_start_method('spawn', force=True)

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)


from core.data_util import sample_noniid_data
from core.data_util import load_data, load_data_new
from core.data_util import get_sensitive_features
from core.data_util import process_features
from core.data_util import threat_model
from core.data_util import subsample
from core.utilities import imputation_training
from core.attack import train_target_model
from core.attack import whitebox_attack
from core.attack import yeom_membership_inference
from core.classifier import get_predictions
from core.classifier import get_layer_outputs
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

RESULT_PATH = 'results/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


def run_experiment(args):
    print("\n\nGPUs used:")
    print(f"{tf.config.list_physical_devices('GPU')}\n\n")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.exists(RESULT_PATH + args.train_dataset):
        os.makedirs(RESULT_PATH + args.train_dataset)
    
    MODEL = str(args.skew_attribute) + '_' + str(args.skew_outcome) + '_' + str(args.target_test_train_ratio) + '_' + str(args.target_model) + '_'
    
    #train_x, train_y, test_x, test_y = load_data('target_data.npz', args)
    D_minus_x, D_minus_y, pool_x, pool_y, test_x, test_y = load_data_new('target_data.npz', args)
    h_train_x, h_train_y, h_test_x, h_test_y = load_data('holdout_data.npz', args)
    sk_train_x, sk_train_y, sk_test_x, sk_test_y = load_data('skewed_data.npz', args)
    sk2_train_x, sk2_train_y, sk2_test_x, sk2_test_y = load_data('skewed_2_data.npz', args)
    
    ############ RERO ###################
    reros = []
    corrections = []
    for i in range(5):
        SEED = i
        random.seed(SEED)
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        D_x = np.vstack((D_minus_x, pool_x))
        D_y = np.append(D_minus_y, pool_y)

        true_x = np.vstack((D_x, test_x, h_train_x, h_test_x, sk_train_x, sk_test_x, sk2_train_x, sk2_test_x))
        true_y = np.concatenate((D_y, test_y, h_train_y, h_test_y, sk_train_y, sk_test_y, sk2_train_y, sk2_test_y))
        
        c_size = args.candidate_size
        train_c_idx, test_c_idx, h_test_idx, sk_test_idx, sk2_test_idx, adv_known_idxs = threat_model(args, len(true_x))
        adv_known_idxs["high"] = [idx for idx in adv_known_idxs["high"] if idx < len(D_minus_x)]

        assert(args.attribute < 3)
        target_attrs, attribute_dict, max_attr_vals, col_flags = get_sensitive_features(args.train_dataset, D_x)
        target_attr = target_attrs[args.attribute]
        labels = [0, 1] if attribute_dict == None else list(attribute_dict[target_attr].keys())
        train_idx = range(len(D_x))

        sensitive_test = true_x[:, target_attr] * max_attr_vals[target_attr]
        known_test = process_features(true_x, args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_sensitive=True, skip_corr=args.skip_corr)

        prior_prob = np.zeros(len(labels))
        for k, v in Counter(sensitive_test).items():
            prior_prob[int(k)] = v / len(sensitive_test)

        threat_level = 'high'
        sample_size = 50000
        adv_known_idx_ = adv_known_idxs[threat_level]
        adv_known_idx = subsample(adv_known_idx_, true_x[adv_known_idx_, target_attr] * max_attr_vals[target_attr], sample_size)
        sensitive_train = true_x[adv_known_idx, target_attr] * max_attr_vals[target_attr]
        known_train = process_features(true_x[adv_known_idx], args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_sensitive=True, skip_corr=args.skip_corr)
        
        # Train imputation model on D_
        imp_clf, imp_conf, imp_aux = imputation_training(args, known_train, sensitive_train, known_test, sensitive_test, clf_type='nn', epochs=10)
        
        # Test on all samples from the pool
        ip_test = process_features(pool_x, args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_sensitive=True, skip_corr=args.skip_corr)
        ip_sensitive_test = pool_x[:, target_attr] * max_attr_vals[target_attr]
        
        imp_conf_test = imp_clf.predict_proba(ip_test)
        y_pred = np.argmax(imp_conf_test, axis=1)
        correct_mask = (y_pred == ip_sensitive_test)
        num_correct = np.sum(correct_mask)
        rero_ip = num_correct / len(pool_x)
        print(f"ReRo IP: {rero_ip}")

        
        ########################### Correction term ###########################
        x_targets = pool_x
        y_targets = pool_y
        print(f"Size: {len(pool_x)}")

        # Combine D_minus with targets
        D_attack_x = np.vstack((D_minus_x, x_targets))
        D_attack_y = np.append(D_minus_y, y_targets)

        # Indices of targets within D_attack
        target_start_idx = len(D_minus_x)
        target_end_idx = target_start_idx + len(pool_x)

        # Check that targets match positions in D_attack
        assert np.all(x_targets == D_attack_x[target_start_idx:target_end_idx])
        assert np.all(y_targets == D_attack_y[target_start_idx:target_end_idx])

        # Combine with all test sets for full attack model training
        true_attack_x = np.vstack((
            D_attack_x,
            test_x,
            h_train_x, h_test_x,
            sk_train_x, sk_test_x,
            sk2_train_x, sk2_test_x
        ))

        true_attack_y = np.concatenate((
            D_attack_y,
            test_y,
            h_train_y, h_test_y,
            sk_train_y, sk_test_y,
            sk2_train_y, sk2_test_y
        ))

        # Double-check targets are still correctly placed
        assert np.all(x_targets == true_attack_x[target_start_idx:target_end_idx])
        assert np.all(y_targets == true_attack_y[target_start_idx:target_end_idx])

        train_c_idx, test_c_idx, h_test_idx, sk_test_idx, sk2_test_idx, adv_known_idxs = threat_model(args, len(true_attack_x))
        adv_known_idxs["high"] = [
            idx for idx in adv_known_idxs["high"]
            if idx < target_start_idx
        ]

        target_attrs, attribute_dict, max_attr_vals, col_flags = get_sensitive_features(args.train_dataset, D_attack_x)
        target_attr = target_attrs[args.attribute]
        labels = [0, 1] if attribute_dict == None else list(attribute_dict[target_attr].keys())

        sensitive_test = true_attack_x[:, target_attr] * max_attr_vals[target_attr]
        known_test = process_features(true_attack_x, args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_sensitive=True, skip_corr=args.skip_corr)

        # use the imputation model
        imp_conf = imp_clf.predict_proba(known_test)

        for k in range(len(pool_x)):
            idx = target_start_idx + k
            if D_attack_x[idx, target_attr] != true_attack_x[idx, target_attr]:
                print(f"  Original Data: {D_attack_x[idx, target_attr]}")
                print(f"  X_true:        {true_attack_x[idx, target_attr]}")
                print("PROBLEM")

        # gather correction for all targets
        correction_ip = 0
        correction_ips = 0
        for k in range(len(pool_x)):
            idx = target_start_idx + k  # index into true_attack_x and all prediction arrays
            true_sensitive = sensitive_test[idx]  # extract true sensitive attribute value
            correction_ips += (true_sensitive == np.argmax(imp_conf[idx]))
        # average across all non-member targets
        correction_ip += correction_ips / len(pool_x)
        print(f"Interim Correction: IP={correction_ip}")
        reros.append(rero_ip)
        corrections.append(correction_ip)

    # Write results
    with open(f"./results/results_dataset{args.train_dataset}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ReRo"] + [np.mean(reros)])
        writer.writerow(["RAD"] + [np.mean(reros) - np.mean(corrections)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--use_cpu', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0, help='0: no save data, instead run experiments (default), 1: save data')
    parser.add_argument('--attribute', type=int, default=0, help='senstive attribute to use: 0, 1 or 2')
    parser.add_argument('--candidate_size', type=int, default=int(1e4), help='candidate set size')
    parser.add_argument('--skew_attribute', type=int, default=0, help='Attribute on which to skew the non-iid data sampling 0 (population, default), 1 or 2 -- for Census 1: Income and 2: Race, and for Texas 1: Charges and 2: Ethnicity')
    parser.add_argument('--skew_outcome', type=int, default=0, help='In case skew_attribute = 2, which outcome to skew the distribution upon -- For Census Race: 0 (White, default), 1 (Black) or 3 (Asian), and for Texas Ethnicity: 0 (Hispanic, default) or 1 (Not Hispanic)')
    parser.add_argument('--sensitive_outcome', type=int, default=0, help='In case skew_attribute = 2, this indicates the sensitive outcome -- For Census Race: 0 (White, default), 1 (Black) or 3 (Asian), and for Texas Ethnicity: 0 (Hispanic, default) or 1 (Not Hispanic)')
    parser.add_argument('--banished_records', type=int, default=0, help='if the set of records in banished.p file are to be removed from model training (default:0 no records are removed)')
    parser.add_argument('--skip_corr', type=int, default=0, help='For Texas-100X, whether to skip Race (or Ethnicity) when the target sensitive attribute is Ethnicity (or Race) -- default is not to skip (0)')
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(5e4))
    parser.add_argument('--target_test_train_ratio', type=float, default=0.5)
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-8)
    parser.add_argument('--target_clipping_threshold', type=float, default=1.0)
    parser.add_argument('--target_privacy', type=str, default='no_privacy')
    parser.add_argument('--target_dp', type=str, default='gdp')
    parser.add_argument('--target_epsilon', type=float, default=0.5)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='nn')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=64)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args()
    print(vars(args))
    
    # Flag to disable GPU
    if args.use_cpu:
    	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    if args.save_data == 1:
        sample_noniid_data(args)
    else:
        run_experiment(args)
