import argparse
import sys

sys.path.append("../src")

import warnings

import pandas as pd

from data.data_encoding import get_zfeaturemap, get_zzfeaturemap
from models.ansatzes import (get_efficientsu2, get_paulitwodesign,
                             get_realamplitudes, get_twolocal)

from util.opt import HyperparameterOptimisation
from util.util import get_configurations

# FutureWarnings are thrown by sklearn
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--enable_noise", type=bool, default=False, help="wheter to use noise model")
parser.add_argument("--train_samples", type=int, default=400, help="number of training samples")
parser.add_argument("--init", type=str, default="uniform", help="initialization method [beta, normal, normal_beta, beta_mu, uniform, zero]")
opt = parser.parse_args()

noise = opt.enable_noise
train_samples = opt.train_samples
beta_init = opt.init == "beta"
normal_init = opt.init == "normal"
normal_init_beta_dist = opt.init == "normal_beta"
beta_mu = opt.init == "beta_mu"
zero_init = opt.init == "zero"

# if more than one initialization is set to true, raise error
if beta_init and normal_init_beta_dist:
    raise ValueError("Cannot use both beta and normal initialization with beta mu and sigma")
if normal_init and normal_init_beta_dist:
    raise ValueError("Cannot use both normal and normal initialization with beta mu and sigma")
if beta_init and normal_init:
    raise ValueError("Cannot use both beta and normal initialization")

# load the data
train = pd.read_csv("../data/external/kddcup99/kddcup99_train.csv")
val = pd.read_csv("../data/external/kddcup99/kddcup99_val.csv")
test = pd.read_csv("../data/external/kddcup99/kddcup99_test.csv")

print(
    f"Training Samples: {train.shape[0]}\nValidation Samples: {val.shape[0]}\nTest Samples: {test.shape[0]}\nBeta Initialization: {beta_init}\nNormal Initialization: {normal_init}\nNormal Initialization with Beta Mu and Sigma: {normal_init_beta_dist}\nIntialize to Mu of Beta: {beta_mu}\nZero Initialization: {zero_init}\nNoise: {noise}"
)

# encode protocol_type, service, and flag
train = pd.get_dummies(train, columns=["protocol_type", "service", "flag"])
val = pd.get_dummies(val, columns=["protocol_type", "service", "flag"])
test = pd.get_dummies(test, columns=["protocol_type", "service", "flag"])

# add zeros for missing columns
missing_cols = set(train.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[train.columns]

# same for validation set
missing_cols = set(train.columns) - set(val.columns)
for c in missing_cols:
    val[c] = 0
val = val[train.columns]

# delete additional columns
cols = set(test.columns) - set(train.columns)
for c in cols:
    del test[c]

cols = set(val.columns) - set(train.columns)
for c in cols:
    del val[c]

# delete highly correlated features
corrs = (
    train.corr(numeric_only=True)
    .abs()
    .unstack()
    .sort_values(ascending=False)
    .drop_duplicates()
)
corrs = corrs[(corrs > 0.9) & (corrs < 1)]

# drop one of the featurs with correlation > 90%
to_drop = [idx[0] for idx in corrs.index]
train.drop(to_drop, axis=1, inplace=True)
val.drop(to_drop, axis=1, inplace=True)
test.drop(to_drop, axis=1, inplace=True)

# get x and y
X_train = train.drop("target", axis=1)
y_train = train["target"]
X_val = val.drop("target", axis=1)
y_val = val["target"]
X_test = test.drop("target", axis=1)
y_test = test["target"]

# create vqcs for all combinations
ansatzes = [
    get_efficientsu2,
    get_paulitwodesign,
    get_realamplitudes,
    get_twolocal,
]

featuremaps = [get_zfeaturemap, get_zzfeaturemap]
entanglement_twolocal_featuremap = ["full", "linear", "circular", "pairwise", "sca"]

entanglement_other = ["full", "linear", "circular", "sca"]

vqcs_coybla = []
vqc_spsa = []
vqc_neldermead = []

if noise:
    iter_cobyla = 250
    iter_spsa = 125
    iter_neldermead = 100
else:
    iter_cobyla = 500
    iter_spsa = 250
    iter_neldermead = 250

vqcs_df_cobyla, vqcs_df_neldermead, vqcs_df_spsa = get_configurations(ansatzes, featuremaps, entanglement_twolocal_featuremap, entanglement_other, [iter_cobyla, iter_neldermead, iter_spsa], beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init)

# get vqcs with indices
print(f"Optimizing {len(vqcs_df_cobyla)} configs per optimizer")

# run hyperparameter optimisation
hyopt = HyperparameterOptimisation(
    "kdd",
    [vqcs_df_cobyla, vqcs_df_neldermead if not normal_init_beta_dist and not beta_mu and not zero_init else [], vqcs_df_spsa],
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    train_samples=train_samples,
    noise_model_qc='ibm_perth' if noise else None
)
hyopt.run()
