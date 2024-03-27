import multiprocessing
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from qiskit_aer import QasmSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider
from qiskit.primitives import BackendSampler, BackendEstimator
from qiskit.providers.fake_provider import FakePerth

from qiskit_machine_learning import QiskitMachineLearningError
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from scipy.stats import beta

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler

from qiskit_machine_learning.algorithms import VQC, VQR
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import qiskit


class HyperparameterOptimisation():
    """
    Implements hyperparameter optimisation for VQC and VQR models.
    """

    def __init__(self, dsname, configurations, X_train, y_train, X_val, y_val, X_test, y_test, train_samples=400, regression=False, noise_model_qc = None):
        """
        Args:
            dsname (str): name of the dataset
            configurations (list): list of configurations to optimise
            X_train (pd.DataFrame): training data
            y_train (pd.DataFrame): training labels
            X_val (pd.DataFrame): validation data
            y_val (pd.DataFrame): validation labels
            X_test (pd.DataFrame): test data
            y_test (pd.DataFrame): test labels
            train_samples (int): number of samples to use for training
            regression (bool): whether to use regression or classification
            noise_model_qc (str): name of the noise model to use
        """
        self.configurations = configurations
        self.dsname = dsname
        self.regression = regression

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.train_samples = train_samples
        self.n_components = 7
        self.noise_model_qc = noise_model_qc

        if self.train_samples > self.X_train.shape[0]:
            self.train_samples = self.X_train.shape[0]

        if not self.noise_model_qc is None:
            if self.noise_model_qc == 'fake_perth':
                b = FakePerth()
            else:
                b = IBMProvider().get_backend(self.noise_model_qc)
            self.noise_model = NoiseModel.from_backend(b)
        else:
            self.noise_model = None

        self.scaler, self.pca, self.lda = self.initialise()
        self.preprocess_data()

    def initialise(self):
        """
        Initialise the scaler, pca, lda and sampler.
        """
        # create scaler for pca
        scaler = StandardScaler()
        scaler.fit(self.X_train)

        X_train_scaled = pd.DataFrame(scaler.transform(self.X_train), index=None)

        # create pca
        pca = PCA(n_components=7)
        pca.fit(X_train_scaled, self.y_train)

        # create lda
        if not self.regression:
            # if number of classes is less than 8, use n_components = n_classes - 1
            if self.y_train.nunique() < 8:
                self.n_components = self.y_train.nunique() - 1
                lda = LinearDiscriminantAnalysis(n_components=self.n_components)
            else:
                lda = LinearDiscriminantAnalysis(n_components=self.n_components)
            lda.fit(self.X_train, self.y_train)
        else:
            lda = None

        return scaler, pca, lda

    def preprocess_data(self):
        """
        Preprocess the data for hyperparameter optimisation.
        """
        # sample subset used for hyperparameter optimisation
        X_train_tune = self.X_train.sample(self.train_samples, random_state=12)
        self.y_train_tune = self.y_train.sample(self.train_samples, random_state=12)

        self.X_train_tune_pca = pd.DataFrame(self.pca.transform(self.scaler.transform(X_train_tune)))
        if not self.regression:
            self.X_train_tune_lda = pd.DataFrame(self.lda.transform(X_train_tune))

        val_samples = 250 if self.X_val.shape[0] > 250 else self.X_val.shape[0]
        X_val_tune = self.X_val.sample(val_samples, random_state=12)
        self.y_val_tune = self.y_val.sample(val_samples, random_state=12)

        self.X_val_tune_pca = pd.DataFrame(self.pca.transform(self.scaler.transform(X_val_tune)))
        if not self.regression:
            self.X_val_tune_lda = pd.DataFrame(self.lda.transform(X_val_tune))

    def create_vqc(self, vqc, lda=False):
        """
        Create a VQC or VQR model.
        """
        _, ansatz, optimizer, iterations, featuremap, entanglement, entanglement_featuremap,_,_,_,_,_ = vqc

        # create ansatz
        if entanglement != None:
            ansatz = ansatz(entanglement=entanglement, num_qubits=self.n_components if lda else 7)
        else:
            ansatz = ansatz(num_qubits=self.n_components if lda else 7)

        # create featuremap
        if entanglement_featuremap != None:
            featuremap = featuremap(entanglement=entanglement_featuremap, feature_dimension=self.n_components if lda else 7)
        else:
            featuremap = featuremap(feature_dimension=self.n_components if lda else 7)

        optimizer = optimizer(maxiter=iterations, tol=0.1)

        if not self.noise_model is None:
             backend = QasmSimulator(noise_model = self.noise_model)
        else:
             backend = QasmSimulator()

        if self.regression:
            vqc = VQR(ansatz=ansatz, optimizer=optimizer, feature_map=featuremap, estimator=BackendEstimator(backend=backend))
        else:
            vqc = VQC(ansatz=ansatz, optimizer=optimizer, feature_map=featuremap, sampler=BackendSampler(backend=backend))
        return vqc

    # adapted from https://github.com/aicaffeinelife/BEINIT/blob/main/train.py
    def init_beta(self, X_train, num_params, get_params = False):
        data_r = X_train.reshape(X_train.size)
        sz = num_params
        data_r[data_r<=0] = data_r[data_r <= 0] + 1e-8
        data_r[data_r>=1] = data_r[data_r >= 1] - 1e-8
        a, b, _, _ = beta.fit(data_r, floc=0, fscale=1)
        print(f"Found alpha:{a}, beta:{b}")
        return np.random.beta(a=a, b=b, size=sz) if not get_params else (a, b)


    def train_vqc(self, vqc, lda=False):
        """
        Train a VQC or VQR model.
        """
        # callback function to record objective function evaluations
        vals = []
        def callback(weights, obj_func_eval):
            vals.append(obj_func_eval)

        ind, ansatz, optimizer, iterations, featuremap, entanglement, entanglement_featuremap, beta_init, normal_init, normal_mu_sigma_init, beta_mu, zero_init = vqc
        print(f"Fitting VQC {ind}")

        vqc = self.create_vqc(vqc, lda=lda)

        vqc.callback = callback 
        if lda:         
            X_train_use = self.X_train_tune_lda.to_numpy()
            X_val_use = self.X_val_tune_lda.to_numpy()
        else:
            X_train_use = self.X_train_tune_pca.to_numpy()
            X_val_use = self.X_val_tune_pca.to_numpy()

        if beta_init:
            scaler = MinMaxScaler()
            X_train_use = scaler.fit_transform(X_train_use)
            X_val_use = scaler.transform(X_val_use)
            vqc.ansatz.assign_parameters(self.init_beta(X_train_use, vqc.ansatz.num_parameters))
        elif normal_init:
            vqc.ansatz.assign_parameters(np.random.normal(loc=0.0, scale=np.sqrt(1/vqc.ansatz.num_layers), size=vqc.ansatz.num_parameters))
        elif zero_init:
            vqc.ansatz.assign_parameters(np.zeros(vqc.ansatz.num_parameters))
        elif normal_mu_sigma_init or beta_mu:
            scaler = MinMaxScaler()
            X_train_use = scaler.fit_transform(X_train_use)
            X_val_use = scaler.transform(X_val_use)
            a, b = self.init_beta(X_train_use, vqc.ansatz.num_parameters, get_params=True)
            # get mu and sd from beta distribution
            mu = a/(a+b)
            sd = np.sqrt((a*b)/((a+b)**2*(a+b+1)))
            if beta_mu:
                vqc.ansatz.assign_parameters([mu]*vqc.ansatz.num_parameters)
            else:
                vqc.ansatz.assign_parameters(np.random.normal(loc=mu, scale=sd, size=vqc.ansatz.num_parameters))
            

        try:
            start = time.time()

            vqc.fit(X_train_use, self.y_train_tune.to_numpy())
            end = time.time()

            # evaluate accuracy and f1
            y_pred = vqc.predict(X_val_use)

            if self.regression:
                mse = mean_squared_error(self.y_val_tune, y_pred)
                mae = mean_absolute_error(self.y_val_tune, y_pred)
                print(f"Fitted VQR {ind} with mse {mse} and mae {mae}")
            else:
                acc = accuracy_score(self.y_val_tune, y_pred)
                f1 = f1_score(self.y_val_tune, y_pred, average="weighted")
                print(f"Fitted VQC {ind} with acc {acc} and f1 {f1}")
            empty = ""

            if len(vals) > 0:
                plt.plot(vals)
                plt.title(f"VQC {ind}")
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.savefig(f"plots/{self.dsname}_{self.noise_model_qc if not self.noise_model_qc is None else empty}_{ind}_{optimizer.__name__}_{lda}_{beta_init}_{normal_init}_{normal_mu_sigma_init}_{beta_mu}_{zero_init}.png")
                plt.close()

            l = [ind, acc if not self.regression else mse, f1 if not self.regression else mae, end-start, str(ansatz), str(optimizer), str(featuremap), entanglement, entanglement_featuremap, len(vals), "pca" if not lda else "lda"]

            # with open(f"{self.dsname}_{ind}_{self.beta_init}_{self.normal_init}.txt", "w") as fp:
            #    fp.write(str(l))

            return [
                ind,
                acc if not self.regression else mse,
                f1 if not self.regression else mae,
                end - start,
                str(ansatz),
                str(optimizer),
                str(featuremap),
                entanglement,
                entanglement_featuremap,
                len(vals),
                "pca" if not lda else "lda",
            ]
        except FileNotFoundError as e:
            print(
                f"Error in VQC {ind} {str(ansatz)} {str(optimizer)} {str(featuremap)} {entanglement} {entanglement_featuremap}: {e}"
            )
            return [
                ind,
                None,
                None,
                None,
                str(ansatz),
                str(optimizer),
                str(featuremap),
                entanglement,
                entanglement_featuremap,
                len(vals),
                "pca" if not lda else "lda",
            ]


    def train_vqc_lda(self, vqc):
        """
        Train a VQC or VQR model with LDA preprocessing.
        """
        return self.train_vqc(vqc, lda=True)
    
    def run(self):
        """
        Run the experiment.
        """
        vqcs_cobyla, vqcs_neldermead, vqcs_spsa = self.configurations[0], self.configurations[1], self.configurations[2]
        self.beta_init = vqcs_cobyla[0][-5]
        self.normal_init = vqcs_cobyla[0][-4]
        self.normal_mu_sigma_init = vqcs_cobyla[0][-3]
        self.beta_mu_init = vqcs_cobyla[0][-2]
        self.zero_init = vqcs_cobyla[0][-1]
        
        for vqcs, name in zip([vqcs_cobyla, vqcs_spsa, vqcs_neldermead], ['cobyla', 'spsa', 'neldermead']):
            for preproc_name in ["pca", "lda"]:
                if self.regression and preproc_name == "lda":
                    continue

                print(f"Starting optimization with optimizer {name} and preprocessing {preproc_name} and noise model {self.noise_model_qc}")
                
                start = time.time()
                
                with multiprocessing.Pool(processes=16) as pool:
                    if preproc_name == "lda":
                        if self.dsname == "rice":
                            continue
                        results = pool.map(self.train_vqc_lda, vqcs)
                    else:
                        results = pool.map(self.train_vqc, vqcs)
                
                end = time.time()

                print(f"Training took {end-start} seconds")

                # save results
                results_df = pd.DataFrame(
                    results,
                    columns=[
                        "index",
                        "accuracy" if not self.regression else "mse",
                        "f1" if not self.regression else "mae",
                        "time",
                        "ansatz",
                        "optimizer",
                        "featuremap",
                        "entanglement",
                        "entanglement_featuremap",
                        "iterations",
                        "preprocessing",
                    ],
                )
                empty = "_"
                results_df.to_csv(
                    f"../reports/results/{self.dsname}_{'ibm_perth' if not self.noise_model_qc is None else empty}_{name}_{preproc_name}_{'beta' if self.beta_init else 'normal' if self.normal_init else 'normal_beta_dist' if self.normal_mu_sigma_init else 'beta_mu' if self.beta_mu_init else 'zero' if self.zero_init else 'uniform'}_results.csv",
                    index=False,
                )
