# On Optimizing Hyperparameters for Quantum Neural Networks
* Authors: Sabrina Herbst, Vincenzo De Maio, Ivona Brandic
* Affiliation: TU Wien, Vienna, Austria

## Run Experiments
All experiments are executed using *Python 3.11.7*. 
1. Create a virtual environment
   - `python -m venv venv`
2. Activate the virtual environment
   - `source venv/bin/activate`
3. Install the required packages
   - `pip install -r requirements.txt`
4. Add IBM Quantum Account (necessary to get noise models). Refer to: https://docs.quantum-computing.ibm.com/run/account-management
5. Run Experiments
   - `bash run.sh`
6. Notebooks to analyze the results can be found in the `notebooks` folder 

## Folders
- `data`: Contains the data used for the experiments
   - `external`: Contains the external data used for the experiments. Will be filled when executing `run.sh`.
- `notebooks`: Contains the Jupyter notebooks used to analyze the results
  - `analyze_compare_results_[dataset].ipynb`: contains the comparison of noisy and noiseless experiments for the datasets
  - `analyze_noise_results_[dataset].ipynb`: contains the analysis of noisy experiments for the datasets
  - `analyze_plain_results_[dataset].ipynb`: contains the analysis of noiseless experiments for the datasets
  - `compare_init.ipynb`: adds additional comparison on initialization strategies
- `experiments`: Contains python and bash scripts to run all experiments
    - `[dataset]_param_opt.py`: Runs the parameter optimization for the cover type dataset, with and without noise
    - `plots`: contains convergence plots that are generated during execution
- `run.sh`: Runs all experiments
- `src`: Contains the source code for the experiments
   - `data`: Contains the data processing code
     - `load_data.py`: Loads the datasets and splits them into training, validation and test set
     - `data_encoding.py`: Contains functions to get qiskit data encoding circuits
   - `util`: Contains utility functions
     - `util.py`: Contains functions to get the configurations
     - `opt.py`: Contains routine for an exhaustive search of parameters
   - `models`: Contains the model code
     - `ansatzes.py`: Contains functions to get qiskit ansatzes
     - `optimizers.py`: Contains functions to get qiskit optimizers
- `reports`: Contains the results and plots
  - `figures`: Contains the figures that are generated during analysis
  - `results`: Contains all results
- `requirements.txt`: Contains the required python packages

## Data
1. Cover Type Dataset
   - Source: https://archive.ics.uci.edu/ml/datasets/Covertype
   - Description: The dataset contains information about different types of forest cover. The goal is to predict the type of forest cover for a given block of land.
   - References: 
     - Remote Sensing and GIS Program. Department of Forest Sciences. College of Natural Resources. Colorado State University
     - © Jock A. Blackard and Colorado State University.
2. KDDCup99 Dataset
   - Source: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
   - Description: The dataset contains information about network traffic. The goal is to predict the type of network traffic for a given block of network traffic.
   - References: 
     - The UCI KDD Archive: Information and Computer Science. University of California, Irvine
3. Glass Identification Dataset
   - Source: https://archive.ics.uci.edu/dataset/42/glass+identification
   - Description: The dataset contains data from the USA Forensic Science Service with the goal of predicting between six different types of glass.
   - References:
      - German,B.. (1987). Glass Identification. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P.
4. Rice Dataset
   - Source: https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
   - Description: The dataset contains features extracted from rice grain images to distinguish between two different types.
   - References:
      - Rice (Cammeo and Osmancik). (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5MW4Z.
      - Ilkay Cinar and Murat Koklu. 2019. Classification of Rice Varieties Using Artificial Intelligence Methods. International Journal of Intelligent Systems and Applications in Engineering 7 (09 2019), 188–194. https://doi.org/10.18201/ijisae.2019355381
