mkdir data
mkdir data/external
mkdir data/external/covtype
mkdir data/external/kddcup99
mkdir data/external/glass_identification
mkdir data/external/rice

mkdir reports/figures

mkdir experiments/plots 

cd src/data
python load_data.py

cd ../../experiments

EXPERIMENTS=("kdd_param_opt.py" "covtype_param_opt.py" "glass_param_opt.py" "rice_param_opt.py")

INITIALIZATIONS=(
    "uniform"
    "beta"
    "normal"
    "normal_beta"
    "beta_mu"
    "zero"
)

NOISES=(
    "0"
    "1"
)

for experiment in "${EXPERIMENTS[@]}"
do
    for init in "${INITIALIZATIONS[@]}"
    do
        for noise in "${NOISES[@]}"
	    do
            if [ "$noise" == "0" ]
            then
                python "$experiment" --init "$init"
            else    
                python "$experiment" --init "$init" --enable_noise
            fi
	    done
    done
done

