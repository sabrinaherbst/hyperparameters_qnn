import pandas as pd
from models.ansatzes import get_paulitwodesign, get_twolocal
from data.data_encoding import get_zfeaturemap
from models.optimizers import get_cobyla, get_neldermead, get_spsa

def get_configurations(ansatzes, featuremaps, entanglement_twolocal_featuremap, entanglement_other, iters, beta_init = False, normal_init = False, normal_init_beta_dist = False, beta_mu = False, zero_init = False):
    """
    Create all possible configurations for the VQCs

    Parameters
    ----------
    ansatzes : list of functions of considered ansatzes
    featuremaps : list of functions of considered featuremaps
    entanglement_twolocal_featuremap : list of entanglement types for the twolocal featuremap
    entanglement_other : list of entanglement types for the other featuremaps
    iters : list of iterations for the optimizers

    Returns
    -------
    lists for each optimizer with all possible configurations
    """
    iter_cobyla, iter_spsa, iter_neldermead = iters

    vqc_cobyla = []
    vqc_spsa = []
    vqc_neldermead = []

    for ansatz in ansatzes:
        for featuremap in featuremaps:
            if ansatz != get_paulitwodesign:
                for ent in (
                    entanglement_other
                    if ansatz != get_twolocal
                    else entanglement_twolocal_featuremap
                ):
                    if featuremap != get_zfeaturemap:
                        for ent2 in entanglement_twolocal_featuremap:
                            vqc_spsa.append([ansatz, get_spsa, iter_spsa, featuremap, ent, ent2, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init])
                            vqc_neldermead.append(
                                [ansatz, get_neldermead, iter_neldermead, featuremap, ent, ent2, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init]
                            )
                            vqc_cobyla.append(
                                [ansatz, get_cobyla, iter_cobyla, featuremap, ent, ent2, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init]
                            )
                    else:
                        vqc_neldermead.append(
                            [ansatz, get_neldermead, iter_neldermead, featuremap, ent, None, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init]
                        )
                        vqc_spsa.append([ansatz, get_spsa, iter_spsa, featuremap, ent, None, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init])
                        vqc_cobyla.append([ansatz, get_cobyla, iter_cobyla, featuremap, ent, None, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init])
            else:
                if featuremap != get_zfeaturemap:
                    for ent2 in entanglement_twolocal_featuremap:
                        vqc_neldermead.append(
                            [ansatz, get_neldermead, iter_neldermead, featuremap, None, ent2, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init]
                        )
                        vqc_spsa.append([ansatz, get_spsa, iter_spsa, featuremap, None, ent2, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init])
                        vqc_cobyla.append(
                            [ansatz, get_cobyla, iter_cobyla, featuremap, None, ent2, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init]
                        )
                else:
                    vqc_neldermead.append(
                        [ansatz, get_neldermead, iter_neldermead, featuremap, None, None, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init]
                    )
                    vqc_spsa.append([ansatz, get_spsa, iter_spsa, featuremap, None, None, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init])
                    vqc_cobyla.append([ansatz, get_cobyla, iter_cobyla, featuremap, None, None, beta_init, normal_init, normal_init_beta_dist, beta_mu, zero_init])

    columns = ["ansatz",
            "optimizer",
            "iterations",
            "featuremap",
            "entanglement",
            "entanglement_featuremap",
            "beta_init",
            "normal_init",
            "normal_init_beta_dist",
            "beta_mu",
            "zero_init"]

    # create dataframe
    vqcs_df_cobyla = (
        pd.DataFrame(
            vqc_cobyla,
            columns=columns)
        .reset_index().values.tolist())

    vqcs_df_spsa = (
        pd.DataFrame(
            vqc_spsa,
            columns=columns)
        .reset_index().values.tolist())

    vqcs_df_neldermead = (
        pd.DataFrame(
            vqc_neldermead,
            columns=columns)
        .reset_index().values.tolist())
    
    return vqcs_df_cobyla, vqcs_df_neldermead, vqcs_df_spsa