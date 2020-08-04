import stimuli as st
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix
from scripts import ssvep_utils as su
import numpy as np

def get_cor_template(X, ref, xweights, yweights):
    corr = []
    X = X.T
    Y =  np.squeeze(ref).T
    X_r = np.dot(X, xweights)
    Y = np.dot(Y, yweights)
    return np.corrcoef(X_r[:,0], Y[:,0])[0, 1]

def run(data, template, reference):
    n_components = 1
    corrs = []
    cca = CCA(n_components)
    for target in range(0, len(st.frequencies)):
        results = []
        X = data
        cor, _, xweights, yweights = su.find_correlation_for_one_pair(cca, 1, data,
                                                                      template[target, :, :])
        cor_ref, _, xweights_ref, yweights_ref = su.find_correlation_for_one_pair(cca, 1,
                                                                                 data,
                                                                                  reference[target, :, :])
        cor_ref_tem, _, xweights_ref_tem, yweights_ref_tem = su.find_correlation_for_one_pair(cca, 1,
                                                                                              np.squeeze(
                                                                                                  template[target, :, :]),
                                                                                              reference[target,
                                                                                              :, :])
        corr_t = get_cor_template(X, template[target, :, :], xweights_ref, xweights_ref)
        corr_ref_temp = get_cor_template(X, template[target, :, :], xweights_ref_tem, xweights_ref_tem)
        corr_temp = get_cor_template(np.squeeze(template[target, :, :]), template[target, :, :], xweights, yweights)

        results = [cor, cor_ref, corr_t, corr_ref_temp, corr_temp]
        corsum = map(lambda x: np.sign(x) * x ** 2, results)
        corsum = list(corsum)
        corsum = np.sum(corsum)
        corrs.append(np.sum(corsum))
    pre = np.argmax(corrs)
    return pre