

def st_weights_fit(X1,X2,Q,zscore=True):
    '''Fit the weights,w, across the sensors to produce a target similarity 
    trajectory, TRG, for space-time measures B1 and B2.

    INPUT

    B1 = spatiotemporal data [Nchan x Nt1]
    B2 = spatiotemporal data [Nchan x Nt2]
    TRG = desired similarity trajectory [Nt x Nt]

    OUTPUT
    w = weights across channels producing esimated match to TRG    [Nchan x 1]
    Z = the similarity trajectory matrix arising from the best-fit weights, w
    '''
    import numpy as np
    import scipy as sci

    (Nchan, Nt1) = np.shape(X1);

    if zscore:
        X1 = sci.stats.zscore(X1)
        X2 = sci.stats.zscore(X2)
        print('zscoring....')

    G = np.linalg.pinv(X2)
    Z = Q.dot(G)

    W = np.zeros(Nchan);
    np.shape(W)

    for n in range(Nchan):
        z = Z[:,n]
        x1 = X1[n,:]
        out = np.linalg.lstsq(x1[:, None], z[:, None])
        W[n,] = out[0]

    W = np.diag(W)
    Z = np.dot(np.dot(X1.T, W),X2);

    return W, Z