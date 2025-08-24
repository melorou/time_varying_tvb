# ======= Full Correlation Matrix =======
def Corr_func(x, Regul=False, Threshold=0, RowVar=False):   
    # x: signals;  x.shape = time x variables

    import numpy as np
    n, m = x.shape
    corr = np.corrcoef(x, rowvar=RowVar)

    if Regul==True: 
        lmbd = np.sort(np.linalg.eigvals(corr))
        flag=1
        for i in range(m):
            if (lmbd[i]>1e-10) * (flag==1):
                lmbdp = lmbd[i]
                flag=0
        lmbd1 = lmbd[-1]
        dlta = np.max([0, (lmbd1-50*lmbdp)/49])
        print(f"Regularization factor = {dlta}")

        corr = ( corr + dlta*np.eye(m) ) / ( 1+dlta )

    corr[abs(corr)<=Threshold] = 0


    return corr



# ======= Partial Correlation Matrix =======
def PCorr_func(x):      
    # x: signals;  x.shape = time x variables
    
    import numpy as np
    n, m = x.shape
    corr = np.corrcoef(x, rowvar=False)
    corrinv = np.linalg.inv(corr)
    pcorr = np.zeros_like(corr)

    for i in range(m):
        for j in range(m):
            if i!=j: pcorr[i,j] = -corrinv[i,j]/np.sqrt( corrinv[i,i]*corrinv[j,j] )
            if i==j: pcorr[i,j] = 1
    
    
    return pcorr



# ======= Regularized Inverse Covariance (Percision) Matrix =======
def Prec_func(x, alpha=0.05, tol=1e-4):
    # x: signals;  x.shape = time x variables
    # alpha: regularization param
    
    x = ( x - x.mean() )/x.std(axis=0)
    from sklearn.covariance import GraphicalLasso
    
    model = GraphicalLasso(alpha=alpha, tol=tol)
    model.fit(x)
    prec = -model.precision_


    return prec



# ======= Magnitude Squared Coherence Matrix =======
# mean over all frequencies (based on smith-2011, mean over all freq.s works best!)
def Coh_func(x, fs=100, n_f=100, window_length=15, normalized_freq_band_for_avg=[0, 1]):
    # x: signals;  x.shape = time x variables
    # window_length: length of the hann window (dafault: 15)
    # normalized_freq_band_for_avg: frequency band normalized by nyquist frequency = fs/2 (maximum frequency of the signals)
    #                               which is used for averaging the COH matrices (dafault: [0,1] over all frequencies)

    import numpy as np
    import scipy as sp

    n,m = x.shape
    
    mywindow = sp.signal.windows.hann(window_length)
    
    freqs = np.linspace(0, fs/2, n_f+1)
    f1 = normalized_freq_band_for_avg[0]*freqs[-1]
    f2 = normalized_freq_band_for_avg[1]*freqs[-1]
    avg_weights = (freqs>=f1) * (freqs<=f2) * 1
    
    COH = np.zeros([m, m])
    for i in range(m):
        for j in range(i, m):
            _, coh_xy = sp.signal.coherence(x[:, i], x[:, j], window=mywindow, nfft=2*n_f)
            coh_xy_avg = np.average( coh_xy, weights=avg_weights)

            
            COH[i, j] = coh_xy_avg
            COH[j, i] = coh_xy_avg
    

    return COH



# ======= Mutual Information =======
# normalization isn't required
def MI_func(x):
    # x: signals;  x.shape = time x variables
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression
    n, m = x.shape
    if n<m: print('data in bad shape!'); x = x.T; n,m = x.shape

    MI = np.zeros([m,m])
    for i in range(m):
        mi = mutual_info_regression(x, x[:,i], discrete_features=False)
        MI[i, :] = mi


    return MI



# ======= zero-lag (instantaneous) Regression =======
def ZeroLagReg_func(x, GS_reg=False, p_test=False):
    # x: signals;  x.shape = time x variables
    # GS_reg: Global signal regression; if True: GS (all signals average) will be added to regressors (default: False)
    # p_test: p-Value test; if True: only significant coefficients (Significance level=5%) will be returned (default: False)
    # task signal (even if available) is not used here!


    import statsmodels.api as sm
    import numpy as np
    signif_level = 0.05
    
    n, m   = x.shape
    GS = np.mean(x, axis=1).reshape([n, 1])
    
    result = np.zeros([m, m])
    for i in range(m):
        
        Y = x[:, i].reshape([n,1])
        Y = Y - np.mean(Y, axis=0)
        other_ROIs = np.delete(x, i, axis=1)
        other_ROIs = other_ROIs - np.mean(other_ROIs, axis=0)
        X = np.hstack( (other_ROIs, np.ones([n, 1])) )
        if GS_reg == True:
            X = np.hstack( (other_ROIs, GS) )
            
                
        est = sm.OLS(Y,X).fit().summary2().tables[1]
        beta = np.array(est['Coef.'])
        pVal = np.array(est['P>|t|'])
        
        if p_test==True:
            beta[pVal>signif_level] = 0

        result[i,:] = np.hstack( (beta[:i], [0], beta[i:m-1]) )
    

    return result



# ======= Pair-wise Granger Causality =======
# normalization isn't Rrequired
def pwGC_func(x, maxLag=10, Oselec=True, Oselec_method='bic', show_order_selection_results=False):
    # x: signals; x.shape = time x variables
    # maxLag: maximum lags order for AR model
    # Oselec: Order selection flag; if True : first the order of lag is computed based on maxLag (default: True)
    #                               if False: AR model is computed based on the given order (maxLag)
    # Oselec_method: Order selection method; if Oselec is True : lag order will be computed based in this criteria (default: 'bic')
    #                                        if Oselec is False: don't bother to specify
    # show_order_selection_results: if True: before computing the model, it will show the lag order selection result based on different criterias (default: False)

    # pwGC: matrix of pairwise Granger causality; pwGC.shape = variable x variable
    # pwGC[i,j] shows the SSR-based f-test value for a the causal effect of variable_j on variable_i
    
    
    import numpy as np
    import statsmodels.tsa.api as sm
    n,m = x.shape
    if n<m: print('data in bad shape!'); x = x.T; n,m = x.shape
    p = maxLag

    pwGC = np.zeros([m,m])
    for i in range(m):
        for j in range(i):

            model = sm.VAR(x[:,[i,j]])

            if show_order_selection_results: print(f"selected order = {model.select_order(p).selected_orders}")
            
            if Oselec==True : results = model.fit(maxlags=p, ic=Oselec_method)
            if Oselec==False: results = model.fit(maxlags=p                  )
            
            pwGC[i,j] = results.test_causality(caused=0, causing=1, kind='f', signif=0.05).test_statistic
            pwGC[j,i] = results.test_causality(caused=1, causing=0, kind='f', signif=0.05).test_statistic
    

    return pwGC



# ======= Geweke's Conditional Granger Causality =======
def GGC_func(x, maxLag=10, Oselec=True, Oselec_method='bic', show_order_selection_results=False):
    # x: signals; x.shape = time x variables
    # maxLag: maximum lags order for MVAR model
    # Oselec: Order selection flag; if True : first the order of lag is computed based on maxLag (default: True)
    #                               if False: MVAR model is computed based on the given order (maxLag)
    # Oselec_method: Order selection method; if Oselec is True : lag order will be computed based in this criteria (default: 'bic')
    #                                        if Oselec is False: don't bother to specify
    # show_order_selection_results: if True: before computing the model, it will show the lag order selection result based on different criterias (default: False)

    # GGC: matrix of conditional Granger causality; GGC.shape = variable x variable
    # GGC[i,j] shows the SSR-based f-test value for a the causal effect of variable_j on variable_i
    
    
    import numpy as np
    import statsmodels.tsa.api as sm
    n,m = x.shape
    if n<m: print('data in bad shape!'); x = x.T
    p = maxLag

    model = sm.VAR(x)

    if show_order_selection_results: print(f"selected order = {model.select_order(p).selected_orders}")
    
    if Oselec==True:  results = model.fit(maxlags=p, ic=Oselec_method, trend='c')
    if Oselec==False: results = model.fit(maxlags=p, trend='c')

    GGC = np.zeros([m,m])
    for i in range(m):
        for j in range(i):
            GGC[i,j] = results.test_causality(caused=i, causing=j, kind='f', signif=0.05).test_statistic
            GGC[j,i] = results.test_causality(caused=j, causing=i, kind='f', signif=0.05).test_statistic


    return GGC



# ======= (Transfer function-based) Coherence Matrix =======
#   COH quantifies the linear coupling strength between two processes in the frequency domain
def MVCoh_func(x, fs=100, n_f=100, maxLag=10, Oselec=True, Oselec_method='bic', show_order_selection_results=False, normalized_freq_band_for_avg=[0, 1]):
    # x: signals; x.shape = time x variables
    # fs: sampling frequency
    # df: frequency resolution: df=1/(n*dt)
    # maxLag: maximum lags order for MVAR model
    # Oselec: Order selection flag; if True : first the order of lag is computed based on maxLag (default: True)
    #                               if False: MVAR model is computed based on the given order (maxLag)
    # Oselec_method: Order selection method; if Oselec is True : lag order will be computed based in this criteria (default: 'bic')
    #                                        if Oselec is False: don't bother to specify
    # show_order_selection_results: if True: before computing the model, it will show the lag order selection result based on different criterias (default: False)
    # normalized_freq_band_for_avg: frequency band normalized by nyquist frequency = fs/2 (maximum frequency of the signals)
    #                               which is used for averaging the COH matrices (dafault: [0,1] over all frequencies)

    # COH: Transfer Function-based Coherence matrix computed using parametric (MVAR) model and averaged over all frequencies
    

    import numpy as np
    import statsmodels.tsa.api as sm
    n, m = x.shape
    if n<m: print('data in bad shape!'); x = x.T
    p = maxLag

    freqs = np.linspace(0, fs/2, n_f+1)
    f1 = normalized_freq_band_for_avg[0]*freqs[-1]
    f2 = normalized_freq_band_for_avg[1]*freqs[-1]
    avg_weights = (freqs>=f1) * (freqs<=f2) * 1

    model = sm.VAR(x)

    if show_order_selection_results: print(f"selected order = {model.select_order(p).selected_orders}")
    
    if Oselec==True:  results = model.fit(maxlags=p, ic=Oselec_method, trend='c')
    if Oselec==False: results = model.fit(maxlags=p, trend='c')

    A = results.coefs
    p, _, _ = A.shape
    
    Sigma = results.resid_acov(nlags=0)[0]
    Sigma = np.diag(np.diag(Sigma))

    
    A_f = np.zeros([n_f+1, m, m], dtype=complex)
    H_f = np.zeros([n_f+1, m, m], dtype=complex)
    S_f = np.zeros([n_f+1, m, m], dtype=complex)
    COH = np.zeros([n_f+1, m, m], dtype=complex)

    for i_f, f in enumerate(freqs):
        
        for lag in range(p):
            A_f[i_f] += A[lag] * np.exp( -1j * 2*np.pi*f * (lag+1) )
        A_f[i_f] = np.eye(m) - A_f[i_f]

        H_f[i_f] = np.linalg.inv(A_f[i_f])
        
        S_f[i_f] = H_f[i_f] @ Sigma @ H_f[i_f].conj().T

        for i in range(m):
            for j in range(m):
                COH[i_f, i, j] = S_f[i_f, i, j] / ( np.sqrt(S_f[i_f, j,j]) * np.sqrt(S_f[i_f, i,i]) )

        
    COH = np.average(abs(COH), axis=0, weights=avg_weights)


    return COH



# ======= Partial Coherence Matrix =======
#   PCOH differentiates direct from indirect connections
def PCoh_func(x, fs=100, n_f=100, maxLag=10, Oselec=True, Oselec_method='bic', show_order_selection_results=False, normalized_freq_band_for_avg=[0, 1]):
    # x: signals; x.shape = time x variables
    # fs: sampling frequency
    # df: frequency resolution: df=1/(n*dt)
    # maxLag: maximum lags order for MVAR model
    # Oselec: Order selection flag; if True : first the order of lag is computed based on maxLag (default: True)
    #                               if False: MVAR model is computed based on the given order (maxLag)
    # Oselec_method: Order selection method; if Oselec is True : lag order will be computed based in this criteria (default: 'bic')
    #                                        if Oselec is False: don't bother to specify
    # show_order_selection_results: if True: before computing the model, it will show the lag order selection result based on different criterias (default: False)
    # normalized_freq_band_for_avg: frequency band normalized by nyquist frequency = fs/2 (maximum frequency of the signals)
    #                               which is used for averaging the COH matrices (dafault: [0,1] over all frequencies)

    # PCOH: Partial Coherence averaged over all frequencies
    

    import numpy as np
    import statsmodels.tsa.api as sm
    n, m = x.shape
    if n<m: print('data in bad shape!'); x = x.T
    p = maxLag

    freqs = np.linspace(0, fs/2, n_f+1)
    f1 = normalized_freq_band_for_avg[0]*freqs[-1]
    f2 = normalized_freq_band_for_avg[1]*freqs[-1]
    avg_weights = (freqs>=f1) * (freqs<=f2) * 1
    
    model = sm.VAR(x)

    if show_order_selection_results: print(f"selected order = {model.select_order(p).selected_orders}")
    
    if Oselec==True:  results = model.fit(maxlags=p, ic=Oselec_method, trend='c')
    if Oselec==False: results = model.fit(maxlags=p, trend='c')

    A = results.coefs
    p, _, _ = A.shape
    
    Sigma = results.resid_acov(nlags=0)[0]
    Sigma = np.diag(np.diag(Sigma))
    
    A_f  = np.zeros([n_f+1, m, m], dtype=complex)
    H_f  = np.zeros([n_f+1, m, m], dtype=complex)
    S_f  = np.zeros([n_f+1, m, m], dtype=complex)
    P_f  = np.zeros([n_f+1, m, m], dtype=complex)
    PCOH = np.zeros([n_f+1, m, m], dtype=complex)

    for i_f, f in enumerate(freqs):
        
        for lag in range(p):
            A_f[i_f] += A[lag] * np.exp( -1j * 2*np.pi*f * (lag+1) )
        A_f[i_f] = np.eye(m) - A_f[i_f]

        H_f[i_f] = np.linalg.inv(A_f[i_f])
        
        S_f[i_f] = H_f[i_f] @ Sigma @ H_f[i_f].conj().T
        
        P_f[i_f] = np.linalg.inv(S_f[i_f])
        
        for i in range(m):
            for j in range(m):
                PCOH[i_f, i, j] = P_f[i_f, i, j] / ( np.sqrt(P_f[i_f, i,i]) * np.sqrt(P_f[i_f, j,j]) )


    PCOH = np.average(abs(PCOH), axis=0, weights=avg_weights)


    return PCOH



# ======= Directed Coherence Matrix =======
#   DCOH quantifies the direct and indirect causal links between two time-series
def DCoh_func(x, fs=100, n_f=100, maxLag=10, Oselec=True, Oselec_method='bic', show_order_selection_results=False, normalized_freq_band_for_avg=[0, 1]):
    # x: signals; x.shape = time x variables
    # fs: sampling frequency
    # df: frequency resolution: df=1/(n*dt)
    # maxLag: maximum lags order for MVAR model
    # Oselec: Order selection flag; if True : first the order of lag is computed based on maxLag (default: True)
    #                               if False: MVAR model is computed based on the given order (maxLag)
    # Oselec_method: Order selection method; if Oselec is True : lag order will be computed based in this criteria (default: 'bic')
    #                                        if Oselec is False: don't bother to specify
    # show_order_selection_results: if True: before computing the model, it will show the lag order selection result based on different criterias (default: False)
    # normalized_freq_band_for_avg: frequency band normalized by nyquist frequency = fs/2 (maximum frequency of the signals)
    #                               which is used for averaging the COH matrices (dafault: [0,1] over all frequencies)

    # DCOH: Directed Coherence averaged over all frequencies
    

    import numpy as np
    import statsmodels.tsa.api as sm
    n, m = x.shape
    if n<m: print('data in bad shape!'); x = x.T
    p = maxLag

    freqs = np.linspace(0, fs/2, n_f+1)
    f1 = normalized_freq_band_for_avg[0]*freqs[-1]
    f2 = normalized_freq_band_for_avg[1]*freqs[-1]
    avg_weights = (freqs>=f1) * (freqs<=f2) * 1
    
    model = sm.VAR(x)

    if show_order_selection_results: print(f"selected order = {model.select_order(p).selected_orders}")
    
    if Oselec==True:  results = model.fit(maxlags=p, ic=Oselec_method, trend='c')
    if Oselec==False: results = model.fit(maxlags=p, trend='c')

    A = results.coefs
    p, _, _ = A.shape
    
    Sigma = results.resid_acov(nlags=0)[0]
    Sigma = np.diag(np.diag(Sigma))

    
    A_f  = np.zeros([n_f+1, m, m], dtype=complex)
    H_f  = np.zeros([n_f+1, m, m], dtype=complex)
    DCOH = np.zeros([n_f+1, m, m], dtype=complex)

    for i_f, f in enumerate(freqs):
        
        for lag in range(p):
            A_f[i_f] += A[lag] * np.exp( -1j * 2*np.pi*f * (lag+1) )
        A_f[i_f] = np.eye(m) - A_f[i_f]
        
        H_f[i_f] = np.linalg.inv(A_f[i_f])
        
        for i in range(m):
            for j in range(m):
                S_f = np.sum( np.diag(Sigma)**2 * abs(H_f[i_f, i,:])**2 )
                DCOH[i_f, i, j] = np.diag(Sigma)[j]*H_f[i_f, i, j] / np.sqrt( S_f )


    DCOH = np.average(abs(DCOH), axis=0, weights=avg_weights)


    return DCOH



def PDCoh_func(x, fs=100, n_f=100, maxLag=10, Oselec=True, Oselec_method='bic', show_order_selection_results=False, normalized_freq_band_for_avg=[0, 1]):
    # x: signals; x.shape = time x variables
    # fs: sampling frequency
    # df: frequency resolution: df=1/(n*dt)
    # maxLag: maximum lags order for MVAR model
    # Oselec: Order selection flag; if True : first the order of lag is computed based on maxLag (default: True)
    #                               if False: MVAR model is computed based on the given order (maxLag)
    # Oselec_method: Order selection method; if Oselec is True : lag order will be computed based in this criteria (default: 'bic')
    #                                        if Oselec is False: don't bother to specify
    # show_order_selection_results: if True: before computing the model, it will show the lag order selection result based on different criterias (default: False)
    # normalized_freq_band_for_avg: frequency band normalized by nyquist frequency = fs/2 (maximum frequency of the signals)
    #                               which is used for averaging the COH matrices (dafault: [0,1] over all frequencies)

    # PDCOH: Partial Directed Coherence averaged over all frequencies
    

    import numpy as np
    import statsmodels.tsa.api as sm
    n, m = x.shape
    if n<m: print('data in bad shape!'); x = x.T
    p = maxLag

    freqs = np.linspace(0, fs/2, n_f+1)
    f1 = normalized_freq_band_for_avg[0]*freqs[-1]
    f2 = normalized_freq_band_for_avg[1]*freqs[-1]
    avg_weights = (freqs>=f1) * (freqs<=f2) * 1
    
    model = sm.VAR(x)

    if show_order_selection_results: print(f"selected order = {model.select_order(p).selected_orders}")
    
    if Oselec==True:  results = model.fit(maxlags=p, ic=Oselec_method, trend='c')
    if Oselec==False: results = model.fit(maxlags=p, trend='c')

    A = results.coefs
    p, _, _ = A.shape

    Sigma = results.resid_acov(nlags=0)[0]
    Sigma = np.diag(np.diag(Sigma))

    A_f   = np.zeros([n_f+1, m, m], dtype=complex)
    PDCOH = np.zeros([n_f+1, m, m], dtype=complex)

    for i_f, f in enumerate(freqs):
        
        for lag in range(p):
            A_f[i_f] += A[lag] * np.exp( -1j * 2*np.pi*f * (lag+1) )
        A_f[i_f] = np.eye(m) - A_f[i_f]
        
        for i in range(m):
            for j in range(m):
                S_f = np.sum( 1/np.diag(Sigma)**2 * abs(A_f[i_f, :,j])**2 )
                PDCOH[i_f, i, j] = (1/np.diag(Sigma)[i])*A_f[i_f, i, j] / np.sqrt( S_f )


    PDCOH = np.average(abs(PDCOH), axis=0, weights=avg_weights)


    return PDCOH



# ======= Directed Transfer Function =======
def DTF_func(x, fs=100, df=0.2, maxLag=10, Oselec=True, Oselec_method='bic', show_order_selection_results=False, normalized_freq_band_for_avg=[0, 1]):
    # x: signals; x.shape = time x variables
    # fs: sampling frequency
    # df: frequency resolution: df=1/(n*dt)
    # maxLag: maximum lags order for MVAR model
    # Oselec: Order selection flag; if True : first the order of lag is computed based on maxLag (default: True)
    #                               if False: MVAR model is computed based on the given order (maxLag)
    # Oselec_method: Order selection method; if Oselec is True : lag order will be computed based in this criteria (default: 'bic')
    #                                        if Oselec is False: don't bother to specify
    # show_order_selection_results: if True: before computing the model, it will show the lag order selection result based on different criterias (default: False)
    # normalized_freq_band_for_avg: frequency band normalized by nyquist frequency = fs/2 (maximum frequency of the signals)
    #                               which is used for averaging the COH matrices (dafault: [0,1] over all frequencies)

    # DTF: Directed Transfer Function matrix averaged over all frequencies; DTF.shape = variable x variable
    # DTF[i,j] shows the SSR-based f-test value for a the causal effect of variable_j on variable_i


    import numpy as np
    import statsmodels.tsa.api as sm
    n, m = x.shape
    if n<m: print('data in bad shape!'); x = x.T
    p = maxLag

    freqs = np.arange(0, fs/2+df, df)
    n_f = len(freqs)
    f1 = normalized_freq_band_for_avg[0]*fs/2
    f2 = normalized_freq_band_for_avg[1]*fs/2
    avg_weights = (freqs>=f1) * (freqs<=f2) * 1
    
    model = sm.VAR(x)

    if show_order_selection_results: print(f"selected order = {model.select_order(p).selected_orders}")
    
    if Oselec==True:  results = model.fit(maxlags=p, ic=Oselec_method, trend='c')
    if Oselec==False: results = model.fit(maxlags=p, trend='c')

    A = results.coefs
    p, _, _ = A.shape
    
    A_f   = np.zeros([n_f, m, m], dtype=complex)
    H_f   = np.zeros([n_f, m, m], dtype=complex)
    DTF = np.zeros([n_f, m, m], dtype=complex)

    for i_f, f in enumerate(freqs):
        
        for lag in range(p):
            A_f[i_f] += A[lag] * np.exp( -1j * 2*np.pi*f * (lag+1) )
        H_f[i_f] = np.eye(m) - A_f[i_f]

        H_f[i_f] = np.linalg.inv(A_f[i_f])
                
        for i in range(m):
            for j in range(m):
                DTF[i_f, i, j] = abs(H_f[i_f, i, j]) / np.sqrt( np.sum( abs(H_f[i_f, i, :])**2 ) )
        

    DTF = np.average(abs(DTF), axis=0, weights=avg_weights)


    return DTF


# ======= PsychoPhysiological interpretation ======= 
# For BOLD-fMRI data, u_conv is the task signal convolved with hrf and mean subtracted 
def PPI_func(x, u_conv):
    import numpy as np
    import scipy as sp
    import statsmodels.api as sm

    n, m = x.shape

    # ======= GLM_PPI =======
    b_map = np.zeros([m,m])
    # t_map = np.zeros([m,m])

    for i in range(m):
        x_seed   = x[:,i]
        x_psych  = u_conv
        x_ppi    = x_seed*x_psych
        
        X = np.column_stack((x_psych, x_seed, x_ppi))
        X = sm.add_constant(X)

        for j in range(m):
            if j==i: continue
            y = x[:,j]
            
            model = sm.OLS(y, X)
            sm.OLS.exog_names = ['const', 'task', 'seed', 'ppi']
            result = model.fit()
            b_map[j,i] = result.summary2().tables[1]['Coef.']['x3']
            # t_map[j,i] = result.summary2().tables[1]['t']['x3']


    PPI_results = b_map
    # PPI_results = t_map


    return PPI_results


# ======= Dynamic Causal Modelling =======
def DCM_func(dx, x, u, u_mod):
    import numpy as np
    n, m = x.shape
    u = u.reshape([n,-1])
    u_mod = u_mod.reshape([n,-1])
    
    result = {'x': np.zeros([m, m]), 'x*u': np.zeros([m, m]), 'u': np.zeros([m, m])}
    
    Y = dx
    Y = Y - np.mean(Y, axis=0)
    
    X_basket = np.hstack( (x, x*u_mod, u) )
    X_basket = X_basket - np.mean(X_basket, axis=0)
    X = np.hstack( (X_basket, np.ones([n, 1])) )
    
    basket = np.linalg.lstsq(X, Y, rcond=None)[0]
    
    result['x']   = basket[:m,:]
    result['x*u'] = basket[m:2*m,:]
    result['u']   = basket[2*m:2*m+1,:]
    
    return result
