from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas
import keras
import gc

class call_roc_hist(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.val_aucs = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.validation_data[0], self.validation_data[1]])
        scoroc = roc_auc_score(self.validation_data[2][:,1], y_pred[:,1])
        self.val_aucs.append(scoroc)
        print('\nEpoch:',epoch,'\troc_auc:',scoroc,'\n')
        return


def scaled_features(dataFrame, train_run=True, scalers={}):
    col_names = dataFrame.columns.values
    x = 6 if train_run else 0  # not using the "answers" in preprocessing
    i=0
    n_samp = dataFrame.shape[0]
    
    mass_square = np.zeros((n_samp,200))
    eta = np.zeros((n_samp,200))
    pT_square = np.zeros((n_samp,200))
    E_val = np.zeros((n_samp,200))
    px_val = np.zeros((n_samp,200))
    py_val = np.zeros((n_samp,200))
    pz_val = np.zeros((n_samp,200))
    particleN = 0
    while i<len(col_names)-x:
        if particleN >=200:
            break
        for j in range(0,4):
            if j == 0:
                E = dataFrame[col_names[i+j]].values
                E_val[:,particleN] = E 
            elif j == 1:
                px= dataFrame[col_names[i+j]].values
                px_val[:,particleN] = px
            elif j == 2:
                py = dataFrame[col_names[i+j]].values
                py_val[:,particleN] = py
            elif j == 3:
                pz = dataFrame[col_names[i+j]].values
                pz_val[:,particleN] = pz
        mass_square[:,particleN] = np.abs(E**2-px**2-py**2-pz**2)
        #eta
        with np.errstate(divide='ignore',invalid='ignore'):
            eta[:,particleN] = np.arctanh(pz/np.sqrt(px**2+py**2+pz**2));momentum0 = np.isnan(eta);eta[momentum0] = 0
            ## don't be scared by dividing by zero, NaN is our friend
        #p_T
        pT_square[:,particleN] = np.sqrt(px**2+py**2)    
        i+=4;particleN+=1
    
    result3D = np.reshape(np.hstack((mass_square,eta,pT_square,E_val,px_val,py_val,pz_val)),(n_samp,200,7))
    #scale features 
    if train_run:
        scalers = {}
        for i in range(result3D.shape[1]):
            scalers[i] = preprocessing.StandardScaler()
            result3D[:, i, :] = scalers[i].fit_transform(result3D[:, i, :])
    else:
        for i in range(result3D.shape[1]):
            result3D[:, i, :] = scalers[i].transform(result3D[:, i, :])
    return result3D, scalers



def load_data(file_name, images=False, scalers={}, training=True, start=10, stop=1010):
    store = pandas.HDFStore(file_name)
    df = store.select("table",start=start, stop=stop)
    pixels = ["img_{0}".format(i) for i in range(1600)]
    if 'is_signal_new' in df.columns:
        y = df['is_signal_new'].values
    else: y = False
    gc.collect()
    if images: return np.expand_dims(np.expand_dims(df[pixels], axis=-1).reshape(-1,40,40), axis=-1), y
    else: return scaled_features(df, training, scalers), y
    

    