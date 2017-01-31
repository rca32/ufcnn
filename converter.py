import numpy as np
import pandas as pd
def window_dataset(data, n_prev=1):
    """
    data should be pd.DataFrame()
    """
    dlistX, dlistY = [], []
    for i in range(len(data) - n_prev):
        dlistX.append(data.iloc[i:i + n_prev].as_matrix())
        dlistY.append(data.iloc[i + n_prev].as_matrix())
    darrX = np.array(dlistX)
    darrY = np.array(dlistY)
    return darrX, darrY

def masked_dataset(data, n_prev=3, n_masked=2, predict_ahead=1):
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data) - n_prev - n_masked - predict_ahead):
        x = data.iloc[i:i + n_prev].as_matrix()
        x_mask = np.zeros((n_masked, x.shape[1]))
        docX.append(np.concatenate((x, x_mask)))

        y = data.iloc[i + predict_ahead: i + n_prev + n_masked + predict_ahead].as_matrix()
        docY.append(y)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def prediction_dataset(data, n_samples=100, n_ahead=3):
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data) - n_samples - n_ahead):
        x = data.iloc[i:i + n_samples].as_matrix()
        docX.append(x)
        y = data.iloc[i + n_ahead: i + n_samples + n_ahead].as_matrix()
        docY.append(y)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def seq2seq_dataset(data, n_samples=50, n_ahead=50):
    docX, docY = [], []
    for i in range(len(data) - n_samples - n_ahead):
        x = data.iloc[i:i + n_samples].as_matrix()
        docX.append(x)
        y = data.iloc[i + n_samples:i + n_samples + n_ahead].as_matrix()
        docY.append(y)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def test_train_split(df, test_size=.2, splitting_method='prediction', **kwargs):

    ntrn = int(len(df) * (1 - test_size))
    X_train, y_train = seq2seq_dataset(df.iloc[0:ntrn], **kwargs)
    X_test, y_test = seq2seq_dataset(df.iloc[ntrn:], **kwargs)
    
    return (X_train, y_train),(X_test, y_test)

df = pd.read_pickle("./org.pkl")
(X_train, y_train),(X_test, y_test) = test_train_split(df)
