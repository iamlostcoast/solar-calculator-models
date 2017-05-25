import keras
import keras.backend as K
#K._BACKEND == 'tensorflow'

import pandas as pd
import numpy as np
import sqlite3

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

import patsy
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Reading in the data from csvs
df_open = pd.read_csv("./data/open_pv_clean.csv")
df_zips = pd.read_csv("./data/nrel_solar_data.csv")

# merging two datasets on zipcode
df = df_open.merge(df_zips, right_on = 'Zipcode', left_on = 'zipcode')

# dropping unnecessary rows
del(df['city'])
del(df['manufacturer'])
del(df['inv_man_clean'])
del(df['model1_clean'])
del(df['sales_tax_cost'])
del(df['annual_insolation'])
del(df['lbnl_tts'])
del(df['county'])
del(df['date_installed'])
del(df['cost_per_watt'])

# Weird extra columns
del(df['Unnamed: 0_x'])
del(df['Unnamed: 0_y'])

# imputing some data.
median_tilt = df.tilt1.median()
median_azimuth = df.azimuth1.median()

df.tilt1.fillna(median_tilt, inplace=True)
df.azimuth1.fillna(median_azimuth, inplace=True)

# getting rid of some outliers
df = df.loc[df['reported_annual_energy_prod'] - df['annual_PV_prod'] < 2*10**6]

formula = """\
reported_annual_energy_prod ~ size_kw + azimuth1 + tilt1 + tech_1 + DHI + DNI + Wind_Speed + tracking_type"""

print 'splitting into X and y design matrices'
y, X = patsy.dmatrices(formula, df, return_type='dataframe')

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# not using this, just doing a built in evaluation split.
print 'splitting into train and test'

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=.75, random_state=25)

# this is a custom r2 function to use in the model for evaluating performance
def custom_r2(y_true, y_pred):
    baseline = K.sum((y_true - K.mean(y_true))**2)
    model_fit = K.sum((y_true - y_pred)**2)
    return 1. - model_fit/baseline


# Building Neural Net Model:
model = Sequential()

model.add(Dense(1024, input_shape=(21,), kernel_initializer='normal'))
model.add(Activation('elu'))
model.add(Dropout(.2))

model.add(Dense(1024, kernel_initializer='normal'))
model.add(Activation('elu'))
model.add(Dropout(.5))

model.add(Dense(1, kernel_initializer='normal'))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=[custom_r2])

def early_stopping_cont_nn(model, X, y, val_data, batch_size=1, verbose=1, epochs_per_iter=1, max_iterations=10,
                           patience=3, min_delta=.1):
    """DOCSTRING:
    This function takes a keras neural net model predicting a continuous variable, with a custom r2 function as
    the evaluation metric.  It will check the change to the r2 value and do an early stop if the r2 does not change by
    the amount specifed in min delta.

    - model: keras model to fit.
    - X: numpy array of X training data.
    - y: numpy array of y training data.
    - val_data: (tuple) of numpy arrays of X and y validation data.
    - batch_size: batch size for keras model fitting.
    - verbose: 0/1 for whether keras model fitting is verbose.
    - epochs_per_iteration: (int) How many epochs to fit before checking to see if the evaluation metric has change\
                enough to continue
    - max_iterations: (int) How many iterations through the specified epochs before automatically stopping fitting.
    - patience: (int) How many iterations can have less than the specified delta to the evaluation metric before stopping.
    - min_delta: (int/float) How much evaluation metric must change by at minimum not to raise patience level.
    """

    # if our model doesn't improve by a certain amount for 'patience' times, we'll break the fitting.
    current_patience = 0
    last_r2 = 0

    for i in range(max_iterations):
        model.fit(x=X, y=y, validation_data = val_data, batch_size=batch_size, verbose=verbose, epochs=epochs_per_iter)

        current_r2 = r2_score(y_test, model.predict(np.array(X_test)))
        print "current r2: ", current_r2
        print "-----------------------"
        r2_diff = current_r2 - last_r2

        print "R2 improvement: ", r2_diff
        print "-----------------------"
        if r2_diff < min_delta:
            print "Improvement Does Not Meet Minimum, Adding One to Patience Score"
            print "-----------------------"
            current_patience += 1
        if current_patience >= patience:
            print "Stopping training, patience threshold met"
            break
        last_r2 = current_r2

print 'Training model!'

# Calling the function we just wrote.  May need to tune batch size, min_delta, as well as the model.
early_stopping_cont_nn(model, X=np.array(X_train), y=np.array(y_train).ravel(),
                       val_data=(np.array(X_test), np.array(y_test).ravel()), epochs_per_iter=1,
                       batch_size=500, verbose=1, max_iterations=1000, min_delta=.0005, patience=2)

print r2_score(y_test, model.predict(np.array(X_test)))

model.save('./neural_net_models/nn_2_wide.h5')
