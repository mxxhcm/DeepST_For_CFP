# -*- coding: utf-8 -*-
""" 
Usage:
    THEANO_FLAGS="device=gpu0" python exptTaxiRM3.py [number_of_residual_units]
"""
from __future__ import print_function
import os
import sys
import cPickle as pickle
import time
import numpy as np
import h5py

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import TaxiBJ

from copy import deepcopy

np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = Config().DATAPATH  # data path, you may set your own data path with the global envirmental variable DATAPATH
CACHEDATA = True  # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE')  # cache path
nb_epoch = 500  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 32  # batch size
T = 48  # number of time intervals in one day
lr = 0.0002  # learning rate
len_closeness = 5  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence
if len(sys.argv) == 1:
    print(__doc__)
    sys.exit(-1)
    # nb_residual_unit = 2  # number of residual units
else:
    nb_residual_unit = int(sys.argv[1])  # number of residual units

nb_flow = 2  # there are two types of flows: inflow and outflow
# divide data into two subsets: Train & Test, of which the test set is the
# last 4 weeks
days_test = 7 * 4
len_test = T * days_test
map_height, map_width = 32, 32  # grid size
path_result = 'RET'
path_model = 'MODEL'
path_model2 = 'MODEL2'
path_model3 = 'MODEL3'


if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if os.path.isdir(path_model2) is False:
    os.mkdir(path_model2)
if os.path.isdir(path_model3) is False:
    os.mkdir(path_model3)

if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)


def build_model(external_dim):
    c_conf = (len_closeness - 2, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def build_model_2(external_dim):
    c_conf = (len_closeness - 1, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def build_model_3(external_dim):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def read_cache(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in xrange(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()


def main():

    # load data
    print("loading data...")
    ts = time.time()
    fname = os.path.join(DATAPATH, 'CACHE', 'TaxiBJ_C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiBJ.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
            preprocess_name='preprocessing.pkl', meta_data=True, meteorol_data=True, holiday_data=True)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    ts = time.time()
    model = build_model(external_dim)
    model_2 = build_model_2(external_dim)
    model_3 = build_model_3(external_dim)

    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    fname_param2 = os.path.join('MODEL2', '{}.best.h5'.format(hyperparams_name))
    fname_param3 = os.path.join('MODEL3', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=10, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    model_checkpoint2 = ModelCheckpoint(
        fname_param2, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    model_checkpoint3 = ModelCheckpoint(
        fname_param3, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print("\nelapsed time (compiling model): %.3f seconds\n" %
          (time.time() - ts))

    # X_train[0] = X_train[0][0:100]
    # X_train[1] = X_train[1][0:100]
    # X_train[2] = X_train[2][0:100]
    # X_train[3] = X_train[3][0:100]
    # Y_train = Y_train[0:100]

    # model_ 1
    X_train_1 = deepcopy(X_train)
    X_train_1[0] = X_train_1[0][:, 0:6, :, :]
    Y_train_1 = deepcopy(X_train[0][:, 6:8, :, :])
    print('=' * 10)
    print("training model...")
    ts = time.time()
    history = model.fit(X_train_1, Y_train_1,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    model.load_weights(fname_param)
    score = model.evaluate(X_train_1, Y_train_1, batch_size=Y_train_1.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    X_test_1 = deepcopy(X_test)
    X_test_1[0] = X_test_1[0][:, 0:6, :, :]
    Y_test_1 = deepcopy(X_test[0][:, 6:8, :, :])
    score = model.evaluate(X_test_1, Y_test_1, batch_size=Y_test_1.shape[0],
                           verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

    Y_predicted_1 = model.predict(X_train_1)
    Y_test_predicted_1 = model.predict(X_test_1)



    # # model 2
    # X_train_2 = deepcopy(X_train)
    # X_train_2[0] = np.hstack((X_train_2[0][:, 0:6, :, :], Y_predicted_1))
    # Y_train_2 = deepcopy(X_train[0][:, 8:10, :, :])
    #
    # print('=' * 10)
    # print("training model2...")
    # ts = time.time()
    # history2 = model_2.fit(X_train_2, Y_train_2,
    #                     nb_epoch=nb_epoch,
    #                     batch_size=batch_size,
    #                     validation_split=0.1,
    #                     callbacks=[early_stopping, model_checkpoint2],
    #                     verbose=1)
    # model_2.save_weights(os.path.join(
    #     'MODEL2', '{}.h5'.format(hyperparams_name)), overwrite=True)
    # pickle.dump((history2.history), open(os.path.join(
    #     path_result, '{}.history2.pkl'.format(hyperparams_name)), 'wb'))
    # print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))
    #
    # print('=' * 10)
    # print('evaluating using the model that has the best loss on the valid set')
    # ts = time.time()
    # model_2.load_weights(fname_param2)
    # score = model_2.evaluate(X_train_2, Y_train_2, batch_size=Y_train_2.shape[
    #                        0] // 48, verbose=0)
    # print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    #
    # X_test_2 = deepcopy(X_test)
    # X_test_2[0] = np.hstack((X_test_2[0][:, 0:6, :, :], Y_test_predicted_1))
    # Y_test_2 = deepcopy(X_test[0][:, 8:10, :, :])
    #
    # score = model_2.evaluate(
    #     X_test_2, Y_test_2, batch_size=Y_test_2.shape[0], verbose=0)
    # print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    # print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    #
    # Y_predicted_2 = model_2.predict(X_train_2)
    # Y_test_predicted_2 = model_2.predict(X_test_2)
    #
    #
    #
    # # model 3
    # X_train_3 = deepcopy(X_train)
    # X_train_3[0] = np.hstack((X_train_2[0], Y_predicted_2))
    # Y_train_3 = deepcopy(Y_train)
    #
    # print('=' * 10)
    # print("training model3...")
    # ts = time.time()
    # history3 = model_3.fit(X_train_3, Y_train_3,
    #                     nb_epoch=nb_epoch,
    #                     batch_size=batch_size,
    #                     validation_split=0.1,
    #                     callbacks=[early_stopping, model_checkpoint3],
    #                     verbose=1)
    # model_3.save_weights(os.path.join(
    #     'MODEL3', '{}.h5'.format(hyperparams_name)), overwrite=True)
    # pickle.dump((history3.history), open(os.path.join(
    #     path_result, '{}.history3.pkl'.format(hyperparams_name)), 'wb'))
    # print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))
    #
    # print('=' * 10)
    # print('evaluating using the model that has the best loss on the valid set')
    # ts = time.time()
    # model_3.load_weights(fname_param3)
    # score = model_3.evaluate(X_train_3, Y_train_3, batch_size=Y_train_3.shape[
    #                        0] // 48, verbose=0)
    # print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    #
    # X_test_3 = deepcopy(X_test)
    # X_test_3[0] = np.hstack((X_test_2[0], Y_test_predicted_2))
    # Y_test_3 = deepcopy(Y_test)
    # score = model_3.evaluate(
    #     X_test_3, Y_test_3, batch_size=Y_test_3.shape[0], verbose=0)
    # print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    # print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))
    #
    #

    #
    # print('=' * 10)
    # print("training model (cont)...")
    # ts = time.time()
    # fname_param = os.path.join(
    #     'MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    # model_checkpoint = ModelCheckpoint(
    #     fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    # history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
    #                     model_checkpoint])
    # pickle.dump((history.history), open(os.path.join(
    #     path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    # model.save_weights(os.path.join(
    #     'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    # print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))
    #
    # print('=' * 10)
    # print('evaluating using the final model')
    # score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
    #                        0] // 48, verbose=0)
    # print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    # ts = time.time()
    # score = model.evaluate(
    #     X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    # print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    # print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))


if __name__ == '__main__':
    main()
