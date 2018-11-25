import os
import pickle as cPickle
from collections import defaultdict, OrderedDict
import keras

import numpy as np
import keras.backend as K

#Limited Graphic Card Usage
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#Different Entropy Estimator
import kde
import simplebinmi

import utils
import loggingreporter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style('darkgrid')

import utils

def save_activation(activation, L1):

    #Parameters Of Experiments
    cfg = {}
    cfg['SGD_BATCHSIZE']    = 256
    cfg['SGD_LEARNINGRATE'] = 0.0004
    cfg['NUM_EPOCHS']       = 5000
    cfg['FULL_MI']          = True

    cfg['ACTIVATION']       = 'tanh'

    cfg['LAYER_DIMS']       = [10, 7, 5, 4, 3]
    ARCH_NAME = '-'.join(map(str, cfg['LAYER_DIMS']))

    trn, tst = utils.get_IB_data('2017_12_21_16_51_3_275766')

    cfg['SAVE_DIR'] = 'rawdata/' + cfg['ACTIVATION'] + '_' + ARCH_NAME


    input_layer = keras.layers.Input((trn.X.shape[1],))
    clayer = input_layer
    # clayer = keras.layers.Dense(cfg['LAYER_DIMS'][0],
    #                             activation=cfg['ACTIVATION'],
    #                             kernel_initializer=keras.initializers.truncated_normal(mean=0.0,
    #                                                                                    stddev=1 / np.sqrt(float(cfg['LAYER_DIMS'][0]))),
    #                             bias_initializer='zeros',
    #                             activity_regularizer=keras.regularizers.l1(.01)
    #                             )(clayer)
    for n in cfg['LAYER_DIMS']:
        clayer = keras.layers.Dense(n,
                activation=cfg['ACTIVATION'],
                kernel_initializer=keras.initializers.truncated_normal(mean=0.0, stddev=1/np.sqrt(float(n))),
                bias_initializer = 'zeros',
        )(clayer)

    output_layer = keras.layers.Dense(trn.nb_classes, activation='softmax')(clayer)

    model = keras.models.Model(inputs=input_layer, outputs = output_layer)
    optimizer = keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=cfg['SGD_LEARNINGRATE']))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def do_report(epoch):
        if epoch < 20:
            return True
        elif epoch < 100:
            return (epoch % 5 == 0)
        elif epoch < 2000:
            return (epoch % 20 == 0)
        else:
            return (epoch % 100 == 0)

    reporter = loggingreporter.LoggingReporter(cfg = cfg, trn = trn, tst = tst, do_save_func = do_report)

    r = model.fit(x=trn.X, y=trn.Y,
                verbose=2,
                batch_size=cfg['SGD_BATCHSIZE'],
                epochs=cfg['NUM_EPOCHS'],
                callbacks=[reporter,])

def computeMI(id):
    train, test = utils.get_IB_data('2017_12_21_16_51_3_275766')

    # For both train and test musr correspond with saving code
    FULL_MI = True

    # MI Measure
    infoplane_measure = 'bin'

    DO_SAVE = True
    DO_LOWER = (infoplane_measure == 'lower')
    DO_BINNED = (infoplane_measure == 'bin')

    MAX_EPOCHS = 5000
    NUM_LABELS = 2

    COLORBAR_MAX_EPOCHS = 5000

    # Directory For Loading saved Data
    ARCH = '10-7-5-4-3'
    DIR_TEMPLATE = '%%s_%s' % ARCH

    noise_variance = 1e-3
    binsize = 0.07
    Klayer_activity = K.placeholder(ndim=2)
    entropy_func_upper = K.function([Klayer_activity, ], [kde.entropy_estimator_kl(Klayer_activity, noise_variance), ])
    entropy_func_lower = K.function([Klayer_activity, ], [kde.entropy_estimator_bd(Klayer_activity, noise_variance), ])

    # Nats to bits conversion
    nats2bits = 1.0 / np.log(2)

    # Indexes of tests data for each of Output Classes
    saved_labelixs = {}

    y = test.y
    Y = test.Y
    if FULL_MI:
        full = utils.construct_full_dataset(train, test)
        y = full.y
        Y = full.Y

    for i in range(NUM_LABELS):
        saved_labelixs[i] = (y == i)

    labelprobs = np.mean(Y, axis=0)

    # Layers to plot, None for all
    PLOT_LAYERS = None

    # Store Results
    measures = OrderedDict()
    measures['tanh'] = {}
    measures['relu'] = {}

    for activation in measures.keys():
        cur_dir = 'rawdata/' + DIR_TEMPLATE % activation
        if not os.path.exists(cur_dir):
            print("Directory %s not found" % cur_dir)
            continue

        print("******* Loading %s ******" % cur_dir)
        for epochfile in sorted(os.listdir(cur_dir)):
            if not epochfile.startswith('epoch'):
                continue
            fname = cur_dir + '/' + epochfile
            with open(fname, 'rb') as f:
                d = cPickle.load(f)

            epoch = d['epoch']
            if epoch in measures[activation]:
                continue

            if epoch > MAX_EPOCHS:
                continue

            print("Measureing ", fname)

            num_layers = len(d['data']['activity_tst'])
            if PLOT_LAYERS is None:
                PLOT_LAYERS = []
                for lndx in range(num_layers):
                    PLOT_LAYERS.append(lndx)

            cepochdata = defaultdict(list)
            for lndx in range(num_layers):
                activity = d['data']['activity_tst'][lndx]

                h_upper = entropy_func_upper([activity, ])[0]
                if DO_LOWER:
                    h_lower = entropy_func_lower()

                hM_given_X = kde.kde_condentropy(activity, noise_variance)
                hM_given_Y_upper = 0
                for i in range(NUM_LABELS):
                    hcond_upper = entropy_func_upper([activity[saved_labelixs[i], :], ])[0]
                    hM_given_Y_upper += labelprobs[i] * hcond_upper

                if DO_LOWER:
                    hM_given_Y_lower = 0
                    for i in range(NUM_LABELS):
                        hcond_lower = entropy_func_lower([activity[saved_labelixs[i], :], ])[0]
                        hM_given_Y_lower += labelprobs[i] * hcond_lower

                cepochdata['MI_XM_upper'].append(nats2bits * (h_upper - hM_given_X))
                cepochdata['MI_YM_upper'].append(nats2bits * (h_upper - hM_given_Y_upper))
                cepochdata['H_M_upper'].append(nats2bits * h_upper)

                pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
                if DO_LOWER:
                    cepochdata['MI_XM_lower'].append(nats2bits * (h_lower - hM_given_X))
                    cepochdata['MI_YM_lower'].append(nats2bits * (h_lower - hM_given_Y_lower))
                    cepochdata['H_M_lower'].append(nats2bits * h_lower)
                    pstr += 'lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                    cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

                if DO_BINNED:
                    binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, binsize)
                    cepochdata['MI_XM_bin'].append(nats2bits * binxm)
                    cepochdata['MI_YM_bin'].append(nats2bits * binym)
                    pstr += 'bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                    cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])
                print('- Layer %d %s' % (lndx, pstr))
            measures[activation][epoch] = cepochdata
    with open("MI" + str(id), 'wb') as f:
        cPickle.dump(measures, f)

def avg_MI(ids):
    with open("MI"+str(ids[0]), 'rb') as f:
        measures = cPickle.load(f)
    activation = measures.keys()
    epochs = measures[list(activation)[0]]

    for act in activation:
        for epoch in epochs:
            measures[act][epoch]['MI_XM_bin'] = np.array(measures[act][epoch]['MI_XM_bin'])
            measures[act][epoch]['MI_YM_bin'] = np.array(measures[act][epoch]['MI_YM_bin'])

    for id in ids[1:]:
        with open("MI" + str(id), 'rb') as f:
            single_measure = cPickle.load(f)
        for act in activation:
            for epoch in epochs:
                measures[act][epoch]['MI_XM_bin'] += single_measure[act][epoch]['MI_XM_bin']
                measures[act][epoch]['MI_YM_bin'] += single_measure[act][epoch]['MI_YM_bin']

    for act in activation:
        for epoch in epochs:
            measures[act][epoch]['MI_XM_bin'] /= len(ids)
            measures[act][epoch]['MI_YM_bin'] /= len(ids)

    return measures

def plot_IP(measures):
    COLORBAR_MAX_EPOCHS = 5000
    infoplane_measure = 'bin'
    PLOT_LAYERS = [0, 1, 2, 3, 4]

    # Plot Information Plane
    max_epoch = max((max(vals.keys()) if len(vals) else 0) for vals in measures.values())
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
    sm._A = []

    fig = plt.figure(figsize=(10, 5))
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        if not len(epochs):
            continue
        plt.subplot(1, 2, actndx + 1)
        for epoch in epochs:
            c = sm.to_rgba(epoch)
            xmvals = np.array(vals[epoch]['MI_XM_' + infoplane_measure])[PLOT_LAYERS]
            ymvals = np.array(vals[epoch]['MI_YM_' + infoplane_measure])[PLOT_LAYERS]

            plt.plot(xmvals, ymvals, c=c, zorder=1)
            plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolors='none', zorder=2)

        plt.ylim([0, 1])
        plt.xlim(0, 12)
        plt.xlabel('I(X;M)')
        plt.ylabel('I(Y;M)')
        plt.title(activation)

    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
    plt.colorbar(sm, label='Epochs', cax=cbaxes)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # for run in range(12, 50):
    #     save_activation("tanh", False)
    #     save_activation("relu", False)
    #     computeMI(run)
    measures = avg_MI(list(range(36)))
    plot_IP(measures)