#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import os

import h5py

import brainstorm as brnst
from brainstorm.data_iterators import Minibatches
# from brainstorm.handlers import PyCudaHandler

brnst.global_rnd.set_seed(42)

# ---------------------------- Set up Iterators ----------------------------- #

data_dir = os.environ.get('BRAINSTORM_DATA_DIR', 'data')
data_file = os.path.join(data_dir, 'MNIST.hdf5')
ds = h5py.File(data_file, 'r')['normalized_split']
x_tr, y_tr = ds['training']['default'][:], ds['training']['targets'][:]
x_va, y_va = ds['validation']['default'][:], ds['validation']['targets'][:]

getter_tr = Minibatches(100, default=x_tr, targets=y_tr)
getter_va = Minibatches(100, default=x_va, targets=y_va)

# ----------------------------- Set up Network ------------------------------ #

inp, fc = brnst.tools.get_in_out_layers('classification', (28, 28, 1), 10, projection_name='FC')
network = brnst.Network.from_layer(
    inp >>
    brnst.layers.Dropout(drop_prob=0.2) >>
    brnst.layers.FullyConnected(1200, name='Hid1', activation='rel') >>
    brnst.layers.Dropout(drop_prob=0.5) >>
    brnst.layers.FullyConnected(1200, name='Hid2', activation='rel') >>
    brnst.layers.Dropout(drop_prob=0.5) >>
    fc
)

# Uncomment next line to use GPU
# network.set_handler(PyCudaHandler())
network.initialize(brnst.initializers.Gaussian(0.05))
network.set_weight_modifiers({"FC": brnst.value_modifiers.ConstrainL2Norm(1)})

# ----------------------------- Set up Trainer ------------------------------ #

trainer = brnst.Trainer(brnst.training.MomentumStepper(learning_rate=0.1, momentum=0.9))
trainer.add_hook(brnst.hooks.ProgressBar())
# trainer.add_hook(brnst.hooks.BokehVisualizer('validation.Accuracy'))
scorers = [brnst.scorers.Accuracy(out_name='Output.outputs.predictions')]
trainer.add_hook(brnst.hooks.MonitorScores('valid_getter', scorers,
                                          name='validation'))
trainer.add_hook(brnst.hooks.SaveBestNetwork('validation.Accuracy',
                                          filename='mnist_pi_best500.hdf5',
                                          name='best weights',
                                          criterion='max'))
trainer.add_hook(brnst.hooks.StopAfterEpoch(10))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("Best validation accuracy:", max(trainer.logs["validation"]["Accuracy"]))
