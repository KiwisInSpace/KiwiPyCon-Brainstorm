#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import brainstorm as brnst
import numpy as np
from PIL import Image

network = brnst.Network.from_hdf5('mnist_pi_best500.hdf5')

# Uncomment to get information on network structure
# brnst.tools.print_network_info(network)

image = Image.open("data/test_3.jpg")
data = np.array(image).reshape(
        image.size[0], image.size[1], 3).dot(
        [0.2, 0.7, 0.1]).reshape(
        image.size[0], image.size[1], 1) / 255

network.provide_external_data(
        {'default': np.array([[data]])},
        all_inputs=False)

network.forward_pass(training_pass=False)
classification = network.get('Output.outputs.predictions')[0][0]

print("- " * 7)
print(np.argmax(classification))
print("- " * 7)
for num, val in enumerate(classification):
    print(num, val)
