#!/bin/bash

cd /pylon5/eg5fp3p/rachels/angranl_stuff/GN_NeuronTransport/
source activate /pylon5/eg5fp3p/rachels/angranl_stuff/neurongeo

python vtk2h5_tree.py --num-bc 1 --num-tstep 100 --data-path data/3bifurcation/ --out-fname 3bifurcation.h5