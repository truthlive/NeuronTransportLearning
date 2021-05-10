#!/bin/bash
for i in {0046..0100}; do
    for j in {0001..0200};  do
        cd "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/$i/$j/"
        echo "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/$i/$j/"
        # ls -l | grep controlmesh_allparticle_ | wc -l
        ls -l | grep '.*\.vtk$' | wc -l
    done
done

# for i in {0001..0100}; do

#         cd "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/$i/"
#         echo "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/$i/"
#         ls -l | grep -rl controlmesh_allparticle_ | wc -l
#         # ls -l | grep '.*\.vtk$' | wc -l

# done

# ls -l | grep controlmesh_allparticle_ | wc -l