#!/bin/bash

docker build --tag simanneal-cuda-test .

docker run \
    --gpus all -ti \
    --cap-add=SYS_ADMIN \
    -v /home/samuelshng/git/simanneal-sidb:/app/simanneal-sidb \
    --name simanneal-cuda-test --rm simanneal-cuda-test \
    /bin/bash
