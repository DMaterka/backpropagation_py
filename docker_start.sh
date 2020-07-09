#!/bin/bash

if  [  `docker images python_backprop | grep latest | wc -l` -lt 1 ]; then
    docker build -t python_backprop -f Docker/Dockerfile .
fi

if  [  `docker ps -a --filter "name=python_backprop_container" | grep python_backprop_container | wc -l` -lt 1 ]; then
    imageReference=`docker images -f "reference=python_backprop" -q`
    echo $imageReference
    docker create --rm -t --name python_backprop_container -i $imageReference
fi

if  [  `docker ps --filter "name=python_backprop_container" | grep python_backprop_container | wc -l` -lt 1 ]; then
    docker start python_backprop_container
fi