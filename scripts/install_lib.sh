#!/bin/bash

if [ -z "$P" ]; then
    echo "Please enter the installation path:"
    read installation_path
else
    installation_path=$P
fi

header_path="$installation_path/tensor"

mkdir -p "$header_path"

echo "Copying header files"
sudo cp -r tensor/tensor_lib "$header_path"
sudo cp tensor/tensor_lib.hpp "$header_path"

echo "Header-only library installed successfully to $installation_path"
