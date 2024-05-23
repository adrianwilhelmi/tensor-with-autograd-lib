#!/bin/bash

if [ -z "$P" ]; then
    echo "Please enter the installation path:"
    read installation_path
else
    installation_path=$P
fi

header_path="$installation_path/include"

mkdir -p "$header_path"

echo "Copying header files"
sudo cp -r include/tensor "$header_path"
sudo cp include/tensor_lib.hpp "$header_path"

echo "Header-only library installed successfully to $installation_path"
