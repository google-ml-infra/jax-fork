#!/bin/bash

#if [[ -f $(pwd)/dist/jaxlib-0.4.36.dev20241207-cp310-cp310-manylinux2014_x86_64.whl  ]]; then
file=$(find dist/ -type f -name "jaxlib*.whl" -print -quit 2>/dev/null)
if [[ -f $file ]]; then
 echo "file found"
fi
