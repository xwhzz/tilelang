#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

cd docs

python -m pip install -r requirements.txt

make html

cp CNAME _build/html/
