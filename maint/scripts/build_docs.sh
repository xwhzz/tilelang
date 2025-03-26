#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r docs/requirements.txt
python -m pip install .

cd docs
make html

cp CNAME _build/html/
