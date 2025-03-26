#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

cd docs

make html

cp CNAME _build/html/
