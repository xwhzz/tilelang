#! /bin/bash
docker run -d --name tl \
  -v $(pwd):/root/ \
  tl_118
