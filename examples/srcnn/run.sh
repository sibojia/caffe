#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe train --solver=srcnn_solver.prototxt
