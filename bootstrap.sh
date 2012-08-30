#!/bin/bash

DEPENDENCIES="skdata sthor bangreadout bangmetric thoreano"

for pkg in ${DEPENDENCIES}; do
    (cd external/${pkg} && python setup.py develop);
done;
