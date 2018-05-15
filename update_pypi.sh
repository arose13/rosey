#!/usr/bin/env bash
# Requires package `twine`
# Requires `.pypirc` file to be in the home directory. eg `~/.pypirc`

python setup.py sdist
twine upload dist/*

echo "--- DONE ---"