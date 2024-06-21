#!/bin/bash

if [ -d ".venv" ]; then
	source .venv/bin/activate
fi

python -m preprocessing
python -m fe