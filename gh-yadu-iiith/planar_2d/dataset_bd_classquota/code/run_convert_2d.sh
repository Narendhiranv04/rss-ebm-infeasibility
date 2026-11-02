#!/bin/bash
# All-movable conversion of the old 2D roots with immovable structures (laptop side: --order down, the server converts from the other
# end); when every root is converted it hands the cores back to the 2D runner (MAX_RUNNING 5).  Stop: touch 2d_plus_raw/CONVERT_STOP
B=/home/ayush/codebase/bd_samples
cd /home/ayush/codebase/latent-causal-tamp
export PYTHONPATH=$B/py310_shim:$B/dataset-bd:$B:/home/ayush/codebase/latent-causal-tamp PYTHONWARNINGS=ignore
python3 $B/convert_2d.py --out $B/2d_plus_raw --workers 2 --order down --handback 5 >> $B/convert_2d.log 2>&1
