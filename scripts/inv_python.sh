#!/bin/bash
# List torch + NCCL wheel versions in each conda env / venv
for d in ~/miniconda3 ~/miniconda3/envs/* ~/venvs/*; do
  for sp in "$d"/lib/python3*/site-packages; do
    [ -d "$sp" ] || continue
    tv=$(grep -h "^__version__" "$sp/torch/version.py" 2>/dev/null)
    nv=$(ls -d "$sp"/nvidia_nccl_cu*.dist-info 2>/dev/null | xargs -n1 basename 2>/dev/null | sort -u | tr '\n' ' ')
    [ -n "$tv$nv" ] && echo "$d | torch: ${tv:-none} | nccl wheel: ${nv:-none}"
  done
done
