#!/bin/bash
# Gate A2 -- recovery curve: measured rho_LH vs injected p_spill.
# Spatial injection (SHARE_FRAC) held at 0 so the spectral term is isolated.
set -u
cd /home/ahmed-lamidi/Documents/Yi_Sheng_lab/PCCT
NV=400
mkdir -p gateA/results
for P in 0 0.01 0.05 0.10; do
  TAG=$(echo "$P" | tr -d '.')
  D=/tmp/synth_p$TAG
  echo "=== p_spill=$P -> $D ==="
  rm -rf "$D"
  SHARE_FRAC=0.0 P_SPILL=$P SEED=1 python3 files/make_synthetic.py "$D" $NV || exit 1
  NVIEWS=$NV python3 files/kill_test.py --data "$D" --out "gateA/results/p$TAG" \
      > "gateA/results/p$TAG.log" 2>&1
  echo "  --- verdict lines ---"
  grep -E "^(air_L|air_R|deep_centre)" "gateA/results/p$TAG.log" || tail -5 "gateA/results/p$TAG.log"
  rm -rf "$D"
done
echo "GATE_A2_COMPLETE"
