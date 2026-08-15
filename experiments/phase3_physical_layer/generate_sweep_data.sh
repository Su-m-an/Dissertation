#!/bin/bash
# generate_sweep_data.sh
#
# Drives the original MATLAB generators (matlab/ATD_generator.m and
# ATD_generator_sequence.m, copied unmodified from Script Dataset/) across
# a sweep of rho_E (eavesdropper SNR) values, plus several extra baseline
# replicates for external validation. Explicit rng() seeding is added here
# (the originals have none) so every dataset in this experiment is
# reproducible by value, unlike the original baseline data.
#
# All other parameters held fixed at the values that produced the existing
# baseline data: K=4, L=10, rho_u=5, beta_k=1, beta_E=1, T=50, T0=5, T_hat=2000.

set -e
set -o pipefail
cd "$(dirname "$0")"

SWEEP_VALUES=(0.1 0.5 1 2 5 10 20)
EXTERNAL_SEEDS=(43 44 45 46 47)

echo "=== SNR/channel-difficulty sweep: rho_E in {${SWEEP_VALUES[*]}} ==="
for rho_E in "${SWEEP_VALUES[@]}"; do
    dir="raw_sweep/rho_E_${rho_E}"
    mkdir -p "$dir"
    cp matlab/ATD_generator.m matlab/ATD_generator_sequence.m "$dir/"
    echo "--- rho_E=$rho_E ---"
    matlab -batch "K=4;L=10;rho_u=5;rho_E=${rho_E};beta_k=1;beta_E=1;T=50;T0=5;T_hat=2000; rng(42); ATD_generator; rng(42); ATD_generator_sequence;" -sd "$dir" 2>&1 | grep -E "rows|Saved|Error"
done

echo ""
echo "=== External validation replicates (baseline rho_E=5, distinct seeds) ==="
for seed in "${EXTERNAL_SEEDS[@]}"; do
    dir="raw_external/seed_${seed}"
    mkdir -p "$dir"
    cp matlab/ATD_generator.m "$dir/"
    echo "--- seed=$seed ---"
    matlab -batch "K=4;L=10;rho_u=5;rho_E=5;beta_k=1;beta_E=1;T=50;T0=5;T_hat=2000; rng(${seed}); ATD_generator;" -sd "$dir" 2>&1 | grep -E "rows|Saved|Error"
done

echo ""
echo "All MATLAB generation complete."
