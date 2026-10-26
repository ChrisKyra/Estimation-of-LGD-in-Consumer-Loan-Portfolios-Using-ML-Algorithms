#!/usr/bin/env zsh
# Run notebooks 01-04 in order. Executed copies (with all outputs) go to outputs/executed/.
#   ./run_all.sh          full run
#   ./run_all.sh --quick  smoke test: 3 trials / 3 epochs per model, finishes in a few minutes
set -euo pipefail
cd "${0:A:h}"

[[ "${1:-}" == "--quick" ]] && export LGD_QUICK=1
output_dir="${LGD_OUTPUT_DIR:-outputs}/executed"
mkdir -p "$output_dir"

for notebook in notebooks/0[1-4]_*.ipynb; do
  echo "==> $notebook"
  jupyter nbconvert --to notebook --execute "$notebook" \
    --ExecutePreprocessor.kernel_name=lgd --ExecutePreprocessor.timeout=-1 \
    --output-dir "$output_dir"
done
echo "Done. Results: outputs/  |  figures: outputs/figures/  |  executed notebooks: $output_dir/"
