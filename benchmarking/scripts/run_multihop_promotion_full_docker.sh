#!/usr/bin/env bash
set -euo pipefail

# Host-side launcher for the resumable MultiHop promotion benchmark.
#
# It keeps host results in benchmarking/results/promotion while running the
# Python benchmark inside the API container. Each invocation uses a unique
# container results directory and syncs only the selected model/condition JSONL
# files, so model-specific jobs can run concurrently without overwriting each
# other's in-progress files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER="${CONTAINER:-langconnect-api}"
HOST_RESULTS_DIR="${HOST_RESULTS_DIR:-${REPO_ROOT}/benchmarking/results/promotion}"
CONTAINER_SCRIPT="/app/benchmarking/scripts/run_multihop_promotion_full.py"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
CONTAINER_RESULTS_DIR="${CONTAINER_RESULTS_DIR:-/tmp/multihop_promotion_${RUN_ID}}"

DEFAULT_MODELS="qwen35_122b_port6000,qwen35_35b_port7000"
DEFAULT_CONDITIONS="wiki_off,wiki_doc_routing,wiki_static_refs"
MODELS_ARG="${DEFAULT_MODELS}"
CONDITIONS_ARG="${DEFAULT_CONDITIONS}"
MAX_REWRITES="3"
HAS_SUMMARY_PATH="false"

ARGS=("$@")
for ((index = 0; index < ${#ARGS[@]}; index++)); do
  arg="${ARGS[$index]}"
  case "${arg}" in
    --models)
      MODELS_ARG="${ARGS[$((index + 1))]}"
      ;;
    --models=*)
      MODELS_ARG="${arg#--models=}"
      ;;
    --conditions)
      CONDITIONS_ARG="${ARGS[$((index + 1))]}"
      ;;
    --conditions=*)
      CONDITIONS_ARG="${arg#--conditions=}"
      ;;
    --max-rewrites)
      MAX_REWRITES="${ARGS[$((index + 1))]}"
      ;;
    --max-rewrites=*)
      MAX_REWRITES="${arg#--max-rewrites=}"
      ;;
    --summary-path | --summary-path=*)
      HAS_SUMMARY_PATH="true"
      ;;
    --results-dir | --results-dir=*)
      echo "Do not pass --results-dir to this Docker wrapper; use HOST_RESULTS_DIR instead." >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a MODEL_KEYS <<< "${MODELS_ARG}"
IFS=',' read -r -a CONDITION_KEYS <<< "${CONDITIONS_ARG}"

mkdir -p "${HOST_RESULTS_DIR}"

docker exec "${CONTAINER}" mkdir -p /app/benchmarking/scripts "${CONTAINER_RESULTS_DIR}"
docker cp \
  "${REPO_ROOT}/benchmarking/scripts/run_multihop_promotion_full.py" \
  "${CONTAINER}:${CONTAINER_SCRIPT}"
CONTAINER_UID_GID="$(docker exec "${CONTAINER}" sh -lc 'printf "%s:%s" "$(id -u)" "$(id -g)"')"

sync_results() {
  docker cp "${CONTAINER}:${CONTAINER_RESULTS_DIR}/." "${HOST_RESULTS_DIR}/" \
    >/dev/null 2>&1 || true
}

trap sync_results EXIT

for model_key in "${MODEL_KEYS[@]}"; do
  for condition_key in "${CONDITION_KEYS[@]}"; do
    report="multihop_full_${model_key}_${condition_key}_rw${MAX_REWRITES}.jsonl"
    if [[ -f "${HOST_RESULTS_DIR}/${report}" ]]; then
      docker cp "${HOST_RESULTS_DIR}/${report}" "${CONTAINER}:${CONTAINER_RESULTS_DIR}/${report}"
    fi
  done
done

docker exec -u 0 "${CONTAINER}" \
  chown -R "${CONTAINER_UID_GID}" "${CONTAINER_RESULTS_DIR}"

EXTRA_ARGS=("$@")
if [[ "${HAS_SUMMARY_PATH}" == "false" ]]; then
  EXTRA_ARGS+=(
    --summary-path
    "${CONTAINER_RESULTS_DIR}/multihop_full_summary_${RUN_ID}_rw${MAX_REWRITES}.json"
  )
fi

docker exec -i \
  -e PYTHONPATH=/app \
  -w /app \
  "${CONTAINER}" \
  .venv/bin/python "${CONTAINER_SCRIPT}" \
    --results-dir "${CONTAINER_RESULTS_DIR}" \
    "${EXTRA_ARGS[@]}"

sync_results
