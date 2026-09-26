#!/usr/bin/env bash
# Run from any directory; only the question3 source tree is required.
set -euo pipefail
Q3_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$Q3_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
stage="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
 help|-h|--help)
  echo 'Usage: bash question3/run_q3_server.sh {check|test|train|explain|report|audit|all|mapping|media} [options]'
  echo 'Environment: Q3_DATA_ROOT, Q3_BERT, Q3_OUT (or Q3_RUN), MATHE_DEVICE, Q3_PYTHON'
  exit 0 ;;
 check|test|train|explain|report|audit|all|mapping|media) ;;
 *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
Q3_PYTHON="${Q3_PYTHON:-python}"
if [[ "$stage" == test ]]; then
  "$Q3_PYTHON" -B question3/test_q3.py "$@"
  exit 0
fi
Q3_RUN="${Q3_RUN:-q3_final_01}"
Q3_OUT="${Q3_OUT:-$Q3_ROOT/outputs/question3/$Q3_RUN}"
Q3_BERT="${Q3_BERT:-$Q3_ROOT/work/pretrained_models/bert}"
Q3_DEVICE="${MATHE_DEVICE:-cuda:0}"
Q3_DATA_ROOT="${Q3_DATA_ROOT:-${MATHE_DATA:-}}"
if [[ -z "$Q3_DATA_ROOT" ]]; then
  for candidate in "$Q3_ROOT/E题数据" "$Q3_ROOT/E题数据/E题数据"; do
    if [[ -f "$candidate/附件2-数据集特征文件/aligned_50.pkl" ]]; then
      Q3_DATA_ROOT="$candidate"
      break
    fi
  done
fi
if [[ -z "$Q3_DATA_ROOT" ]]; then
  echo 'Set Q3_DATA_ROOT to the directory containing 附件2-数据集特征文件/.' >&2
  exit 2
fi
case "$stage" in
 check|train|explain|report|all|audit)
  mkdir -p "$Q3_OUT/logs"
  "$Q3_PYTHON" -B -u question3/q3_run.py "$stage" --data-root "$Q3_DATA_ROOT" --bert "$Q3_BERT" \
    --out "$Q3_OUT" --device "$Q3_DEVICE" "$@" 2>&1 | tee "$Q3_OUT/logs/${stage}_$(date +%Y%m%d_%H%M%S).log" ;;
 mapping)
  "$Q3_PYTHON" -B question3/q3_mapping.py --data-root "$Q3_DATA_ROOT" --bert "$Q3_BERT" \
    --out "$Q3_OUT/时间映射" --device "$Q3_DEVICE" "$@" ;;
 media)
  "$Q3_PYTHON" -B question3/q3_media.py --data-root "$Q3_DATA_ROOT" --out "$Q3_OUT" "$@" ;;
esac
