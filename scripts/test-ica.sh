#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="${ROOT}/.tmp"
mkdir -p "${TMP}"

if [[ -z "${ICA_OPENCODE_API_KEY:-}" ]]; then
  echo "ERROR: ICA_OPENCODE_API_KEY not set" >&2
  exit 1
fi

BASE="${ICA_API_BASE:-$(sed -n 's/^ICA_API_BASE = "\(.*\)"$/\1/p' "${ROOT}/opencode-launcher.py")}"
TIMEOUT="${REQUEST_TIMEOUT:-10}"
AUTH="Authorization: Bearer ${ICA_OPENCODE_API_KEY}"

echo "Base URL: ${BASE}"

echo "--- GET /models"
code=$(curl -sS -o "${TMP}/ica-models.json" -w "%{http_code}" --max-time "${TIMEOUT}" -H "${AUTH}" "${BASE}/models")
echo "HTTP ${code}"
if [[ "${code}" != "200" ]]; then
  jq -r '.error.message // .' "${TMP}/ica-models.json" 2>/dev/null || cat "${TMP}/ica-models.json"
  exit 1
fi

count=$(jq '.data | length' "${TMP}/ica-models.json")
echo "Models: ${count}"
jq -r '.data[].id' "${TMP}/ica-models.json" | sort | head -20

model="${ICA_TEST_MODEL:-$(jq -r '.data[0].id // empty' "${TMP}/ica-models.json")}"
if [[ -z "${model}" ]]; then
  echo "ERROR: no models returned" >&2
  exit 1
fi

echo "--- POST /chat/completions (${model})"
payload=$(jq -n --arg m "${model}" '{model: $m, max_tokens: 256, messages: [{role: "user", content: "Reply with: ok"}]}')
code=$(curl -sS -o "${TMP}/ica-chat.json" -w "%{http_code}" --max-time 60 -H "${AUTH}" -H "Content-Type: application/json" -d "${payload}" "${BASE}/chat/completions")
echo "HTTP ${code}"
if [[ "${code}" != "200" ]]; then
  jq -r '.error.message // .' "${TMP}/ica-chat.json" 2>/dev/null || cat "${TMP}/ica-chat.json"
  exit 1
fi
reply=$(jq -r '.choices[0].message.content // empty' "${TMP}/ica-chat.json")
if [[ -z "${reply}" ]]; then
  echo "ERROR: empty reply (finish_reason: $(jq -r '.choices[0].finish_reason' "${TMP}/ica-chat.json"))" >&2
  exit 1
fi
echo "Reply: ${reply}"
echo "OK"
