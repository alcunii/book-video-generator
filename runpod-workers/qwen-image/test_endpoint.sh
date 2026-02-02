#!/bin/bash
# Test script for Qwen-Image-2512 RunPod Serverless Endpoint
# Usage: bash test_endpoint.sh

ENDPOINT_ID="${RUNPOD_IMAGE_ENDPOINT_ID:-mdmbtm3xut6m0a}"
API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY environment variable}"
BASE_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}"

echo "=== Submitting test job to Qwen-Image-2512 endpoint ==="
echo ""

RESPONSE=$(curl -s -X POST "${BASE_URL}/run" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A serene person meditating in a sunlit room, peaceful atmosphere, soft morning light, mindfulness concept, no text",
      "negative_prompt": "text, words, letters, watermark, blurry, low quality, distorted",
      "width": 928,
      "height": 1664,
      "num_inference_steps": 30,
      "cfg_scale": 4.0,
      "seed": 42
    }
  }')

echo "Submit response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)

if [ -z "$JOB_ID" ]; then
  echo "ERROR: Failed to get job ID from response"
  exit 1
fi

echo "Job ID: ${JOB_ID}"
echo "Polling for completion (first run downloads ~55GB model, be patient)..."
echo ""

while true; do
  STATUS_RESPONSE=$(curl -s "${BASE_URL}/status/${JOB_ID}" \
    -H "Authorization: Bearer ${API_KEY}")

  STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','UNKNOWN'))" 2>/dev/null)

  echo "[$(date +%H:%M:%S)] Status: ${STATUS}"

  if [ "$STATUS" = "COMPLETED" ]; then
    echo ""
    echo "=== Job completed! ==="

    # Extract output without the base64 blob for readability
    echo "$STATUS_RESPONSE" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
output = data.get('output', {})
if 'error' in output:
    print(f'ERROR: {output[\"error\"]}')
    sys.exit(1)
seed = output.get('seed', '?')
width = output.get('width', '?')
height = output.get('height', '?')
b64 = output.get('image_base64', '')
size_kb = len(base64.b64decode(b64)) / 1024 if b64 else 0
print(f'Seed: {seed}')
print(f'Size: {width}x{height}')
print(f'Image: {size_kb:.0f} KB')

# Save the image
with open('/tmp/qwen_test_output.png', 'wb') as f:
    f.write(base64.b64decode(b64))
print(f'Saved to: /tmp/qwen_test_output.png')
"
    break

  elif [ "$STATUS" = "FAILED" ]; then
    echo ""
    echo "=== Job FAILED ==="
    echo "$STATUS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$STATUS_RESPONSE"
    exit 1

  elif [ "$STATUS" = "CANCELLED" ]; then
    echo "Job was cancelled."
    exit 1
  fi

  sleep 15
done
