#!/bin/bash

# OpenScholar Model Startup Script
# Please ensure vllm package is installed: pip install vllm

echo "Starting OpenScholar model services..."

# Set CUDA device (optional, adjust according to your GPU configuration)
export CUDA_VISIBLE_DEVICES=0

# Start large model service (OpenScholar-8B)
echo "Starting large model service: OpenSciLM/Llama-3.1_OpenScholar-8B"
nohup vllm serve OpenSciLM/Llama-3.1_OpenScholar-8B \
    --port 38011 \
    --max_model_len 70000 \
    --quantization fp8 \
    --gpu_memory_utilization=0.8 \
    > large_model.log 2>&1 &

# Wait for large model to start
echo "Waiting for large model to start (30 seconds)..."
sleep 30

# Start small model service (Qwen3-0.6B)
echo "Starting small model service: Qwen/Qwen3-0.6B"
nohup vllm serve Qwen/Qwen3-0.6B \
    --port 38014 \
    --max_model_len 10000 \
    --quantization fp8 \
    --gpu_memory_utilization=0.05 \
    > small_model.log 2>&1 &

# Wait for small model to start
echo "Waiting for small model to start (20 seconds)..."
sleep 20

echo "Model services startup complete!"
echo "Large model port: 38011"
echo "Small model port: 38014"
echo ""
echo "View logs:"
echo "  Large model log: tail -f large_model.log"
echo "  Small model log: tail -f small_model.log"
echo ""
echo "Now you can start the OpenScholar API service:"
echo "  python openscholar_api.py --s2_api_key YOUR_SEMANTIC_SCHOLAR_API_KEY --reranker_path OpenSciLM/OpenScholar_Reranker" 