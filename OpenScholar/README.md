# OpenScholar API Service

OpenScholar is a Retrieval-Augmented Generation (RAG) based academic research Q&A system that can search relevant papers through Semantic Scholar and generate high-quality academic answers.

## 🚀 Quick Start

### Step 1: Apply for Semantic Scholar API Key

1. Visit [Semantic Scholar API](https://www.semanticscholar.org/product/api)
2. Register an account and apply for API key
3. Record your API key for later use in configuration

### Step 2: Install Dependencies

```bash
pip install vllm
pip install flask
pip install transformers
pip install torch
pip install FlagEmbedding
pip install openai
```

### Step 3: Start Model Services

#### Linux/Mac Users

```bash
# Give execution permission to the startup script
chmod +x start_models.sh

# Start model services
./start_models.sh
```

#### Windows Users

```cmd
# Run the startup script directly
start_models.bat
```

#### Manual Model Service Startup

If the automatic script encounters issues, you can start manually:

```bash
# Start large model service
vllm serve OpenSciLM/Llama-3.1_OpenScholar-8B \
    --port 38011 \
    --max_model_len 70000 \
    --quantization fp8 \
    --gpu_memory_utilization=0.8

# Start small model service in another terminal
vllm serve Qwen/Qwen3-0.6B \
    --port 38014 \
    --max_model_len 10000 \
    --quantization fp8 \
    --gpu_memory_utilization=0.05
```

### Step 4: Configure and Start API Service

#### Method 1: Using Command Line Arguments

```bash
python openscholar_api.py \
    --s2_api_key YOUR_SEMANTIC_SCHOLAR_API_KEY \
    --reranker_path OpenSciLM/OpenScholar_Reranker \
    --api_key sk-your-api-key-here
```

#### Method 2: Modify Configuration File

1. Edit the `config.py` file
2. Modify the configuration parameters:
   ```python
   self.S2_API_KEY = "your_actual_s2_api_key"
   self.RERANKER_PATH = "OpenSciLM/OpenScholar_Reranker"
   ```
3. Run the API service:
   ```bash
   python openscholar_api.py
   ```

## 📋 Configuration Parameters

### Required Parameters

- `--s2_api_key`: Semantic Scholar API key
- `--reranker_path`: Reranker model path

### Optional Parameters

- `--api_key`: vLLM API key (default: "YOUR_API_KEY_HERE")
- `--large_model_port`: Large model service port (default: 38011)
- `--small_model_port`: Small model service port (default: 38014)
- `--api_port`: API service port (default: 38015)
- `--top_n`: Number of papers to retrieve (default: 10)
- `--max_tokens`: Maximum tokens for generation (default: 3000)
- `--search_batch_size`: Search batch size (default: 100)
- `--scholar_batch_size`: OpenScholar processing batch size (default: 100)

## 🔌 API Usage

### Batch Q&A Interface

**URL**: `POST http://localhost:38015/batch_ask`

**Request Format**:
```json
{
    "questions": [
        "How do retrieval-augmented LMs perform well in knowledge-intensive tasks?",
        "What are the latest developments in transformer architectures?"
    ],
    "titles": ["paper_title_1", "paper_title_2"]  // Optional
}
```

**Response Format**:
```json
{
    "results": [
        {
            "final_passages": "Retrieved relevant passages...",
            "output": "Generated answer...",
            "total_cost": 0.123,
            "keywords": ["keyword1", "keyword2", "keyword3"]
        }
    ]
}
```

### Usage Example

```python
import requests
import json

# API request example
url = "http://localhost:38015/batch_ask"
data = {
    "questions": [
        "How do retrieval-augmented LMs perform well in knowledge-intensive tasks?"
    ]
}

response = requests.post(url, json=data)
result = response.json()

print("Generated answer:", result["results"][0]["output"])
print("Used keywords:", result["results"][0]["keywords"])
```

## 🛠️ Troubleshooting

### Common Issues

1. **Slow model download**
   - Set HuggingFace mirror: `export HF_ENDPOINT=https://hf-mirror.com`

2. **Insufficient GPU memory**
   - Lower `gpu_memory_utilization` parameter
   - Reduce `max_model_len` parameter

3. **Port occupied**
   - Modify port numbers in configuration file
   - Use `netstat -tulpn | grep port_number` to check port usage

4. **API key error**
   - Confirm Semantic Scholar API key is valid
   - Check network connection

### Log Viewing

```bash
# View large model logs
tail -f large_model.log

# View small model logs
tail -f small_model.log
```

## 📊 Performance Optimization

### GPU Configuration Recommendations

- **8GB GPU**: Use smaller batch_size and max_model_len
- **16GB GPU**: Can use default configuration
- **24GB+ GPU**: Can increase batch_size and max_model_len for better performance

### Batch Processing Optimization

- Adjust `search_batch_size` and `scholar_batch_size` according to GPU memory
- Larger batch sizes can improve throughput but increase memory consumption

## 📚 Related Resources

- [OpenScholar Paper](https://arxiv.org/abs/2305.14334)
- [Semantic Scholar API Documentation](https://api.semanticscholar.org/)
- [vLLM Documentation](https://docs.vllm.ai/)

## 📝 License

This project follows the license terms of the main project. See [LICENSE.md](../LICENSE.md) for details. 
