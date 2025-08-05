1. Using VLLM for Significant Large Model Inference Acceleration

conda create -n vllm python=3.10
conda activate vllm
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

2. Starting Inference

# Start the VLLM OpenAI-compatible API server.
python -m vllm.entrypoints.openai.api_server --tensor-parallel-size=1 --trust-remote-code --max-model-len 1024 --model THUDM/chatglm3-6b

# Specify IP and Port: --host 127.0.0.1 --port 8101
# Example of starting the server with a specific host and port.
python -m vllm.entrypoints.openai.api_server --port 8101 --tensor-parallel-size=1 --trust-remote-code --max-model-len 1024 --model THUDM/chatglm3-6b

# Example for starting the server using specific CUDA devices and a local model path.
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
--model="/data/mnt/ShareFolder/common_models/Ziya-Reader-13B-v1.0" \
--max-model-len=8192 \
--tensor-parallel-size=2 \
--trust-remote-code \
--port=8101

3. Testing

# Send a POST request to the VLLM /v1/completions endpoint.
curl http://127.0.0.1:8101/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "THUDM/chatglm3-6b",
        "prompt": "Please reply within 20 words, how old are you this year?",
        "max_tokens": 20,
        "temperature": 0
    }'

# Send a POST request to the VLLM /v1/completions endpoint for a multi-turn conversation.
# The 'history' field contains previous user and assistant turns.
curl -X POST "http://127.0.0.1:8101/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        "model": "THUDM/chatglm3-6b",
        "prompt": "What is your name?", 
        "history": [
            {"role": "user", "content": "Where were you born?"}, 
            {"role": "assistant", "content": "Born in Beijing"}
        ]}"

# Send a POST request to the VLLM /v1/chat/completions endpoint, which is more aligned with OpenAI's chat API.
# This uses the 'messages' array format with explicit roles.
curl -X POST "http://127.0.0.1:8101/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        "model": "THUDM/chatglm3-6b", 
        "messages": [
            {"role": "system", "content": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."}, 
            {"role": "user", "content": "Hello, tell me a story, about 100 words."}], 
        "stream": false,
        "max_tokens": 100, 
        "temperature": 0.8, 
        "top_p": 0.8}"


4. Launching Frontend Access

# Run an Nginx Docker container in detached mode (-d).
docker run -d \
--network=host \
--name nginx2 --restart=always \
-v $PWD/nginx/conf/nginx.conf:/etc/nginx/nginx.conf \
-v $PWD/nginx/html:/usr/share/nginx/html \
-v $PWD/nginx/logs:/var/log/nginx \
--privileged=true \
--restart=always \
nginx


Reference Documentation: https://docs.vllm.ai/en/latest/