# 1. Kill any hung 'hf' processes
pkill -9 hf

# 2. Force the most stable, non-Xet, non-threaded mode
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=300

# 3. Download the small models one-by-one
hf download vikhyatk/moondream2 --local-dir ./models/moondream2
hf download Qwen/Qwen2-VL-2B-Instruct --local-dir ./models/qwen2-vl
hf download llava-hf/llava-1.5-7b-hf --local-dir ./models/llava
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/qwen2.5-3b