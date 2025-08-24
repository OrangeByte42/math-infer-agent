import os

from huggingface_hub import snapshot_download


hf_cache_dir: str = os.path.join('.', 'data', 'hf_cache')
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ['HF_HUB_CACHE'] = hf_cache_dir

snapshot_download(repo_id='meta-llama/Llama-3.2-1B-Instruct', repo_type='model', cache_dir=hf_cache_dir)
snapshot_download(repo_id='meta-llama/Llama-3.2-3B-Instruct', repo_type='model', cache_dir=hf_cache_dir)
snapshot_download(repo_id='meta-llama/Llama-3.1-8B-Instruct', repo_type='model', cache_dir=hf_cache_dir)
snapshot_download(repo_id='microsoft/Phi-4-mini-instruct', repo_type='model', cache_dir=hf_cache_dir)




