from os import environ as env
from pathlib import Path

# specify your project name (used for logging and caching)
PROJECT_NAME = "conceptarium" 

# specify your wandb identity (used for logging)
WANDB_ENTITY = "" 

CACHE = Path(
    env.get(
        f"{PROJECT_NAME.upper()}_CACHE",
        Path(
            env.get("XDG_CACHE_HOME", Path("~", ".cache")),
            PROJECT_NAME,
        ),
    )
).expanduser()
CACHE.mkdir(exist_ok=True)

# if needed, set your huggingface token here
HUGGINGFACEHUB_TOKEN=''   

# if needed, set your openai api key here
OPENAI_API_KEY=''