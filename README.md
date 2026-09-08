# FMNN25

Run these commands from the repository root to synchronize the dependencies and start JupyterLab with the correct configuration:

```bash
uv sync
uv run jupyter lab \
  --port 8888 \
  --ip 127.0.0.1 \
  --IdentityProvider.token=fmnn25-local \
  --ServerApp.root_dir=.
```

Then open `http://127.0.0.1:8888` in a browser. 

If `uv` is not installed on your machine, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
