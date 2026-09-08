# FMNN25 Project Instructions

## Notebook workflow

- Use `project1/project01.ipynb` as the main assignment artifact for assignment 1.
- Create, edit, and run the notebook through the configured Jupyter MCP tools. Do not edit notebook JSON directly.
- Keep Markdown explanations short and separate the numerical components clearly.
- Write straightforward code. Avoid defensive scaffolding, excessive comments, and unnecessary abstractions.

## Starting Jupyter

The MCP adapter in `.codex/config.toml` connects to an existing Jupyter server; it does not start JupyterLab itself. Before notebook work, the user should start JupyterLab from the repository root with:

```bash
uv run jupyter lab \
  --port 8888 \
  --ip 127.0.0.1 \
  --IdentityProvider.token=fmnn25-local \
  --ServerApp.root_dir=.
```

Confirm that `http://127.0.0.1:8888` responds. If the Jupyter MCP tools are still absent, ask the user to reconnect or reload the coding agent harness after the server is running.

For agent harnesses other than Codex, guide the user (or help user) to install the Jupyter MCP adapter:
https://github.com/datalayer/jupyter-mcp-server

## MPI workflow

- A normal Jupyter kernel is a single MPI process unless the kernel itself was launched under `mpiexec`.
- Keep the derivation, reusable numerical functions, experiment launch, results, and plots readable from the notebook.
- Put only the small `mpi4py` process orchestration in a separate Python driver, then launch it from the notebook with three ranks. Do not duplicate the numerical implementation between the notebook and the driver.
