# Coding Agent Instructions

This project is a local-only Microsoft Agent Framework agent using
the loopback OpenAI-compatible endpoint of Foundry Local. Never introduce Azure model inference, Azure credentials,
or a cloud fallback without an explicit user request.

## Key files

- `src/agent-framework-agent-basic-responses/main.py`: local model client and
  Responses HTTP server.
- `src/agent-framework-agent-basic-responses/inventario.py`: local function tools
  for Laboratorios Roster (stock, demand forecast, reorder point, alerts).
- `src/agent-framework-agent-basic-responses/data/`: illustrative CSV data.
- `src/agent-framework-agent-basic-responses/.env`: `FOUNDRY_LOCAL_MODEL` and
  `ENABLE_MICROSOFT_LEARN_MCP`.
- `src/agent-framework-agent-basic-responses/pyproject.toml` and `uv.lock`:
  reproducible Python dependencies.
- `.vscode/`: F5 debugging and Agent Inspector tasks.

## Development workflow

From the agent source directory, run `uv sync --frozen`, then
`uv run --frozen python main.py`. Use Foundry Local to acquire and load a model
such as `phi-4-mini`. No Azure sign-in or deployment is required.

Preserve the local Responses interface on port 8088 and existing conversation
behavior. Test model inference locally, not against an Azure endpoint.

The agent uses the public Microsoft Learn MCP at
`https://learn.microsoft.com/api/mcp` for read-only documentation search, article
fetching, and code samples. Keep the MCP connection inside the agent's async
lifecycle. This remote tool requires internet and receives tool arguments;
never send secrets or private source code in documentation queries.

If you are in VS Code, read the vscode-microsoft-foundry skill first.
