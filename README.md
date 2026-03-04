# mcp-code-search

Local semantic code search MCP server. AST-aware chunking with Tree-sitter, hybrid search via LanceDB + sentence-transformers.

## Features

- Tree-sitter AST-aware code chunking (function, class, method, module)
- Hybrid search: vector similarity + full-text keyword (RRF merge)
- Incremental indexing (only re-indexes changed files)
- 20+ programming language support
- Fully local, no internet required

## Installation

```bash
uv pip install -e .
```

With Ollama support:

```bash
uv pip install -e ".[ollama]"
```

## MCP Configuration

Add to `claude_desktop_config.json` or your MCP client config:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-code-search", "mcp-code-search"]
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `index_directory` | Index a directory (Tree-sitter AST parsing) |
| `search_code` | Hybrid semantic + keyword search |
| `search_text` | Full-text keyword search (grep alternative) |
| `find_similar_code` | Find similar code snippets |
| `get_index_status` | View index statistics |
| `list_projects` | List indexed projects |

## Configuration

Project-level `.code-search.toml` or global `~/.config/mcp-code-search/config.toml`:

```toml
[embedding]
provider = "sentence-transformers"  # or "ollama"
model = "jinaai/jina-embeddings-v2-base-code"

[embedding.ollama]
model = "nomic-embed-text"
base_url = "http://localhost:11434"

[indexing]
max_file_size_kb = 500
batch_size = 32
max_chunk_lines = 50
ignore_patterns = ["node_modules", ".git", "dist"]

[search]
rrf_k = 60
snippet_max_lines = 30

[storage]
base_path = "~/.local/share/mcp-code-search"
```

## Embedding Models

Default model: `jinaai/jina-embeddings-v2-base-code` (768 dim, 307MB). All models run comfortably on MacBook Air M2 8GB.

### sentence-transformers (Local)

| Model | Params | Size | Dim | Notes |
|-------|--------|------|-----|-------|
| [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | ~80MB | 384 | Lightest, fastest. General purpose. |
| [`jinaai/jina-embeddings-v2-base-code`](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) | 161M | ~307MB | 768 | Code-specific, 30 languages, 8K token context. **Default.** |
| [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 137M | ~262MB | 768 | Matryoshka (adjustable dim), open license. |
| [`codesage/codesage-small`](https://huggingface.co/codesage/codesage-small) | 130M | ~250MB | 1024 | Code-specific, MLM + contrastive training. |
| [`codesage/codesage-base`](https://huggingface.co/codesage/codesage-base) | 356M | ~680MB | 1024 | Better quality but heavier. |
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33M | ~130MB | 384 | Very lightweight, good general performance. |
| [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) | 308M | <200MB\* | 768 | Best MTEB under 500M. Matryoshka, 100+ languages, code+docs training. 2K context. |

\* with quantization

### Ollama (Local)

| Model | Size | Dim | Notes |
|-------|------|-----|-------|
| `nomic-embed-text` | ~274MB | 768 | Default Ollama model. |
| `mxbai-embed-large` | ~670MB | 1024 | Higher quality. |

### How to Choose

- **Speed priority**: `all-MiniLM-L6-v2` or `bge-small-en-v1.5`
- **Code search quality**: `jina-embeddings-v2-base-code` (default) or `codesage-small`
- **Best general benchmark + low RAM**: `embeddinggemma-300m` (with quantization <200MB, but 2K context limit)
- **General purpose + flexible dim**: `nomic-embed-text-v1.5`

To change the model, update `.code-search.toml`:

```toml
[embedding]
model = "all-MiniLM-L6-v2"
```

When the model changes, dimension mismatch is automatically detected and a full reindex is performed.

## Testing

```bash
uv run pytest tests/ -v
```

## License

MIT
