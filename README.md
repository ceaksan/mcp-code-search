# mcp-code-search

Local semantic code search MCP server. LanceDB + sentence-transformers + Tree-sitter ile AST-aware chunking ve hybrid search.

## Ozellikler

- Tree-sitter ile AST-aware kod chunking (function, class, method, module)
- Hybrid search: vector similarity + full-text keyword (RRF merge)
- Incremental indexing (sadece degisen dosyalari yeniden indexle)
- 20+ programlama dili destegi
- Tamamen lokal, internet gerektirmez

## Kurulum

```bash
uv pip install -e .
```

Ollama destegi icin:

```bash
uv pip install -e ".[ollama]"
```

## MCP Yapilandirmasi

`claude_desktop_config.json` veya MCP client config'ine ekle:

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

## Araclar

| Arac | Aciklama |
|------|----------|
| `index_directory` | Dizini indexle (Tree-sitter AST parsing) |
| `search_code` | Hybrid semantic + keyword arama |
| `search_text` | Full-text keyword arama (grep alternatifi) |
| `find_similar_code` | Benzer kod parcalari bul |
| `get_index_status` | Index istatistiklerini gor |
| `list_projects` | Indexlenmis projeleri listele |

## Yapilandirma

Proje kokune `.code-search.toml` veya global `~/.config/mcp-code-search/config.toml` ile:

```toml
[embedding]
provider = "sentence-transformers"  # veya "ollama"
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

## Embedding Modelleri

Default model: `jinaai/jina-embeddings-v2-base-code` (768 dim, 307MB). Tum modeller MacBook Air M2 8GB'da rahat calisir.

### sentence-transformers (Lokal)

| Model | Param | Boyut | Dim | Ozellik |
|-------|-------|-------|-----|---------|
| [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | ~80MB | 384 | En hafif, en hizli. Genel amacli. |
| [`jinaai/jina-embeddings-v2-base-code`](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) | 161M | ~307MB | 768 | Kod-spesifik, 30 dil, 8K token context. **Default.** |
| [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 137M | ~262MB | 768 | Matryoshka (dim kisilabiir), acik lisans. |
| [`codesage/codesage-small`](https://huggingface.co/codesage/codesage-small) | 130M | ~250MB | 1024 | Kod-spesifik, MLM + contrastive training. |
| [`codesage/codesage-base`](https://huggingface.co/codesage/codesage-base) | 356M | ~680MB | 1024 | Daha iyi kalite ama daha agir. |
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33M | ~130MB | 384 | Cok hafif, iyi genel performans. |
| [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) | 308M | <200MB* | 768 | MTEB 500M alti en iyi. Matryoshka, 100+ dil, code+docs egitimi. 2K context. |

\* quantize ile

### Ollama (Lokal)

| Model | Boyut | Dim | Ozellik |
|-------|-------|-----|---------|
| `nomic-embed-text` | ~274MB | 768 | Varsayilan Ollama modeli. |
| `mxbai-embed-large` | ~670MB | 1024 | Daha yuksek kalite. |

### Nasil Secilir

- **Hiz oncelikliyse**: `all-MiniLM-L6-v2` veya `bge-small-en-v1.5`
- **Kod arama kalitesi oncelikliyse**: `jina-embeddings-v2-base-code` (default) veya `codesage-small`
- **En iyi genel benchmark + dusuk RAM**: `embeddinggemma-300m` (quantize ile <200MB, ama 2K context limiti var)
- **Genel amacli + esnek dim**: `nomic-embed-text-v1.5`

Model degistirmek icin `.code-search.toml`:

```toml
[embedding]
model = "all-MiniLM-L6-v2"
```

Model degistiginde dimension uyumsuzlugu otomatik algilanir ve full reindex yapilir.

## Test

```bash
uv run pytest tests/ -v
```

## Lisans

MIT
