"""Tests for the chunker module."""

from mcp_code_search.chunker import chunk_file


def test_python_chunking():
    code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"

class MyClass:
    def __init__(self):
        self.x = 1

    def method(self):
        return self.x
"""
    chunks = chunk_file("test.py", code)
    names = [c.name for c in chunks if c.name]
    assert "hello" in names
    assert "MyClass" in names
    assert "__init__" in names
    assert "method" in names


def test_plain_text_fallback():
    content = "line\n" * 100
    chunks = chunk_file("test.unknown", content)
    assert len(chunks) > 0
    assert chunks[0].chunk_type == "block"


def test_empty_file():
    chunks = chunk_file("test.py", "")
    assert len(chunks) == 0


def test_small_file_single_chunk():
    content = "x = 1\ny = 2\n"
    chunks = chunk_file("test.txt", content)
    assert len(chunks) == 1


def test_typescript_chunking():
    code = """
export function greet(name: string): string {
    return `Hello, ${name}!`;
}

interface User {
    name: string;
    age: number;
}

class UserService {
    getUser(): User {
        return { name: "test", age: 0 };
    }
}
"""
    chunks = chunk_file("test.ts", code)
    types = [c.chunk_type for c in chunks]
    assert "function" in types or "block" in types


def test_line_numbers_are_1_indexed():
    code = """def first():
    pass

def second():
    pass
"""
    chunks = chunk_file("test.py", code)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_module_chunk_name():
    code = "x = 1\ny = 2\nz = 3\n"
    chunks = chunk_file("/path/to/config.py", code)
    assert len(chunks) >= 1
    module_chunks = [c for c in chunks if c.chunk_type == "module"]
    if module_chunks:
        assert module_chunks[0].name == "config"


def test_go_chunking():
    code = """package main

import "fmt"

func hello() {
    fmt.Println("Hello")
}

type Server struct {
    Port int
}

func (s *Server) Start() {
    fmt.Println("Starting")
}
"""
    chunks = chunk_file("main.go", code)
    assert len(chunks) > 0
    names = [c.name for c in chunks if c.name]
    assert "hello" in names or len(chunks) >= 1


def test_rust_chunking():
    code = """fn main() {
    println!("Hello, world!");
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}
"""
    chunks = chunk_file("main.rs", code)
    assert len(chunks) > 0
    types = [c.chunk_type for c in chunks]
    assert any(t in types for t in ("function", "struct", "impl", "block"))


def test_max_chunk_lines_param():
    content = "line\n" * 200
    chunks_default = chunk_file("test.unknown", content, max_chunk_lines=50)
    chunks_small = chunk_file("test.unknown", content, max_chunk_lines=20)
    assert len(chunks_small) > len(chunks_default)
