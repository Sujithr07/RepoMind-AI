from dataclasses import dataclass
from typing import Iterator
from pathlib import Path
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust
import tree_sitter_cpp as tscpp
import tree_sitter_c as tsc
import tree_sitter_ruby as tsruby
import tree_sitter_c_sharp as tscsharp
import tree_sitter_php as tsphp
import tree_sitter_scala as tsscala
from tree_sitter import Language, Parser, Node

LANGUAGE_MAP = {
    ".py": Language(tspython.language()),
    ".js": Language(tsjavascript.language()),
    ".jsx": Language(tsjavascript.language()),
    ".ts": Language(tstypescript.language_typescript()),
    ".tsx": Language(tstypescript.language_tsx()),
    ".java": Language(tsjava.language()),
    ".go": Language(tsgo.language()),
    ".rs": Language(tsrust.language()),
    ".cpp": Language(tscpp.language()),
    ".cc": Language(tscpp.language()),
    ".cxx": Language(tscpp.language()),
    ".c": Language(tsc.language()),
    ".rb": Language(tsruby.language()),
    ".cs": Language(tscsharp.language()),
    ".php": Language(tsphp.language_php()),
    ".scala": Language(tsscala.language()),
}

CHUNK_NODE_TYPES = {
    "python": {"function_definition", "async_function_definition", "class_definition"},
    "javascript": {
        "function_definition",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "export_statement",
    },
    "typescript": {
        "function_definition",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "export_statement",
    },
    "java": {"method_declaration", "class_declaration", "interface_declaration", "enum_declaration"},
    "go": {"function_declaration", "type_declaration", "method_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "trait_item", "enum_item"},
    "cpp": {"function_definition", "class_definition", "struct_specifier"},
    "c": {"function_definition"},
    "ruby": {"method", "class", "module"},
    "csharp": {"method_declaration", "class_declaration", "interface_declaration"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
    "scala": {"function_definition", "class_definition", "object_definition", "trait_definition"},
}


@dataclass
class CodeChunk:
    content: str
    context_prefix: str
    file_path: str
    language: str
    chunk_type: str
    name: str | None
    start_line: int
    end_line: int


def get_parser(file_path: str) -> Parser | None:
    """Get a parser for the given file path based on its extension.
    Returns None if the file extension is not supported."""
    ext = Path(file_path).suffix.lower()
    lang = LANGUAGE_MAP.get(ext)
    if lang is None:
        return None
    return Parser(lang)


def _node_name(node: Node, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    return name_node.text.decode() if name_node else None


def _build_prefix(
    file_path: str, parent_name: str | Node | None, node_name: str | None
) -> str:
    parts = [file_path]
    if parent_name and isinstance(parent_name, str):
        parts.append(parent_name)
    if node_name:
        parts.append(node_name)
    return " > ".join(parts)


def _walk(
    node: Node,
    source: bytes,
    file_path: str,
    language: str,
    parent_name: str | None = None,
) -> Iterator[CodeChunk]:
    """Recursively walk the AST and yield CodeChunk for each function/class/method."""
    target_types = CHUNK_NODE_TYPES.get(language, set())

    if node.type in target_types:
        name = _node_name(node, source)
        chunk_type = "class" if "class" in node.type else "function"
        content = source[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

        # Skip tiny stubs (<3 lines) - not worth indexing
        if node.end_point[0] - node.start_point[0] >= 2:
            yield CodeChunk(
                content=content,
                context_prefix=_build_prefix(file_path, parent_name, name),
                file_path=file_path,
                language=language,
                chunk_type=chunk_type,
                name=name,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
            )

        # Recurse into class bodies - methods become their own chunks
        if chunk_type == "class":
            for child in node.children:
                yield from _walk(child, source, file_path, language, parent_name=name)

    else:
        for child in node.children:
            yield from _walk(
                child, source, file_path, language, parent_name=parent_name
            )


def chunk_file(file_path: str, source_code: str, language: str) -> list[CodeChunk]:
    parser = get_parser(file_path)
    if parser is None:
        return []
    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)
    chunks = list(_walk(tree.root_node, source_bytes, file_path, language))

    ## Fallback: if AST yields nothing, treat whole file as one module
    if not chunks and len(source_code.strip()) > 0:
        chunks = [
            CodeChunk(
                content=source_code,
                context_prefix=file_path,
                file_path=file_path,
                language=language,
                chunk_type="module",
                name="module",
                start_line=0,
                end_line=source_code.count("\n"),
            )
        ]

    return chunks
