#!/usr/bin/env python3
"""
Cap'n'Proto Documentation Generator

Generates reStructuredText documentation from Cap'n'Proto schema files.
Each struct, interface, and enum gets its own RST page, grouped by source
schema file. Cross-references link types across pages.

Usage:
    python capnp_docs.py --input path/to/schema.capnp --output docs/_capnp
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Set

from capnp_parser import (
    parse_schema, parse_schema_file, Schema, Struct, Interface, Enum,
    Field, Method, UnionGroup
)


def sanitize_rst(text: str) -> str:
    """Sanitize text from .capnp comments for safe RST rendering.
    
    Escapes characters that RST would interpret as markup:
    - Asterisks (*) used for emphasis/strong
    - Backticks (`) used for interpreted text
    - Underscores (_) in the middle of words (e.g. nTurns_2pi)
    - Pipe (|) used for substitution references
    - Double dashes (--) at line starts
    
    But preserves intentional RST from the schema author if any.
    """
    if not text:
        return text
    # Escape asterisks that aren't part of intentional RST emphasis
    # A simple heuristic: escape all * since capnp comments rarely use RST emphasis
    text = text.replace('*', '\\*')
    # Escape backticks
    text = text.replace('`', '\\`')
    # Escape pipes
    text = text.replace('|', '\\|')
    # Escape underscores within words (e.g. nTurns_2pi -> nTurns\\_2pi)
    # But not at the end of words which could be intentional RST links
    text = re.sub(r'(\w)_(\w)', r'\1\\_\2', text)
    return text


def render_comment(comment: str, indent: str = "") -> list:
    """Render a comment as sanitized RST lines with given indentation."""
    if not comment:
        return []
    result = []
    for cl in comment.strip().split('\n'):
        result.append(f"{indent}{sanitize_rst(cl.strip())}")
    return result


@dataclass
class DocContext:
    """Shared context for documentation generation."""
    all_type_names: Set[str] = field(default_factory=set)
    type_to_schema: Dict[str, str] = field(default_factory=dict)
    import_aliases: Dict[str, str] = field(default_factory=dict)
    type_aliases: Dict[str, str] = field(default_factory=dict)
    current_schema: str = ""


# ---- Type formatting ----

def resolve_import_aliases(type_str: str, import_aliases: Dict[str, str] = None,
                           type_aliases: Dict[str, str] = None,
                           all_type_names: Set[str] = None) -> str:
    """Resolve import-prefixed type names like D.DataRef -> DataRef.
    
    Handles patterns like:
      D.DataRef(Float64Tensor) -> DataRef(Float64Tensor)
      G.Transformed           -> Transformed
      D.Float64Tensor         -> Float64Tensor (resolved via import alias D -> data, then checking all_type_names)
    
    type_aliases maps short_name -> prefixed_name (e.g. Float64Tensor -> D.Float64Tensor).
    This means when we see the prefixed form, we should use the short name instead.
    """
    if not import_aliases and not type_aliases:
        return type_str
    
    result = type_str
    
    # Build reverse map: prefixed_name -> short_name from type_aliases
    # e.g. D.Float64Tensor -> Float64Tensor
    reverse_aliases = {}
    if type_aliases:
        for short, prefixed in type_aliases.items():
            reverse_aliases[prefixed] = short
    
    # First try exact matches in reverse_aliases
    if result in reverse_aliases:
        return reverse_aliases[result]
    
    # Resolve import-prefixed names: D.Something -> Something
    if import_aliases:
        for alias, schema_name in import_aliases.items():
            # Match alias.TypeName (possibly with generic params like DataRef(Foo))
            pattern = re.escape(alias) + r'\.(\w+)'
            def replace_prefix(m, _all_types=all_type_names, _reverse=reverse_aliases):
                resolved = m.group(1)
                # If the full prefix.name is in reverse_aliases, use the short name
                full = m.group(0)
                if full in _reverse:
                    return _reverse[full]
                # If the resolved name is a known type, just use it directly
                if _all_types and resolved in _all_types:
                    return resolved
                # Otherwise keep the resolved name without the alias prefix
                return resolved
            result = re.sub(pattern, replace_prefix, result)
    
    return result


def format_type(type_str: str, ctx: DocContext = None) -> str:
    """Format a Cap'n'Proto type string for RST documentation.
    
    Keeps the original Cap'n'Proto type names (no Python mapping).
    Wraps known user-defined types in :doc: cross-references.
    For types in a different schema, uses the relative path.
    """
    result = type_str.strip()
    
    if ctx is None:
        return result
    
    # First, resolve import-prefixed type names (D.DataRef -> DataRef, G.Transformed -> Transformed)
    result = resolve_import_aliases(result, ctx.import_aliases, ctx.type_aliases, ctx.all_type_names)
    
    # Handle parameterized types like List(Float64), DataRef(Foo)
    # We want to cross-ref the inner type if it's a user type
    if ctx.all_type_names:
        # Find type names inside the string and wrap them
        for tname in sorted(ctx.all_type_names, key=len, reverse=True):
            # Only match whole words not already preceded by a doc directive
            pattern = r'(?<!:doc:`)(?<!\w)' + re.escape(tname) + r'(?!\w)(?!`)'
            if re.search(pattern, result):
                # Build the :doc: path
                if ctx.type_to_schema and tname in ctx.type_to_schema:
                    schema = ctx.type_to_schema[tname]
                    if schema != ctx.current_schema:
                        # Cross-schema reference: use relative path
                        doc_path = f"../{schema}/{tname}"
                    else:
                        doc_path = tname
                else:
                    doc_path = tname
                result = re.sub(pattern, f':doc:`{doc_path}`', result)
    
    # Fix: RST chokes on :doc:`...`(T) — closing backtick before ( confuses parser.
    # Insert escaped space \ between ` and ( to separate them.
    # This only occurs with generic types like Transformed(T) where the
    # :doc: cross-ref is followed by the generic parameter in parens.
    result = result.replace('`(', '`\ (')

    return result


# ---- Page generators ----

def generate_struct_page(struct: Struct, schema_name: str, ctx: DocContext,
                         is_nested: bool = False) -> str:
    """Generate an RST page for a single struct.
    
    Args:
        struct: The Struct to document
        schema_name: Name of the source schema file (without extension)
        ctx: Shared documentation context for type resolution
        is_nested: Whether this is a nested struct (shown under a parent)
    """
    lines = []
    
    # Title
    lines.append(struct.name)
    lines.append("=" * len(struct.name))
    lines.append("")
    
    # Schema source info
    lines.append(f".. rst-class:: schema-source")
    lines.append("")
    lines.append(f"Schema: :doc:`{schema_name}/index`")
    if is_nested and struct.parent:
        lines.append(f"Parent: :doc:`{struct.parent}`")
    lines.append("")
    
    # Comment if present
    if struct.comment:
        lines.extend(render_comment(struct.comment))
        lines.append("")
    
    # Fields
    if struct.fields or struct.unions:
        lines.append("Fields")
        lines.append("------")
        lines.append("")
        
        # Regular fields — use definition list style for clean RST
        for f in struct.fields:
            type_display = format_type(f.type_, ctx)
            default_str = f" = {f.default}" if f.default else ""
            lines.append(f"**{f.name}**")
            lines.append(f"   {type_display}{default_str}")
            if f.comment:
                lines.extend(render_comment(f.comment, "   "))
                lines.append("")
            lines.append("")
        
        # Union groups
        for ug in struct.unions:
            lines.extend(_render_union_group(ug, ctx, indent=0))
    
    # Nested enums (if any)
    if hasattr(struct, 'nested_enums') and struct.nested_enums:
        lines.append("")
        lines.append("Nested Enums")
        lines.append("------------")
        lines.append("")
        for ne in struct.nested_enums:
            lines.append(f":doc:`{ne.name}`")
            if ne.comment:
                lines.extend(render_comment(ne.comment, "   "))
    
    lines.append("")
    return "\n".join(lines)


def _render_union_group(ug: UnionGroup, ctx: DocContext, indent: int = 0) -> list:
    """Render a union or group recursively as RST definition list."""
    lines = []
    prefix = "  " * indent
    childPrefix = "  " * (indent + 1)
    kind = "union" if ug.kind == "union" else "group"
    label = ug.name if ug.name else f"anonymous {kind}"
    
    # Union/group header — bold label, no underline (avoids section title conflicts)
    lines.append(f"{prefix}**{label}** ({kind})")
    lines.append("")
    
    for f in ug.fields:
        type_display = format_type(f.type_, ctx)
        default_str = f" = {f.default}" if f.default else ""
        lines.append(f"{childPrefix}**{f.name}**")
        lines.append(f"{childPrefix}  {type_display}{default_str}")
        if f.comment:
            lines.extend(render_comment(f.comment, f"{prefix}  "))
        lines.append("")
    
    for nested in ug.nested_groups:
        lines.extend(_render_union_group(nested, ctx, indent=indent + 1))
    
    return lines


def generate_interface_page(interface: Interface, schema_name: str,
                            ctx: DocContext) -> str:
    """Generate an RST page for a single interface."""
    lines = []
    
    # Title
    lines.append(interface.name)
    lines.append("=" * len(interface.name))
    lines.append("")
    
    # Schema source info
    lines.append(f".. rst-class:: schema-source")
    lines.append("")
    lines.append(f"Schema: :doc:`{schema_name}/index`")
    if interface.extends:
        ext_refs = [format_type(e, ctx) for e in interface.extends]
        lines.append(f"Extends: {', '.join(ext_refs)}")
    lines.append("")
    
    # Comment if present
    if interface.comment:
        lines.extend(render_comment(interface.comment))
        lines.append("")
    
    # Methods
    if interface.methods:
        lines.append("Methods")
        lines.append("-------")
        lines.append("")
        
        for m in interface.methods:
            params_str = ", ".join(f"{p.name}: {format_type(p.type_, ctx)}" for p in m.parameters)
            ret_str = format_type(m.return_type, ctx)
            lines.append(f".. rst-class:: capnp-method")
            lines.append("")
            lines.append(f"{m.name}({params_str}) -> {ret_str}")
            lines.append("")
            if m.comment:
                lines.extend(render_comment(m.comment, "   "))
                lines.append("")
    
    # Nested structs
    if interface.nested_structs:
        lines.append("Nested Structs")
        lines.append("--------------")
        lines.append("")
        for ns in interface.nested_structs:
            lines.append(f":doc:`{ns.name}`")
            if ns.comment:
                lines.extend(render_comment(ns.comment, "   "))
            lines.append("")
    
    # Nested enums
    if interface.nested_enums:
        lines.append("Nested Enums")
        lines.append("------------")
        lines.append("")
        for ne in interface.nested_enums:
            lines.append(f":doc:`{ne.name}`")
            if ne.comment:
                lines.extend(render_comment(ne.comment, "   "))
            lines.append("")
    
    lines.append("")
    return "\n".join(lines)


def generate_enum_page(enum: Enum, schema_name: str, ctx: DocContext) -> str:
    """Generate an RST page for a single enum."""
    lines = []
    
    # Title
    lines.append(enum.name)
    lines.append("=" * len(enum.name))
    lines.append("")
    
    # Schema source info
    lines.append(f".. rst-class:: schema-source")
    lines.append("")
    lines.append(f"Schema: :doc:`{schema_name}/index`")
    if enum.parent_interface:
        lines.append(f"Parent: :doc:`{enum.parent_interface}`")
    lines.append("")
    
    # Comment if present
    if enum.comment:
        lines.extend(render_comment(enum.comment))
        lines.append("")
    
    # Enumerators
    if enum.values:
        lines.append("Values")
        lines.append("------")
        lines.append("")
        for v in enum.values:
            lines.append(f"- ``{v}``")
    
    lines.append("")
    return "\n".join(lines)


def generate_schema_index(schema: Schema, schema_name: str) -> str:
    """Generate an index page for all types in a schema."""
    lines = []
    
    title = f"{schema_name}.capnp"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    
    if schema.file_comment:
        lines.extend(render_comment(schema.file_comment))
        lines.append("")
    
    # Structs
    if schema.structs:
        lines.append("Structs")
        lines.append("-------")
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 1")
        lines.append("")
        for s in schema.structs:
            lines.append(f"   {s.name}")
        lines.append("")
    
    # Interfaces
    if schema.interfaces:
        lines.append("Interfaces")
        lines.append("----------")
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 1")
        lines.append("")
        for i in schema.interfaces:
            lines.append(f"   {i.name}")
        lines.append("")
    
    # Enums
    if schema.enums:
        top_enums = [e for e in schema.enums if not e.parent_interface]
        if top_enums:
            lines.append("Enums")
            lines.append("-----")
            lines.append("")
            lines.append(".. toctree::")
            lines.append("   :maxdepth: 1")
            lines.append("")
            for e in top_enums:
                lines.append(f"   {e.name}")
            lines.append("")
    
    return "\n".join(lines)


def generate_root_index(all_schema_names: List[str]) -> str:
    """Generate the root index page linking to all schemas."""
    lines = []
    
    lines.append("Cap'n'Proto API Reference")
    lines.append("=========================")
    lines.append("")
    lines.append(".. toctree::")
    lines.append("   :maxdepth: 2")
    lines.append("")
    for name in sorted(all_schema_names):
        lines.append(f"   {name}/index")
    lines.append("")
    
    return "\n".join(lines)


def generate_complete_docs(input_files: List[str], output_path: Path) -> None:
    """Generate complete documentation from multiple schema files.
    
    Each schema gets its own subdirectory with per-type RST pages.
    """
    all_schemas = []  # (schema_name, Schema)
    ctx = DocContext()
    
    # First pass: parse all schemas and collect type names
    for input_file in input_files:
        schema = parse_schema_file(input_file)
        schema_name = Path(input_file).stem
        all_schemas.append((schema_name, schema))
        
        for s in schema.structs:
            ctx.all_type_names.add(s.name)
            ctx.type_to_schema[s.name] = schema_name
        for i in schema.interfaces:
            ctx.all_type_names.add(i.name)
            ctx.type_to_schema[i.name] = schema_name
            for ns in i.nested_structs:
                ctx.all_type_names.add(ns.name)
                ctx.type_to_schema[ns.name] = schema_name
            for ne in i.nested_enums:
                ctx.all_type_names.add(ne.name)
                ctx.type_to_schema[ne.name] = schema_name
        for e in schema.enums:
            ctx.all_type_names.add(e.name)
            ctx.type_to_schema[e.name] = schema_name
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Root index
    schema_names = [name for name, _ in all_schemas]
    (output_path / "index.rst").write_text(generate_root_index(schema_names))
    print(f"Generated: {output_path / 'index.rst'}")
    
    # Second pass: generate pages
    for schema_name, schema in all_schemas:
        # Set per-schema context
        ctx.current_schema = schema_name
        ctx.import_aliases = schema.import_aliases
        ctx.type_aliases = schema.type_aliases
        
        schema_dir = output_path / schema_name
        schema_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-schema index
        index_content = generate_schema_index(schema, schema_name)
        (schema_dir / "index.rst").write_text(index_content)
        print(f"Generated: {schema_dir / 'index.rst'}")
        
        # Per-struct pages
        for struct in schema.structs:
            page = generate_struct_page(struct, schema_name, ctx)
            (schema_dir / f"{struct.name}.rst").write_text(page)
            print(f"Generated: {schema_dir / f'{struct.name}.rst'}")
        
        # Per-interface pages
        for interface in schema.interfaces:
            page = generate_interface_page(interface, schema_name, ctx)
            (schema_dir / f"{interface.name}.rst").write_text(page)
            print(f"Generated: {schema_dir / f'{interface.name}.rst'}")
            
            # Nested struct pages
            for ns in interface.nested_structs:
                page = generate_struct_page(ns, schema_name, ctx, is_nested=True)
                (schema_dir / f"{ns.name}.rst").write_text(page)
                print(f"Generated: {schema_dir / f'{ns.name}.rst'}")
            
            # Nested enum pages
            for ne in interface.nested_enums:
                page = generate_enum_page(ne, schema_name, ctx)
                (schema_dir / f"{ne.name}.rst").write_text(page)
                print(f"Generated: {schema_dir / f'{ne.name}.rst'}")
        
        # Per-enum pages (top-level only)
        for enum in schema.enums:
            if not enum.parent_interface:
                page = generate_enum_page(enum, schema_name, ctx)
                (schema_dir / f"{enum.name}.rst").write_text(page)
                print(f"Generated: {schema_dir / f'{enum.name}.rst'}")


def main():
    parser = argparse.ArgumentParser(description="Generate Cap'n'Proto documentation")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input .capnp schema files")
    parser.add_argument("--output", required=True,
                        help="Output directory for generated RST files")
    args = parser.parse_args()
    
    generate_complete_docs(args.input, Path(args.output))


if __name__ == "__main__":
    main()
