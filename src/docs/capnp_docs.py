#!/usr/bin/env python3
"""
Cap'n'Proto Documentation Generator

Generates reStructuredText documentation from Cap'n'Proto schema files.
Each struct, interface, and enum gets its own RST page, grouped by source
schema file. Nested types (struct-in-struct, struct-in-interface) have their
pages nested below their parent's directory. Fully qualified names are used
as titles and for cross-references.

Directory structure:
  _capnp/
    index.rst                         # Root index
    schema_name/
      index.rst                       # Schema index (toctree of top-level types)
      TopLevelStruct.rst              # Top-level struct page
      TopLevelStruct/
        NestedStruct.rst              # Nested struct page
        NestedStruct/
          DeeperNested.rst            # Multiply-nested struct page
      TopLevelInterface.rst           # Top-level interface page
      TopLevelInterface/
        NestedInInterface.rst         # Nested struct inside interface
      TopEnum.rst                     # Top-level enum page

Usage:
    python capnp_docs.py --input path/to/schema.capnp --output docs/_capnp
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple

from capnp_parser import (
    parse_schema, parse_schema_file, Schema, Struct, Interface, Enum,
    Field, Method, UnionGroup
)


def sanitize_rst(text: str) -> str:
    """Sanitize text from .capnp comments for safe RST rendering."""
    if not text:
        return text
    text = text.replace('*', '\\*')
    text = text.replace('`', '\\`')
    text = text.replace('|', '\\|')
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
    # Maps qualified_name or short_name -> (schema_name, doc_path_within_schema)
    # doc_path_within_schema is like "LoadBalancerConfig/Backend" or "WarehouseConfig"
    all_type_names: Set[str] = field(default_factory=set)
    type_to_schema: Dict[str, str] = field(default_factory=dict)
    type_to_doc_path: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    import_aliases: Dict[str, str] = field(default_factory=dict)
    type_aliases: Dict[str, str] = field(default_factory=dict)
    current_schema: str = ""


# ---- Type formatting ----

def resolve_import_aliases(type_str: str, import_aliases: Dict[str, str] = None,
                           type_aliases: Dict[str, str] = None,
                           all_type_names: Set[str] = None) -> str:
    """Resolve import-prefixed type names like D.DataRef -> DataRef."""
    if not import_aliases and not type_aliases:
        return type_str
    
    result = type_str
    
    # Build reverse map: prefixed_name -> short_name from type_aliases
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
            pattern = re.escape(alias) + r'\.(\w+)'
            def replace_prefix(m, _all_types=all_type_names, _reverse=reverse_aliases):
                resolved = m.group(1)
                full = m.group(0)
                if full in _reverse:
                    return _reverse[full]
                if _all_types and resolved in _all_types:
                    return resolved
                return resolved
            result = re.sub(pattern, replace_prefix, result)
    
    return result


def _doc_ref_abs(tname: str, ctx: DocContext) -> str:
    """Compute an absolute :doc: path from Sphinx source root for a type name.
    
    Uses the leading / to make it an absolute Sphinx doc reference.
    The _capnp directory is under the Sphinx source root, so all absolute
    references must include the _capnp/ prefix.
    """
    if tname in ctx.type_to_doc_path:
        schema_name, doc_path = ctx.type_to_doc_path[tname]
        return f"/_capnp/{schema_name}/{doc_path}"
    return tname.replace('.', '/')


def format_type(type_str: str, ctx: DocContext = None) -> str:
    """Format a Cap'n'Proto type string for RST documentation.
    
    Keeps the original Cap'n'Proto type names (no Python mapping).
    Wraps known user-defined types in :doc: cross-references.
    Uses absolute paths (from _capnp root) so references work from any depth.
    """
    result = type_str.strip()
    
    if ctx is None:
        return result
    
    # First, resolve import-prefixed type names
    result = resolve_import_aliases(result, ctx.import_aliases, ctx.type_aliases, ctx.all_type_names)
    
    # Build a list of type name replacements.
    # Process longest names first to avoid partial matches.
    # Track claimed character positions to avoid double-wrapping.
    if ctx.all_type_names:
        matches = []  # list of (start, end, tname, doc_path)
        claimed = set()
        
        for tname in sorted(ctx.all_type_names, key=len, reverse=True):
            pattern = re.compile(r'(?<!\w)' + re.escape(tname) + r'(?!\w)')
            for m in pattern.finditer(result):
                start, end = m.start(), m.end()
                if any(pos in claimed for pos in range(start, end)):
                    continue
                doc_path = _doc_ref_abs(tname, ctx)
                matches.append((start, end, tname, doc_path))
                for pos in range(start, end):
                    claimed.add(pos)
        
        # Sort right-to-left so replacement indices stay valid
        matches.sort(key=lambda x: x[0], reverse=True)
        
        for start, end, tname, doc_path in matches:
            result = result[:start] + f':doc:`{doc_path}`' + result[end:]
    
    # Fix: RST chokes on :doc:`...`(T) — insert escaped space
    result = result.replace('`(', '`\\ (')

    return result


# ---- Doc path computation ----

def _doc_path_for_type(qualified_name: str) -> str:
    """Convert a qualified name to a doc path within the schema directory.
    
    e.g., "LoadBalancerConfig.Backend" -> "LoadBalancerConfig/Backend"
          "WarehouseConfig" -> "WarehouseConfig"
          "ReversibleFieldlineMapping.Section.Inverse" -> "ReversibleFieldlineMapping/Section/Inverse"
    """
    return qualified_name.replace('.', '/')


def _toctree_ref_from_page(child_qname: str, page_qname: str) -> str:
    """Compute a toctree reference from a parent page to its child.
    
    toctree entries resolve relative to the directory of the RST file.
    
    For a page at schema_dir/LoadBalancerConfig.rst referencing
    child LoadBalancerConfig.Backend (at schema_dir/LoadBalancerConfig/Backend.rst):
      toctree ref = "LoadBalancerConfig/Backend"
    
    For a page at schema_dir/LoadBalancerConfig/Rule.rst referencing
    child LoadBalancerConfig.Rule.MethodSpec (at schema_dir/LoadBalancerConfig/Rule/MethodSpec.rst):
      toctree ref = "Rule/MethodSpec"
      (because from the directory schema_dir/LoadBalancerConfig/,
       the path to MethodSpec.rst is Rule/MethodSpec)
    
    The general rule: strip the page's directory path prefix from the child's full path.
    """
    child_path = _doc_path_for_type(child_qname)
    page_path = _doc_path_for_type(page_qname)
    
    # page_path might be "LoadBalancerConfig" or "LoadBalancerConfig/Rule"
    # If the page is a top-level type (no /), the child path starts with page_path/
    # If the page is nested, we need to compute relative path
    
    # The RST file for page_qname is at schema_dir/page_path.rst
    # Its containing directory is schema_dir/ + everything before the last /
    # or just schema_dir/ if there's no /
    
    if '/' in page_path:
        # Page is in a subdirectory, e.g., schema_dir/LoadBalancerConfig/Rule.rst
        # The directory is schema_dir/LoadBalancerConfig/
        page_dir = page_path.rsplit('/', 1)[0]  # "LoadBalancerConfig"
        # Child path might be "LoadBalancerConfig/Rule/MethodSpec"
        if child_path.startswith(page_dir + '/'):
            return child_path[len(page_dir) + 1:]
    # Page is at top level, e.g., schema_dir/LoadBalancerConfig.rst
    # child_path like "LoadBalancerConfig/Backend" is already correct
    return child_path


def _register_all_types(schema: Schema, schema_name: str, ctx: DocContext):
    """Register all types from a schema (including nested) in the context."""
    
    def _register_struct(struct: Struct, ctx: DocContext):
        qname = struct.qualified_name
        doc_path = _doc_path_for_type(qname)
        ctx.all_type_names.add(qname)
        ctx.type_to_schema[qname] = schema_name
        ctx.type_to_doc_path[qname] = (schema_name, doc_path)
        # Also register by short name (field types use short names like "Backend")
        if struct.name not in ctx.all_type_names:
            ctx.all_type_names.add(struct.name)
            ctx.type_to_schema[struct.name] = schema_name
            ctx.type_to_doc_path[struct.name] = (schema_name, doc_path)
        for ns in struct.nested_structs:
            _register_struct(ns, ctx)
        for ne in struct.nested_enums:
            _register_enum(ne, ctx)
    
    def _register_enum(enum: Enum, ctx: DocContext):
        qname = enum.qualified_name
        doc_path = _doc_path_for_type(qname)
        ctx.all_type_names.add(qname)
        ctx.type_to_schema[qname] = schema_name
        ctx.type_to_doc_path[qname] = (schema_name, doc_path)
        if enum.name not in ctx.all_type_names:
            ctx.all_type_names.add(enum.name)
            ctx.type_to_schema[enum.name] = schema_name
            ctx.type_to_doc_path[enum.name] = (schema_name, doc_path)
    
    for s in schema.structs:
        _register_struct(s, ctx)
    
    for i in schema.interfaces:
        qname = i.qualified_name
        doc_path = _doc_path_for_type(qname)
        ctx.all_type_names.add(qname)
        ctx.type_to_schema[qname] = schema_name
        ctx.type_to_doc_path[qname] = (schema_name, doc_path)
        if i.name not in ctx.all_type_names:
            ctx.all_type_names.add(i.name)
            ctx.type_to_schema[i.name] = schema_name
            ctx.type_to_doc_path[i.name] = (schema_name, doc_path)
        for ns in i.nested_structs:
            _register_struct(ns, ctx)
        for ne in i.nested_enums:
            _register_enum(ne, ctx)
    
    for e in schema.enums:
        _register_enum(e, ctx)


# ---- Page generators ----

def generate_struct_page(struct: Struct, schema_name: str, ctx: DocContext) -> str:
    """Generate an RST page for a single struct.
    
    Uses the struct's qualified_name as the title.
    Includes toctree entries for any nested structs/enums.
    """
    lines = []
    
    # Title: fully qualified name
    qname = struct.qualified_name
    lines.append(qname)
    lines.append("=" * len(qname))
    lines.append("")
    
    # Schema source info - use absolute :doc: path
    lines.append(f".. rst-class:: schema-source")
    lines.append("")
    lines.append(f"Schema: :doc:`/_capnp/{schema_name}/index`")
    if struct.parent:
        parent_ref = _doc_ref_abs(struct.parent, ctx)
        lines.append(f"Parent: :doc:`{parent_ref}`")
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
        
        # Regular fields
        for f in struct.fields:
            type_display = format_type(f.type_, ctx)
            default_str = f" = {f.default}" if f.default else ""
            #lines.append(f"**{f.name}**")
            #lines.append(f"   {type_display}{default_str}")
            lines.append(f"- {f.name} - {type_display}{default_str}")
            if f.comment:
                lines.append("")
                lines.extend(render_comment(f.comment, "  "))
            lines.append("")
        
        # Union groups
        for ug in struct.unions:
            lines.extend(_render_union_group(ug, ctx, indent=0))
    
    # Nested structs - toctree uses paths relative to this page's directory
    if struct.nested_structs:
        lines.append("")
        lines.append("Nested Structs")
        lines.append("--------------")
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 1")
        lines.append("")
        for ns in struct.nested_structs:
            ref = _toctree_ref_from_page(ns.qualified_name, qname)
            lines.append(f"   {ref}")
        lines.append("")
    
    # Nested enums
    if struct.nested_enums:
        lines.append("")
        lines.append("Nested Enums")
        lines.append("------------")
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 1")
        lines.append("")
        for ne in struct.nested_enums:
            ref = _toctree_ref_from_page(ne.qualified_name, qname)
            lines.append(f"   {ref}")
        lines.append("")
    
    lines.append("")
    return "\n".join(lines)


def _render_union_group(ug: UnionGroup, ctx: DocContext, indent: int = 0) -> list:
    """Render a union or group recursively as RST definition list."""
    lines = []
    prefix = "  " * indent
    childPrefix = "  " * (indent + 1)
    kind = "union" if ug.kind == "union" else "group"
    label = ug.name if ug.name else f"anonymous {kind}"
    
    #lines.append(f"{prefix}**{label}** ({kind})")
    lines.append(f"{prefix}- {label} - {kind}")
    lines.append("")
    
    for f in ug.fields:
        type_display = format_type(f.type_, ctx)
        default_str = f" = {f.default}" if f.default else ""
        #lines.append(f"{childPrefix}**{f.name}**")
        #lines.append(f"{childPrefix}  {type_display}{default_str}")
        lines.append(f"{childPrefix}- {f.name} - {type_display}{default_str}")
        if f.comment:
            lines.append("")
            lines.extend(render_comment(f.comment, f"{childPrefix}  "))
        lines.append("")
    
    for nested in ug.nested_groups:
        lines.extend(_render_union_group(nested, ctx, indent=indent + 1))
    
    return lines


def generate_interface_page(interface: Interface, schema_name: str,
                            ctx: DocContext) -> str:
    """Generate an RST page for a single interface."""
    lines = []
    
    # Title: fully qualified name
    qname = interface.qualified_name
    lines.append(qname)
    lines.append("=" * len(qname))
    lines.append("")
    
    # Schema source info - use absolute :doc: path
    lines.append(f".. rst-class:: schema-source")
    lines.append("")
    lines.append(f"Schema: :doc:`/_capnp/{schema_name}/index`")
    if interface.extends:
        ext_refs = [format_type(e, ctx) for e in interface.extends]
        lines.append(f"Extends: {', '.join(ext_refs)}")
    if interface.parent:
        parent_ref = _doc_ref_abs(interface.parent, ctx)
        lines.append(f"Parent: :doc:`{parent_ref}`")
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
            # Format parameters
            if m.parameters:
                param_parts = []
                for p in m.parameters:
                    if p.is_struct_ref:
                        # Bare struct reference: show as link to the struct
                        param_parts.append(format_type(p.type_, ctx))
                    else:
                        # Inline field: name: Type
                        param_parts.append(f"{p.name}: {format_type(p.type_, ctx)}")
                params_str = ", ".join(param_parts)
            else:
                params_str = ""
            
            # Format return type
            if m.return_fields:
                ret_parts = []
                for rf in m.return_fields:
                    if rf.is_struct_ref:
                        # Bare struct reference: show as link to the struct
                        ret_parts.append(format_type(rf.type_, ctx))
                    else:
                        # Inline field: name : Type
                        ret_parts.append(f"{rf.name} : {format_type(rf.type_, ctx)}")
                if len(ret_parts) == 1 and m.return_fields[0].is_struct_ref:
                    # Single struct ref: no parens needed
                    ret_str = ret_parts[0]
                else:
                    # Multiple fields or inline: use parens
                    ret_str = "(" + ", ".join(ret_parts) + ")"
            elif m.return_type and m.return_type != "()":
                # Fallback to raw return_type string
                ret_str = format_type(m.return_type, ctx)
            else:
                ret_str = "()"
            
            lines.append(f".. rst-class:: capnp-method")
            lines.append("")
            lines.append(f"{m.name}: ({params_str}) -> {ret_str}")
            lines.append("")
            if m.comment:
                lines.extend(render_comment(m.comment, "   "))
                lines.append("")
    
    # Nested structs
    if interface.nested_structs:
        lines.append("")
        lines.append("Nested Structs")
        lines.append("--------------")
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 1")
        lines.append("")
        for ns in interface.nested_structs:
            ref = _toctree_ref_from_page(ns.qualified_name, qname)
            lines.append(f"   {ref}")
        lines.append("")
    
    # Nested enums
    if interface.nested_enums:
        lines.append("")
        lines.append("Nested Enums")
        lines.append("------------")
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 1")
        lines.append("")
        for ne in interface.nested_enums:
            ref = _toctree_ref_from_page(ne.qualified_name, qname)
            lines.append(f"   {ref}")
        lines.append("")
    
    lines.append("")
    return "\n".join(lines)


def generate_enum_page(enum: Enum, schema_name: str, ctx: DocContext) -> str:
    """Generate an RST page for a single enum."""
    lines = []
    
    # Title: fully qualified name
    qname = enum.qualified_name
    lines.append(qname)
    lines.append("=" * len(qname))
    lines.append("")
    
    # Schema source info - use absolute :doc: path
    lines.append(f".. rst-class:: schema-source")
    lines.append("")
    lines.append(f"Schema: :doc:`/_capnp/{schema_name}/index`")
    if enum.parent:
        parent_ref = _doc_ref_abs(enum.parent, ctx)
        lines.append(f"Parent: :doc:`{parent_ref}`")
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
    """Generate an index page for all top-level types in a schema.
    
    Only lists top-level types (no parent). Nested types are linked
    from their parent's page via toctree.
    """
    lines = []
    
    title = f"{schema_name}.capnp"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    
    if schema.file_comment:
        lines.extend(render_comment(schema.file_comment))
        lines.append("")
    
    # Top-level structs only (no parent)
    top_structs = [s for s in schema.structs if not s.parent]
    if top_structs:
        lines.append("Structs")
        lines.append("-------")
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 2")
        lines.append("")
        for s in top_structs:
            lines.append(f"   {s.name}")
        lines.append("")
    
    # Top-level interfaces only (no parent)
    top_ifaces = [i for i in schema.interfaces if not i.parent]
    if top_ifaces:
        lines.append("Interfaces")
        lines.append("----------")
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 2")
        lines.append("")
        for i in top_ifaces:
            lines.append(f"   {i.name}")
        lines.append("")
    
    # Top-level enums only (no parent)
    top_enums = [e for e in schema.enums if not e.parent]
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
    lines.append("   :maxdepth: 3")
    lines.append("")
    for name in sorted(all_schema_names):
        lines.append(f"   {name}/index")
    lines.append("")
    
    return "\n".join(lines)


def _write_type_pages(type_obj, schema_dir: Path, schema_name: str, ctx: DocContext):
    """Recursively write RST pages for a type and all its nested types."""
    if isinstance(type_obj, Struct):
        qname = type_obj.qualified_name
        doc_path = _doc_path_for_type(qname)
        
        # Write the page
        page_path = schema_dir / f"{doc_path}.rst"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page = generate_struct_page(type_obj, schema_name, ctx)
        page_path.write_text(page)
        print(f"Generated: {page_path}")
        
        # Recursively write nested types
        for ns in type_obj.nested_structs:
            _write_type_pages(ns, schema_dir, schema_name, ctx)
        for ne in type_obj.nested_enums:
            _write_type_pages(ne, schema_dir, schema_name, ctx)
    
    elif isinstance(type_obj, Interface):
        qname = type_obj.qualified_name
        doc_path = _doc_path_for_type(qname)
        
        page_path = schema_dir / f"{doc_path}.rst"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page = generate_interface_page(type_obj, schema_name, ctx)
        page_path.write_text(page)
        print(f"Generated: {page_path}")
        
        for ns in type_obj.nested_structs:
            _write_type_pages(ns, schema_dir, schema_name, ctx)
        for ne in type_obj.nested_enums:
            _write_type_pages(ne, schema_dir, schema_name, ctx)
    
    elif isinstance(type_obj, Enum):
        qname = type_obj.qualified_name
        doc_path = _doc_path_for_type(qname)
        
        page_path = schema_dir / f"{doc_path}.rst"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page = generate_enum_page(type_obj, schema_name, ctx)
        page_path.write_text(page)
        print(f"Generated: {page_path}")


def generate_complete_docs(input_files: List[str], output_path: Path) -> None:
    """Generate complete documentation from multiple schema files.
    
    Each schema gets its own subdirectory with per-type RST pages.
    Nested types are placed in subdirectories matching their qualified name path.
    """
    all_schemas = []  # (schema_name, Schema)
    ctx = DocContext()
    
    # First pass: parse all schemas and register type names
    for input_file in input_files:
        schema = parse_schema_file(input_file)
        schema_name = Path(input_file).stem
        all_schemas.append((schema_name, schema))
        _register_all_types(schema, schema_name, ctx)
    
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
        
        # Write pages for top-level structs (and their nested types recursively)
        for struct in schema.structs:
            _write_type_pages(struct, schema_dir, schema_name, ctx)
        
        # Write pages for top-level interfaces (and their nested types recursively)
        for interface in schema.interfaces:
            _write_type_pages(interface, schema_dir, schema_name, ctx)
        
        # Write pages for top-level enums
        for enum in schema.enums:
            _write_type_pages(enum, schema_dir, schema_name, ctx)


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
