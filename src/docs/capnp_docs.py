#!/usr/bin/env python3
"""
Cap'n'Proto Documentation Generator

This module generates reStructuredText documentation from Cap'n'Proto schema files.
It creates documentation for both C++ and Python API usage.

Usage:
    python capnp_docs.py --input path/to/schema.capnp --output docs/output.rst
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

from capnp_parser import parse_schema, parse_schema_file, Schema, Struct, Interface, Enum, Field, Method


def escape_rst(text: str) -> str:
    """Escape special characters in reStructuredText."""
    if not text:
        return ""
    # Escape backticks
    text = text.replace('`', '\\`')
    return text


def create_qualified_name_map(structs: List[Struct]) -> Dict[str, str]:
    """
    Create a mapping from struct names to their fully qualified names.
    For nested structs, this includes parent names (e.g., 'Section' -> 'ReversibleFieldlineMapping.Section')
    """
    name_map = {}
    
    for struct in structs:
        qualified = build_qualified_name(struct, structs)
        name_map[struct.name] = qualified
    
    return name_map


def format_type(type_str: str, qualified_name_map: Dict[str, str] = None) -> str:
    """Format a Cap'n'Proto type for documentation."""
    # Replace common Cap'n'Proto types with their Python equivalents
    replacements = {
        'Data': 'bytes',
        'Text': 'str',
        'Bool': 'bool',
        'Int8': 'int',
        'Int16': 'int',
        'Int32': 'int',
        'Int64': 'int',
        'UInt8': 'int',
        'UInt16': 'int',
        'UInt32': 'int',
        'UInt64': 'int',
        'Float32': 'float',
        'Float64': 'float',
        'Void': 'None',
    }
    
    result = type_str
    for capnp_type, py_type in replacements.items():
        # Use word boundaries to avoid partial matches
        result = re.sub(r'\\b' + capnp_type + r'\\b', py_type, result)
    
    # Replace struct names with their qualified versions if available
    if qualified_name_map:
        # Sort by length (longest first) to avoid partial replacement issues
        # But only replace if not preceded by a dot (already part of qualified name)
        for struct_name, qualified_name in sorted(qualified_name_map.items(), key=lambda x: -len(x[0])):
            # Match word boundary but NOT preceded by dot (using fixed-width negative lookbehind)
            pattern = r'(?<!\.)\b' + re.escape(struct_name) + r'\b'
            result = re.sub(pattern, qualified_name, result)
    
    return result


def generate_struct_doc(schema: Schema, struct: Struct) -> str:
    """Generate documentation for a struct."""
    lines = []
    
    # Header
    lines.append(f"``{struct.name}``")
    lines.append("-" * (len(struct.name) + 4))
    lines.append("")
    
    # Comment if present
    if struct.comment:
        lines.append(struct.comment)
        lines.append("")
    
    # Fields
    if struct.fields:
        lines.append("**Fields:**")
        lines.append("")
        for field in struct.fields:
            lines.append(f"- ``{field.name}`` (:class:`{escape_rst(field.type_)}`)")
        lines.append("")
    
    # Union fields if any
    if struct.union_fields:
        lines.append("**Union Fields:**")
        lines.append("")
        for field in struct.union_fields:
            lines.append(f"- ``{field.name}`` ({field.type_}): {len(field.fields)} nested field(s)")
        lines.append("")
    
    return '\n'.join(lines)


def generate_interface_doc(schema: Schema, interface: Interface) -> str:
    """Generate documentation for an interface."""
    lines = []
    
    # Header
    lines.append(f"``{interface.name}``")
    lines.append("-" * (len(interface.name) + 4))
    lines.append("")
    
    # Comment if present
    if interface.comment:
        lines.append(interface.comment)
        lines.append("")
    
    # Methods
    if interface.methods:
        lines.append("**Methods:**")
        lines.append("")
        for method in interface.methods:
            lines.append(f"- ``{method.name}({', '.join(p.name for p in method.parameters)})``")
            if method.return_type != "()":
                lines.append(f"  - Returns: :class:`{escape_rst(method.return_type)}`")
            if method.parameters:
                lines.append("  - Parameters:")
                for param in method.parameters:
                    lines.append(f"    - ``{param.name}`` (:class:`{escape_rst(param.type_)}`)")
            lines.append("")
    
    return '\n'.join(lines)


def generate_enum_doc(schema: Schema, enum: Enum) -> str:
    """Generate documentation for an enum."""
    lines = []
    
    # Header
    lines.append(f"``{enum.name}``")
    lines.append("-" * (len(enum.name) + 4))
    lines.append("")
    
    # Comment if present
    if enum.comment:
        lines.append(enum.comment)
        lines.append("")
    
    # Values
    if enum.values:
        lines.append("**Values:**")
        lines.append("")
        for value in enum.values:
            lines.append(f"- ``{value}``")
        lines.append("")
    
    return '\n'.join(lines)


def get_nested_structs_by_ancestors(structs: List[Struct]) -> Dict[str, Dict]:
    """
    Build a tree structure of nested structs based on parent relationships.
    Returns a dict mapping parent name to nested struct info including their own children.
    """
    nested = {}
    
    # First pass: group by direct parent
    for struct in structs:
        if struct.parent:
            if struct.parent not in nested:
                nested[struct.parent] = {}
            nested[struct.parent][struct.name] = struct
    
    # Second pass: attach children to their respective parents recursively
    def build_tree(parent_name: str, parent_structs: Dict, all_structs: List[Struct]) -> Dict:
        """Recursively build the nested tree structure."""
        result = {}
        for name, struct in parent_structs.items():
            # Find children of this struct
            children = {}
            for s in all_structs:
                if s.parent == name:
                    children[s.name] = s
            
            result[name] = {
                'struct': struct,
                'children': build_tree(name, children, all_structs) if children else {}
            }
        return result
    
    # Build the tree for each top-level parent
    final_tree = {}
    for parent_name, parent_structs in nested.items():
        final_tree[parent_name] = build_tree(parent_name, parent_structs, structs)
    
    return final_tree


def build_qualified_name(struct: Struct, structs: List[Struct]) -> str:
    """Build the qualified name for a struct including all ancestor names."""
    parts = [struct.name]
    current = struct
    visited = set()
    while current.parent and current.name not in visited:
        visited.add(current.name)
        parts.insert(0, current.parent)
        # Find the parent struct
        for s in structs:
            if s.name == current.parent:
                current = s
                break
        else:
            break
    return ".".join(parts)


def generate_struct_with_nested_doc(schema: Schema, struct: Struct, 
                                    nesting_level: int = 0) -> str:
    """Generate documentation for a struct with nested children, using RST section levels."""
    lines = []
    
    # Build qualified name
    qualified_name = build_qualified_name(struct, schema.structs)
    
    # Build qualified name map for all structs
    name_map = create_qualified_name_map(schema.structs)
    
    # RST header characters based on nesting level
    # Level 0: =, Level 1: -, Level 2: ^, Level 3: ", Level 4: '
    header_chars = ['=', '-', '^', '"', "'"]
    
    # Determine which header char to use
    if nesting_level < len(header_chars):
        header_char = header_chars[nesting_level]
    else:
        header_char = header_chars[-1]
    
    # Header
    lines.append(qualified_name)
    lines.append(header_char * len(qualified_name))
    lines.append("")
    
    # Add comment if present
    if struct.comment:
        lines.append(struct.comment)
        lines.append("")
    
    # Fields section
    if struct.fields:
        lines.append("**Fields:**")
        lines.append("")
        for field in struct.fields:
            default = f" = {field.default}" if field.default else ""
            lines.append(f"- ``{field.name}`` (:class:`{format_type(field.type_, name_map)}`){default}")
        lines.append("")

    # Union and Group fields if any
    if struct.union_fields:
        lines.append("**Union and Group Fields:**")
        lines.append("")
        for field in struct.union_fields:
            lines.append(f"- ``{field.name}`` ({field.type_}): {len(field.fields)} nested field(s)")
        lines.append("")

    # Process nested structs (children)
    nested_tree = get_nested_structs_by_ancestors(schema.structs)
    if struct.name in nested_tree:
        for child_name, child_info in nested_tree[struct.name].items():
            child_struct = child_info['struct']
            # Append the entire doc string, not individual characters
            child_doc = generate_struct_with_nested_doc(schema, child_struct, 
                                                        nesting_level + 1)
            lines.append(child_doc)
    
    return '\n'.join(lines)




def generate_schema_doc(schema: Schema, namespace: str = "fsc") -> str:
    """Generate complete documentation for a schema."""
    lines = []
    
    # Title
    lines.append("=" * len(schema.filename))
    lines.append(schema.filename)
    lines.append("=" * len(schema.filename))
    lines.append("")
    
    # File comment if present
    if schema.file_comment:
        lines.append(schema.file_comment)
        lines.append("")
    
    # Namespace
    lines.append(f"Namespace: ``{schema.namespace or namespace}``")
    lines.append("")
    
    # Schema header
    lines.append("Schema Information")
    lines.append("-" * 18)
    lines.append("")
    
    if schema.imports:
        lines.append("Imports:")
        for imp in schema.imports:
            lines.append(f"- {imp}")
        lines.append("")
    
    # Structs section
    if schema.structs:
        # Find top-level structs (those without parents)
        nested_tree = get_nested_structs_by_ancestors(schema.structs)
        for struct in schema.structs:
            if not struct.parent:
                lines.append(generate_struct_with_nested_doc(schema, struct))
    
    # Interfaces section
    if schema.interfaces:
        lines.append("Interfaces")
        lines.append("-" * 10)
        lines.append("")
        for interface in schema.interfaces:
            lines.append(generate_interface_doc(schema, interface))
    
    # Enums section
    if schema.enums:
        lines.append("Enums")
        lines.append("-" * 5)
        lines.append("")
        for enum in schema.enums:
            lines.append(generate_enum_doc(schema, enum))
    
    return '\n'.join(lines)


def generate_api_examples(schema: Schema, lang: str = "cpp") -> str:
    """Generate API usage examples in the specified language."""
    if lang == "cpp":
        return generate_cpp_examples(schema)
    elif lang == "python":
        return generate_python_examples(schema)
    else:
        raise ValueError(f"Unsupported language: {lang}")


def generate_cpp_examples(schema: Schema) -> str:
    """Generate C++ API usage examples."""
    lines = []
    
    lines.append("C++ API Usage Examples")
    lines.append("=" * 22)
    lines.append("")
    
    # Namespace mention
    if schema.namespace:
        lines.append(f"All types are in the ``{schema.namespace}`` namespace.")
        lines.append("")
    
    # Struct examples
    if schema.structs:
        lines.append("Structs")
        lines.append("-" * 7)
        lines.append("")
        
        for struct in schema.structs[:3]:  # Limit to first 3 for brevity
            lines.append(f"Creating ``{struct.name}``:")
            lines.append("")
            lines.append(f".. code-block:: c++")
            lines.append("")
            lines.append(f"    {schema.namespace}::{struct.name} message;")
            if struct.fields:
                lines.append(f"    message.set_{struct.fields[0].name}(...);")
            lines.append("")
    
    # Interface examples
    if schema.interfaces:
        lines.append("Interfaces")
        lines.append("-" * 10)
        lines.append("")
        
        for interface in schema.interfaces[:2]:  # Limit to first 2
            lines.append(f"Using ``{interface.name}``:")
            lines.append("")
            lines.append(f".. code-block:: c++")
            lines.append("")
            lines.append(f"    // Get interface reference")
            lines.append(f"    auto interface = connection.get<{schema.namespace}::{interface.name}>();")
            lines.append("")
            
            if interface.methods:
                method = interface.methods[0]
                lines.append(f"    // Call method: {method.name}")
                params = ", ".join(f"{p.type_} {p.name}" for p in method.parameters)
                lines.append(f"    auto result = interface.{method.name}({params}).wait();")
            lines.append("")
    
    return '\n'.join(lines)


def generate_python_examples(schema: Schema) -> str:
    """Generate Python API usage examples."""
    lines = []
    
    lines.append("Python API Usage Examples")
    lines.append("=" * 25)
    lines.append("")
    
    # Namespace mention
    if schema.namespace:
        lines.append(f"All types are in the ``{schema.namespace}`` module.")
        lines.append("")
    
    # Struct examples
    if schema.structs:
        lines.append("Structs")
        lines.append("-" * 7)
        lines.append("")
        
        for struct in schema.structs[:3]:  # Limit to first 3 for brevity
            lines.append(f"Creating ``{struct.name}``:")
            lines.append("")
            lines.append(f".. code-block:: python")
            lines.append("")
            lines.append(f"    from fsc.capnp import {struct.name}")
            lines.append("")
            lines.append(f"    message = {struct.name}()")
            if struct.fields:
                field = struct.fields[0]
                lines.append(f"    message.{field.name} = ...  # {format_type(field.type_)}")
            lines.append("")
    
    # Interface examples
    if schema.interfaces:
        lines.append("Interfaces")
        lines.append("-" * 10)
        lines.append("")
        
        for interface in schema.interfaces[:2]:  # Limit to first 2
            lines.append(f"Using ``{interface.name}``:")
            lines.append("")
            lines.append(f".. code-block:: python")
            lines.append("")
            lines.append(f"    import capnp")
            lines.append(f"    from fsc.capnp import {interface.name}")
            lines.append("")
            lines.append(f"    # Connect to service")
            lines.append(f"    connection = capnp.TwoPartyClient('localhost:12345', bootstrap='...')")
            lines.append(f"    service = connection.root.as_capability({interface.name})")
            lines.append("")
            
            if interface.methods:
                method = interface.methods[0]
                lines.append(f"    # Call method: {method.name}")
                params = ", ".join(f"{p.name}=..." for p in method.parameters)
                lines.append(f"    result = await service.{method.name}({params})")
            lines.append("")
    
    return '\n'.join(lines)


def generate_complete_docs(input_files: List[str], output_dir: str, 
                           namespace: str = "fsc") -> str:
    """Generate complete documentation for multiple schema files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each schema file - store tuples of (schema_name, schema_doc)
    all_schema_docs = []
    all_api_examples = []
    
    for input_file in input_files:
        schema = parse_schema_file(input_file)
        schema_name = Path(input_file).stem
        
        # Generate schema documentation
        schema_doc = generate_schema_doc(schema, namespace)
        all_schema_docs.append((schema_name, schema_doc))
        
        # Save schema documentation
        output_file = output_path / f"{schema_name}.rst"
        output_file.write_text(schema_doc)
        print(f"Generated: {output_file}")
        
        # Generate API examples
        cpp_examples = generate_api_examples(schema, "cpp")
        python_examples = generate_api_examples(schema, "python")
        
        # Save examples
        (output_path / f"{schema_name}_cpp.rst").write_text(cpp_examples)
        (output_path / f"{schema_name}_python.rst").write_text(python_examples)
    
    # Create index file
    index_content = generate_index(all_schema_docs, namespace)
    (output_path / "index.rst").write_text(index_content)
    print(f"Generated: {output_path / 'index.rst'}")
    
    return output_path


def generate_index(schema_docs: List[tuple], namespace: str = "fsc") -> str:
    """Generate an index RST file linking all documentation.
    
    Args:
        schema_docs: List of tuples (schema_name, schema_doc_string)
    """
    lines = []

    lines.append("Cap'n'Proto API Documentation")
    lines.append("=" * 28)
    lines.append("")

    lines.append("Overview")
    lines.append("-" * 8)
    lines.append("")
    lines.append(f"This documentation covers the Cap'n'Proto schemas used in the ``{namespace}``")
    lines.append("project, including data structures and RPC interfaces.")
    lines.append("")

    lines.append("Contents")
    lines.append("-" * 8)
    lines.append("")
    lines.append(".. toctree::")
    lines.append("   :maxdepth: 2")
    lines.append("")

    # Add TOC entries - use schema filenames without extension
    for schema_name, _ in schema_docs:
        lines.append(f"   {schema_name}")
    
    lines.append("")
    lines.append("API Examples")
    lines.append("-" * 12)
    lines.append("")
    lines.append("See the following sections for usage examples in different languages:")
    lines.append("")
    lines.append(".. toctree::")
    lines.append("   :maxdepth: 1")
    lines.append("")
    lines.append("   C++ Examples <cpp_examples>")
    lines.append("   Python Examples <python_examples>")
    lines.append("")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate documentation from Cap'n'Proto schema files"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input schema files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for generated documentation"
    )
    parser.add_argument(
        "--namespace", "-n",
        default="fsc",
        help="Namespace for the generated documentation (default: fsc)"
    )
    
    args = parser.parse_args()
    
    output_dir = generate_complete_docs(args.input, args.output, args.namespace)
    print(f"\nDocumentation generated in: {output_dir}")


if __name__ == "__main__":
    main()
