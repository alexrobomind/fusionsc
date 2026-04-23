#!/usr/bin/env python3
"""
Cap'n'Proto Schema Parser

This module provides a parser for Cap'n'Proto schema files that extracts
semantic information for documentation generation.

Usage:
    from capnp_parser import parse_schema
    
    schema = parse_schema("path/to/schema.capnp")
    print(f"Namespace: {schema.namespace}")
    for struct in schema.structs:
        print(f"  Struct: {struct.name}")
    for interface in schema.interfaces:
        print(f"  Interface: {interface.name}")
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class Field:
    """Represents a struct field or method parameter."""
    name: str
    index: int
    type_: str
    default: Optional[str] = None
    annotations: Dict[str, str] = field(default_factory=dict)
    comment: Optional[str] = None


@dataclass
class Method:
    """Represents an interface method."""
    name: str
    index: int
    parameters: List[Field] = field(default_factory=list)
    return_type: str = "()"
    comment: Optional[str] = None


@dataclass
class Struct:
    """Represents a Cap'n'Proto struct."""
    name: str
    namespace: str
    fields: List[Field] = field(default_factory=list)
    comment: Optional[str] = None
    union_fields: List[Field] = field(default_factory=list)


@dataclass
class Interface:
    """Represents a Cap'n'Proto interface."""
    name: str
    namespace: str
    methods: List[Method] = field(default_factory=list)
    comment: Optional[str] = None


@dataclass
class Enum:
    """Represents a Cap'n'Proto enum."""
    name: str
    namespace: str
    values: List[str] = field(default_factory=list)
    comment: Optional[str] = None


@dataclass 
class Schema:
    """Represents a parsed Cap'n'Proto schema file."""
    filename: str
    namespace: str = ""
    structs: List[Struct] = field(default_factory=list)
    interfaces: List[Interface] = field(default_factory=list)
    enums: List[Enum] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    file_comment: Optional[str] = None


def remove_comments(text: str) -> str:
    """Remove C++ style comments from text."""
    # Remove single-line comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def extract_comment(lines: List[str], start_idx: int) -> Optional[str]:
    """Extract comment block before a definition."""
    comment_lines = []
    for i in range(start_idx - 1, max(-1, start_idx - 20), -1):
        line = lines[i].strip()
        if line.startswith('#//!'):
            comment_lines.insert(0, line[4:].strip())
        elif line.startswith('#'):
            # Skip comment marker for non-documentation comments
            if not line.startswith('#//!') and not line.startswith('# '):
                break
            comment_lines.insert(0, line[1:].strip())
        elif line == '' and comment_lines:
            break
        elif line and not line.startswith('#'):
            break
    
    if comment_lines:
        return '\n'.join(comment_lines)
    return None


def parse_schema(content: str, filename: str = "<string>") -> Schema:
    """
    Parse a Cap'n'Proto schema string.
    
    Args:
        content: The schema file content
        filename: Original filename for reference
        
    Returns:
        Schema object with parsed elements
    """
    # Remove comments for easier parsing
    clean_content = remove_comments(content)
    lines = content.split('\n')  # Keep original for comment extraction
    
    schema = Schema(filename=filename)
    
    # Extract file-level comment
    file_comment_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#//!'):
            file_comment_lines.append(stripped[4:].strip())
        elif stripped.startswith('#') and not stripped.startswith('#//!'):
            break
        elif stripped and not stripped.startswith('#'):
            break
    if file_comment_lines:
        schema.file_comment = '\n'.join(file_comment_lines)
    
    # Extract namespace
    namespace_match = re.search(r'\$Cxx\.namespace\("([^"]+)"\)', content)
    if namespace_match:
        schema.namespace = namespace_match.group(1)
    
    # Extract imports
    import_matches = re.findall(r'using\s+(\w+)\s*=\s*import\s+"([^"]+)"', content)
    schema.imports = [f"{name}: {path}" for name, path in import_matches]
    
    # Parse structs
    struct_pattern = r'struct\s+(\w+)'
    for match in re.finditer(struct_pattern, clean_content):
        name = match.group(1)
        start_pos = match.start()
        
        # Find the struct body
        brace_count = 0
        body_start = -1
        body_end = -1
        
        for i, char in enumerate(clean_content[start_pos:], start_pos):
            if char == '{':
                if brace_count == 0:
                    body_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    body_end = i
                    break
        
        if body_start >= 0 and body_end >= 0:
            body = clean_content[body_start+1:body_end]
            
            # Extract comment from original lines
            start_idx = len(content[:start_pos].split('\n')) - 1
            comment = extract_comment(lines, start_idx)
            
            struct = Struct(name=name, namespace=schema.namespace, comment=comment)
            
            # Parse struct fields
            field_pattern = r'(\w+)\s+@(\d+)\s*:\s*([^;]+);'
            for field_match in re.finditer(field_pattern, body):
                field_name = field_match.group(1)
                field_idx = int(field_match.group(2))
                field_type = field_match.group(3).strip()
                
                # Check for union
                if 'union' in field_type:
                    # Handle union as a special case
                    union_body_match = re.search(r'union\s*{([^}]+)}', field_type)
                    if union_body_match:
                        union_body = union_body_match.group(1)
                        union_pattern = r'(\w+)\s+@(\d+)\s*:\s*([^;]+);'
                        for union_match in re.finditer(union_pattern, union_body):
                            union_field = Field(
                                name=union_match.group(1),
                                index=int(union_match.group(2)),
                                type_=union_match.group(3).strip()
                            )
                            struct.union_fields.append(union_field)
                else:
                    field = Field(
                        name=field_name,
                        index=field_idx,
                        type_=field_type
                    )
                    struct.fields.append(field)
            
            schema.structs.append(struct)
    
    # Parse interfaces
    interface_pattern = r'interface\s+(\w+)'
    for match in re.finditer(interface_pattern, clean_content):
        name = match.group(1)
        start_pos = match.start()
        
        # Find the interface body
        brace_count = 0
        body_start = -1
        body_end = -1
        
        for i, char in enumerate(clean_content[start_pos:], start_pos):
            if char == '{':
                if brace_count == 0:
                    body_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    body_end = i
                    break
        
        if body_start >= 0 and body_end >= 0:
            body = clean_content[body_start+1:body_end]
            
            # Extract comment from original lines
            start_idx = len(content[:start_pos].split('\n')) - 1
            comment = extract_comment(lines, start_idx)
            
            interface = Interface(name=name, namespace=schema.namespace, comment=comment)
            
            # Parse interface methods
            # Cap'n'Proto syntax: name @index [T] (params) -> return;
            # We need to handle nested parentheses in both params and return types
            # e.g., clone @1 [T] (source : DataRef(T)) -> (ref : DataRef(T));
            for line in body.split('\n'):
                stripped = line.strip()
                
                # Skip empty lines, comments, and nested interface declarations
                if not stripped or stripped.startswith('#') or stripped.startswith('interface '):
                    continue
                
                # Try to match method pattern
                arrow_pos = stripped.find('->')
                if arrow_pos < 0:
                    continue
                
                before_arrow = stripped[:arrow_pos].strip()
                after_arrow = stripped[arrow_pos+2:].strip().rstrip(';').strip()
                
                # Check if this looks like a method (has @index)
                at_match = re.search(r'@\s*(\d+)', before_arrow)
                if not at_match:
                    continue
                
                # Extract method name
                name_match = re.search(r'^(\w+)\s*$', before_arrow[:at_match.start()].strip())
                if not name_match:
                    continue
                
                method_name = name_match.group(1)
                method_idx = int(at_match.group(1))
                
                # Extract parameters - find the opening paren and match balanced parens
                paren_start = re.search(r'\(', before_arrow)
                if paren_start:
                    paren_start_pos = paren_start.start()
                    paren_count = 1
                    paren_end_pos = paren_start_pos + 1
                    while paren_end_pos < len(before_arrow) and paren_count > 0:
                        if before_arrow[paren_end_pos] == '(':
                            paren_count += 1
                        elif before_arrow[paren_end_pos] == ')':
                            paren_count -= 1
                        paren_end_pos += 1
                    
                    if paren_count == 0:
                        params = before_arrow[paren_start_pos+1:paren_end_pos-1]
                    else:
                        continue
                else:
                    params = ""
                
                method = Method(
                    name=method_name,
                    index=method_idx,
                    return_type=after_arrow
                )
                
                # Parse parameters
                if params.strip():
                    param_pattern = r'(\w+)\s*:\s*([^,)]+)'
                    for param_match in re.finditer(param_pattern, params):
                        param_name = param_match.group(1)
                        param_type = param_match.group(2).strip()
                        method.parameters.append(Field(
                            name=param_name,
                            index=0,
                            type_=param_type
                        ))
                
                interface.methods.append(method)
            
            schema.interfaces.append(interface)
    
    # Parse enums
    enum_pattern = r'enum\s+(\w+)'
    for match in re.finditer(enum_pattern, clean_content):
        name = match.group(1)
        start_pos = match.start()
        
        # Find the enum body
        brace_count = 0
        body_start = -1
        body_end = -1
        
        for i, char in enumerate(clean_content[start_pos:], start_pos):
            if char == '{':
                if brace_count == 0:
                    body_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    body_end = i
                    break
        
        if body_start >= 0 and body_end >= 0:
            body = clean_content[body_start+1:body_end]
            
            # Extract values
            values = []
            value_pattern = r'(\w+)\s+@(\d+)'
            for value_match in re.finditer(value_pattern, body):
                values.append(value_match.group(1))
            
            # Extract comment from original lines
            start_idx = len(content[:start_pos].split('\n')) - 1
            comment = extract_comment(lines, start_idx)
            
            enum = Enum(name=name, namespace=schema.namespace, values=values, comment=comment)
            schema.enums.append(enum)
    
    return schema


def parse_schema_file(filepath: str) -> Schema:
    """
    Parse a Cap'n'Proto schema file.
    
    Args:
        filepath: Path to the schema file
        
    Returns:
        Schema object with parsed elements
    """
    path = Path(filepath)
    content = path.read_text()
    return parse_schema(content, path.name)


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python capnp_parser.py <schema_file.capnp>")
        sys.exit(1)
    
    schema = parse_schema_file(sys.argv[1])
    
    # Output as JSON for debugging
    output = {
        "filename": schema.filename,
        "namespace": schema.namespace,
        "structs": [
            {
                "name": s.name,
                "fields": [{"name": f.name, "type": f.type_, "index": f.index} for f in s.fields],
                "union_fields": [{"name": f.name, "type": f.type_, "index": f.index} for f in s.union_fields]
            }
            for s in schema.structs
        ],
        "interfaces": [
            {
                "name": i.name,
                "methods": [
                    {
                        "name": m.name,
                        "parameters": [{"name": p.name, "type": p.type_} for p in m.parameters],
                        "return_type": m.return_type
                    }
                    for m in i.methods
                ]
            }
            for i in schema.interfaces
        ],
        "enums": [
            {
                "name": e.name,
                "values": e.values
            }
            for e in schema.enums
        ]
    }
    
    print(json.dumps(output, indent=2))
