#!/usr/bin/env python3
"""
Cap'n'Proto Schema Parser - FIXED VERSION

This module provides a parser for Cap'n'Proto schema files that extracts
semantic information for documentation generation.
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
class UnionField:
    """Represents a union or group field with its nested fields."""
    name: str
    type_: str  # 'union' or 'group'
    fields: List[Field] = field(default_factory=list)
    index: int = 0


@dataclass
class Struct:
    """Represents a Cap'n'Proto struct."""
    name: str
    namespace: str
    parent: Optional[str] = None  # Name of parent struct if nested
    fields: List[Field] = field(default_factory=list)
    comment: Optional[str] = None
    union_fields: List[UnionField] = field(default_factory=list)


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
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
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


def _parse_struct_body(struct: Struct, body: str):
    """Parse fields and union/group definitions from struct body."""
    lines = body.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            i += 1
            continue
        
        # Skip struct/interface/enum definitions (nested definitions, not fields)
        if stripped.startswith('struct ') or stripped.startswith('interface ') or stripped.startswith('enum '):
            # Skip nested struct definitions entirely
            brace_count = 0
            start_found = False
            while i < len(lines):
                if '{' in lines[i]:
                    start_found = True
                    brace_count += lines[i].count('{')
                if '}' in lines[i]:
                    brace_count -= lines[i].count('}')
                i += 1
                if start_found and brace_count == 0:
                    break
            continue
        
        # Check for union or group field definition: name : union { ... }
        union_match = re.match(r'^\s*(\w+)\s+:\s+(union|group)\s*{', stripped)
        if union_match:
            field_name = union_match.group(1)
            field_type = union_match.group(2)
            
            # Find the end of the union/group by counting braces
            brace_count = stripped.count('{') - stripped.count('}')
            j = i + 1
            union_body_lines = []
            
            while j < len(lines) and brace_count > 0:
                union_body_lines.append(lines[j])
                brace_count += lines[j].count('{') - lines[j].count('}')
                j += 1
            
            # Parse nested fields from union/group body
            nested_fields = []
            for union_line in union_body_lines:
                union_stripped = union_line.strip()
                if not union_stripped or union_stripped.startswith('#'):
                    continue
                
                # Parse nested field: name @index : type [= default];
                nested_field_match = re.match(
                    r'^(\w+)\s+@(\d+)\s*:\s*([^=;]+?)(?:\s*=\s*(.+?))?\s*;\s*$', 
                    union_stripped
                )
                if nested_field_match:
                    nested_fields.append(Field(
                        name=nested_field_match.group(1),
                        index=int(nested_field_match.group(2)),
                        type_=nested_field_match.group(3).strip(),
                        default=nested_field_match.group(4).strip() if nested_field_match.group(4) else None
                    ))
            
            # Create UnionField
            union_field = UnionField(
                name=field_name,
                type_=field_type,
                fields=nested_fields,
                index=0  # Union/group fields do not have a direct index in Cap'n'Proto
            )
            struct.union_fields.append(union_field)
            
            i = j
            continue
        
        # Regular field definition: name @index : type [= default];
        field_match = re.match(r'^(\w+)\s+@(\d+)\s*:\s*([^=;]+?)(?:\s*=\s*(.+?))?\s*;\s*$', stripped)
        if field_match:
            struct.fields.append(Field(
                name=field_match.group(1),
                index=int(field_match.group(2)),
                type_=field_match.group(3).strip(),
                default=field_match.group(4).strip() if field_match.group(4) else None
            ))
        
        i += 1


def parse_schema(content: str, filename: str = "<string>") -> Schema:
    """
    Parse a Cap'n'Proto schema string.
    
    Args:
        content: The schema file content
        filename: Original filename for reference
        
    Returns:
        Schema object with parsed elements
    """
    clean_content = remove_comments(content)
    lines = content.split('\n')
    
    schema = Schema(filename=filename)
    
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
    
    namespace_match = re.search(r'\$Cxx\.namespace\("([^"]+)"\)', content)
    if namespace_match:
        schema.namespace = namespace_match.group(1)
    
    import_matches = re.findall(r'using\s+(\w+)\s*=\s*import\s+"([^"]+)"', content)
    schema.imports = [f"{name}: {path}" for name, path in import_matches]
    # Parse structs
    struct_pattern = r'struct\s+(\w+)'
    
    # Stack to track nested struct relationships
    struct_stack: List[tuple] = []  # (name, start_pos, end_pos)
    
    # First pass: find all struct definitions and their positions
    struct_positions = []
    for match in re.finditer(struct_pattern, clean_content):
        struct_positions.append((match.start(), match.group(1)))
    
    # For each struct, determine its parent by tracking brace depth
    for start_pos, name in struct_positions:
        # Find the body of this struct
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
        
        # Parent is the most recent struct that:
        # 1. Starts before this one
        # 2. Contains this one (starts before, ends after)
        # 3. Is the innermost such struct
        parent = None
        for stack_name, stack_start, stack_end in struct_stack:
            # Check if this stack entry is an ancestor of current struct
            if stack_start < start_pos and stack_end > body_end:
                # This is an ancestor, but we need the innermost one
                # So we track the candidate and continue (later entries are more nested)
                parent = stack_name
        
        # Push this struct onto the stack
        struct_stack.append((name, start_pos, body_end))
        
        # Extract the struct body
        struct_body = clean_content[body_start+1:body_end]
        
        # Create the struct
        struct = Struct(name=name, namespace=schema.namespace, parent=parent)
        
        # Parse fields and union/group definitions from the struct body
        _parse_struct_body(struct, struct_body)
        
        schema.structs.append(struct)
    
    # Parse interfaces
    interface_pattern = r'interface\s+(\w+)'
    for match in re.finditer(interface_pattern, clean_content):
        name = match.group(1)
        start_pos = match.start()
        
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
            
            start_idx = len(content[:start_pos].split('\n')) - 1
            comment = extract_comment(lines, start_idx)
            
            interface = Interface(name=name, namespace=schema.namespace, comment=comment)
            
            for line in body.split('\n'):
                stripped = line.strip()
                
                if not stripped or stripped.startswith('#') or stripped.startswith('interface '):
                    continue
                
                arrow_pos = stripped.find('->')
                if arrow_pos < 0:
                    continue
                
                before_arrow = stripped[:arrow_pos].strip()
                after_arrow = stripped[arrow_pos+2:].strip().rstrip(';').strip()
                
                at_match = re.search(r'@\s*(\d+)', before_arrow)
                if not at_match:
                    continue
                
                name_match = re.search(r'^(\w+)\s*$', before_arrow[:at_match.start()].strip())
                if not name_match:
                    continue
                
                method_name = name_match.group(1)
                method_idx = int(at_match.group(1))
                
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
            
            values = []
            value_pattern = r'(\w+)\s+@(\d+)'
            for value_match in re.finditer(value_pattern, body):
                values.append(value_match.group(1))
            
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
    
    output = {
        "filename": schema.filename,
        "namespace": schema.namespace,
        "structs": [
            {
                "name": s.name,
                "parent": s.parent,
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
