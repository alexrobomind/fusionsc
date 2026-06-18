#!/usr/bin/env python3
"""
Cap'n'Proto Schema Parser

This module provides a parser for Cap'n'Proto schema files that extracts
semantic information for documentation generation. It properly handles
union/group nesting, field-level comments, interface extends clauses,
and nested types inside interfaces.
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
    is_union_member: bool = False
    union_name: Optional[str] = None  # Name of the enclosing union (empty str for anonymous)
    group_name: Optional[str] = None  # Name of enclosing group (if nested in a group)


@dataclass
class UnionGroup:
    """Represents a union or group with its member fields and nested groups."""
    name: str                          # Empty string for anonymous union/group, or named like "w7x"
    kind: str                          # 'union' or 'group'
    fields: List[Field] = field(default_factory=list)
    nested_groups: List['UnionGroup'] = field(default_factory=list)
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
    parent: Optional[str] = None  # Name of parent struct if nested
    fields: List[Field] = field(default_factory=list)  # Non-union regular fields
    unions: List[UnionGroup] = field(default_factory=list)  # Top-level unions/groups
    comment: Optional[str] = None


@dataclass
class Interface:
    """Represents a Cap'n'Proto interface."""
    name: str
    namespace: str
    methods: List[Method] = field(default_factory=list)
    comment: Optional[str] = None
    extends: List[str] = field(default_factory=list)
    nested_structs: List[Struct] = field(default_factory=list)
    nested_enums: List['Enum'] = field(default_factory=list)


@dataclass
class Enum:
    """Represents a Cap'n'Proto enum."""
    name: str
    namespace: str
    values: List[str] = field(default_factory=list)
    comment: Optional[str] = None
    parent_interface: Optional[str] = None  # e.g., "FieldCalculator"


@dataclass
class Schema:
    """Represents a parsed Cap'n'Proto schema file."""
    filename: str
    namespace: str = ""
    structs: List[Struct] = field(default_factory=list)
    interfaces: List[Interface] = field(default_factory=list)
    enums: List[Enum] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    import_aliases: Dict[str, str] = field(default_factory=dict)  # e.g., {"D": "data", "G": "geometry"}
    type_aliases: Dict[str, str] = field(default_factory=dict)  # e.g., {"Float64Tensor": "D.Float64Tensor"}
    file_comment: Optional[str] = None


# --- Comment utilities ---

def _strip_internal_markers(text: Optional[str]) -> Optional[str]:
    """Remove internal BEGIN/END marker lines and separator lines from comment text."""
    if text is None:
        return None
    lines = text.split('\n')
    filtered = [
        line for line in lines
        if not re.match(r'^=+\s*$', line.strip())  # Pure separator lines
        and not re.match(r'^(BEGIN|END|begin|end)\s*\[', line.strip())
        and not re.match(r'^=+\s*\w[\w\s-]*\w\s*=+\s*$', line.strip())  # === Section Name ===
        and not re.match(r'^-+\s*\w[\w\s-]*\w\s*-+\s*$', line.strip())  # --- Section Name ---
    ]
    result = '\n'.join(filtered).strip()
    return result if result else None


def remove_comments(text: str) -> str:
    """Remove C++ style comments from text (for structure parsing)."""
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def extract_comment(lines: List[str], start_idx: int) -> Optional[str]:
    """Extract comment block before a definition (looking backwards from start_idx)."""
    comment_lines = []
    for i in range(start_idx - 1, max(-1, start_idx - 30), -1):
        line = lines[i].strip()
        if line.startswith('#//!'):
            comment_lines.insert(0, line[4:].strip())
        elif line.startswith('#'):
            # Regular comment line -- include if it looks like documentation
            # (not just internal markers or bare #)
            content = line[1:].strip()
            if content:
                comment_lines.insert(0, content)
        elif line == '' and comment_lines:
            # Blank line within a comment block -- stop
            break
        elif line and not line.startswith('#'):
            break
    
    if comment_lines:
        return _strip_internal_markers('\n'.join(comment_lines))
    return None


def _extract_field_comment(original_lines: List[str], field_line_idx: int) -> Optional[str]:
    """Extract comment lines for a field definition.
    
    Cap'n'Proto schemas use two comment styles:
    1. Comments AFTER the field (common in some schemas):
        fieldName @0 : Type;
        # Description of this field
    2. Comments BEFORE the field (common in other schemas):
        # Description of this field
        fieldName @0 : Type;
    
    We check both directions and prefer the comment that actually exists.
    """
    # First, check for comments AFTER the field definition
    # A comment "after" is only valid if it's directly after the field (no blank line gap)
    after_comment_lines = []
    for i in range(field_line_idx + 1, min(field_line_idx + 20, len(original_lines))):
        stripped = original_lines[i].strip()
        if stripped.startswith('#'):
            comment_text = stripped[1:].strip()
            if comment_text.startswith('//!'):
                comment_text = comment_text[3:].strip()
            # Skip pure separator lines and section markers
            if re.match(r'^=+\s*$', comment_text):
                continue
            if re.match(r'^=+\s*\w[\w\s-]*\w\s*=+\s*$', comment_text):
                continue
            after_comment_lines.append(comment_text)
        elif stripped == '':
            # Blank line: if we haven't found any comments yet, stop
            # (the comment after must be directly adjacent, no gap)
            if not after_comment_lines:
                break
            # If we already have comments, a blank line ends the block
            break
        else:
            break
    
    # Then, check for comments BEFORE the field definition
    before_comment_lines = []
    for i in range(field_line_idx - 1, max(field_line_idx - 20, -1), -1):
        stripped = original_lines[i].strip()
        if stripped.startswith('#'):
            comment_text = stripped[1:].strip()
            if comment_text.startswith('//!'):
                comment_text = comment_text[3:].strip()
            # Skip pure separator lines and section markers
            if re.match(r'^=+\s*$', comment_text):
                break
            if re.match(r'^=+\s*\w[\w\s-]*\w\s*=+\s*$', comment_text):
                break
            before_comment_lines.insert(0, comment_text)
        elif stripped == '':
            # Blank line - could be separator, stop scanning before
            break
        else:
            break
    
    # Prefer the comment that exists. If both exist, prefer "after" (traditional Cap'n'Proto style)
    # but if only one exists, use that one.
    if after_comment_lines:
        result = ' '.join(after_comment_lines)
        return _strip_internal_markers(result) if result else None
    elif before_comment_lines:
        result = ' '.join(before_comment_lines)
        return _strip_internal_markers(result) if result else None
    return None


# --- Struct body parsing ---

def _parse_union_group_body(lines: List[str], start_idx: int, original_lines: List[str] = None, 
                             body_start_line: int = 0) -> tuple:
    """Parse the body of a union or group, returning (UnionGroup, end_idx).
    
    lines: list of lines inside the braces of the union/group
    start_idx: where to start parsing in those lines
    original_lines: the full original source lines (for comment extraction)
    body_start_line: the line offset of the struct body in original_lines
    """
    pass  # Implemented below


def _parse_struct_body(struct: Struct, body: str, original_lines: List[str] = None, 
                        body_start_line: int = 0):
    """Parse fields and union/group definitions from struct body.
    
    Args:
        struct: The Struct to populate
        body: The text content between the struct's braces
        original_lines: The full original source lines (for comment extraction)
        body_start_line: The line number in original_lines where the struct body starts
    """
    lines = body.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            i += 1
            continue
        
        # Skip nested struct/interface/enum definitions
        if stripped.startswith('struct ') or stripped.startswith('interface ') or stripped.startswith('enum '):
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
        
        # Check for named union/group: name : union { ... }  or  name : group { ... }
        named_union_match = re.match(r'^\s*(\w+)\s*:\s*(union|group)\s*\{', stripped)
        # Check for anonymous union: union { ... }
        anon_union_match = re.match(r'^\s*(union|group)\s*\{', stripped)
        
        if named_union_match or anon_union_match:
            if named_union_match:
                ug_name = named_union_match.group(1)
                ug_kind = named_union_match.group(2)
            else:
                ug_name = ""
                ug_kind = anon_union_match.group(1)
            
            # Find the end of this union/group by counting braces
            brace_count = stripped.count('{') - stripped.count('}')
            j = i + 1
            ug_body_lines = [line]
            
            while j < len(lines) and brace_count > 0:
                ug_body_lines.append(lines[j])
                brace_count += lines[j].count('{') - lines[j].count('}')
                j += 1
            
            # Extract comment for the union/group itself
            # Look backwards for a #//! doc comment (struct-level style).
            # Skip over field-level comments (which follow a field definition ending in ';')
            # because those belong to the preceding field, not to this union/group.
            ug_comment = None
            scanning = True
            for ci in range(i - 1, max(i - 10, -1), -1):
                prev_stripped = lines[ci].strip()
                if not prev_stripped:
                    # Blank line - stop scanning for union comment
                    break
                if prev_stripped.startswith('#//!'):
                    # Struct-level doc comment - use it
                    ug_comment = prev_stripped[4:].strip()
                    break
                if prev_stripped.startswith('#'):
                    # Regular comment - could be field-level (after ';') or standalone
                    # Check if the comment follows a field definition
                    content = prev_stripped[1:].strip()
                    if content and not re.match(r'^=+', content):
                        # Look further back - if we find a ';' line, this comment
                        # belongs to that field, not to the union. Skip it.
                        for cci in range(ci - 1, max(ci - 5, -1), -1):
                            check = lines[cci].strip()
                            if check.endswith(';'):
                                # This comment belongs to a field - skip
                                break
                            if check.startswith('#') or check == '':
                                continue
                            break
                        else:
                            # No field definition found before - this comment might
                            # belong to the union/group
                            ug_comment = content
                            break
                    continue
                elif prev_stripped.endswith(';'):
                    # Previous non-comment line ends with ';' (field definition)
                    # Any comments after it belong to that field. Stop.
                    break
                else:
                    break
            
            # Parse the union/group body recursively
            ug = _parse_union_group(lines, i, j, original_lines, body_start_line)
            ug.name = ug_name
            ug.kind = ug_kind
            ug.comment = _strip_internal_markers(ug_comment)
            
            struct.unions.append(ug)
            i = j
            continue
        
        # Regular field definition: name @index : type [= default];
        field_match = re.match(r'^(\w+)\s+@(\d+)\s*:\s*([^=;]+?)(?:\s*=\s*(.+?))?\s*;\s*$', stripped)
        if field_match:
            f = Field(
                name=field_match.group(1),
                index=int(field_match.group(2)),
                type_=field_match.group(3).strip(),
                default=field_match.group(4).strip() if field_match.group(4) else None
            )
            
            # Extract field-level comment
            if original_lines is not None:
                orig_idx = body_start_line + i
                f.comment = _extract_field_comment(original_lines, orig_idx)
            
            struct.fields.append(f)
        
        i += 1


def _parse_union_group(lines: List[str], start_idx: int, end_idx: int, 
                        original_lines: List[str] = None, body_start_line: int = 0) -> UnionGroup:
    """Parse a union or group block from lines[start_idx:end_idx].
    
    Returns a UnionGroup with fields and nested groups populated.
    """
    ug = UnionGroup(name="", kind="union")  # Name and kind set by caller
    
    i = start_idx + 1  # Skip the line with the opening brace (already counted)
    
    while i < end_idx:
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            i += 1
            continue
        
        # Check for named nested union/group: name : union { ... } or name : group { ... }
        named_match = re.match(r'^\s*(\w+)\s*:\s*(union|group)\s*\{', stripped)
        # Check for anonymous nested union: union { ... }
        anon_match = re.match(r'^\s*(union|group)\s*\{', stripped)
        
        if named_match or anon_match:
            if named_match:
                nested_name = named_match.group(1)
                nested_kind = named_match.group(2)
            else:
                nested_name = ""
                nested_kind = anon_match.group(1)
            
            # Count braces to find the end
            brace_count = stripped.count('{') - stripped.count('}')
            j = i + 1
            while j < end_idx and brace_count > 0:
                brace_count += lines[j].count('{') - lines[j].count('}')
                j += 1
            
            # Extract comment for nested union/group
            # Same logic as in _parse_struct_body: skip field-level comments
            nested_comment = None
            for ci in range(i - 1, max(i - 10, start_idx), -1):
                prev_stripped = lines[ci].strip()
                if not prev_stripped:
                    break
                if prev_stripped.startswith('#//!'):
                    nested_comment = prev_stripped[4:].strip()
                    break
                if prev_stripped.startswith('#'):
                    content = prev_stripped[1:].strip()
                    if content and not re.match(r'^=+', content):
                        for cci in range(ci - 1, max(ci - 5, start_idx), -1):
                            check = lines[cci].strip()
                            if check.endswith(';'):
                                break
                            if check.startswith('#') or check == '':
                                continue
                            break
                        else:
                            nested_comment = content
                            break
                    continue
                elif prev_stripped.endswith(';'):
                    break
                else:
                    break
            
            # Recursively parse
            nested_ug = _parse_union_group(lines, i, j, original_lines, body_start_line)
            nested_ug.name = nested_name
            nested_ug.kind = nested_kind
            nested_ug.comment = _strip_internal_markers(nested_comment)
            
            ug.nested_groups.append(nested_ug)
            i = j
            continue
        
        # Regular field definition: name @index : type [= default];
        field_match = re.match(r'^(\w+)\s+@(\d+)\s*:\s*([^=;]+?)(?:\s*=\s*(.+?))?\s*;\s*$', stripped)
        if field_match:
            f = Field(
                name=field_match.group(1),
                index=int(field_match.group(2)),
                type_=field_match.group(3).strip(),
                default=field_match.group(4).strip() if field_match.group(4) else None,
                is_union_member=True,
                union_name=ug.name,
            )
            
            # Extract field-level comment
            if original_lines is not None:
                orig_idx = body_start_line + i
                f.comment = _extract_field_comment(original_lines, orig_idx)
            
            # If this field is inside a named group, mark it
            if ug.kind == 'group' and ug.name:
                f.group_name = ug.name
            
            ug.fields.append(f)
        
        i += 1
    
    return ug


# --- Main schema parser ---

def parse_schema(content: str, filename: str = "<string>") -> Schema:
    """Parse a Cap'n'Proto schema string.
    
    Args:
        content: The schema file content
        filename: Original filename for reference
        
    Returns:
        Schema object with parsed elements
    """
    clean_content = remove_comments(content)
    lines = content.split('\n')
    
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
        schema.file_comment = _strip_internal_markers('\n'.join(file_comment_lines))
    
    # Extract namespace
    namespace_match = re.search(r'\$Cxx\.namespace\("([^"]+)"\)', content)
    if namespace_match:
        schema.namespace = namespace_match.group(1)
    
    # Extract imports (filter out compiler internals)
    import_matches = re.findall(r'using\s+(\w+)\s*=\s*import\s+"([^"]+)"', content)
    raw_imports = [f"{name}: {path}" for name, path in import_matches]
    schema.imports = [
        imp for imp in raw_imports
        if not any(internal in imp for internal in ['/capnp/', 'Cxx:', 'Java:'])
    ]
    
    # Build import alias map: alias -> schema stem name
    # e.g. using D = import "data.capnp" -> {"D": "data"}
    for alias, path in import_matches:
        if not any(internal in path for internal in ['/capnp/', 'Cxx:', 'Java:']) and alias not in ('Cxx', 'Java'):
            stem = Path(path).stem
            schema.import_aliases[alias] = stem
    
    # Extract type aliases: using Foo = D.Foo or using Foo = D.Foo.Bar
    # These let us resolve D.Float64Tensor -> Float64Tensor
    type_alias_matches = re.findall(r'using\s+(\w+)\s*=\s*(\w+\.\w+(?:\.\w+)*)\s*;', content)
    for alias, qualified in type_alias_matches:
        schema.type_aliases[alias] = qualified
    
    # --- Parse structs ---
    struct_pattern = r'struct\s+(\w+)'
    struct_stack = []  # (name, start_pos, end_pos)
    struct_positions = []
    
    for match in re.finditer(struct_pattern, clean_content):
        struct_positions.append((match.start(), match.group(1)))
    
    for start_pos, name in struct_positions:
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
        
        # Determine parent
        parent = None
        for stack_name, stack_start, stack_end in struct_stack:
            if stack_start < start_pos and stack_end > body_end:
                parent = stack_name
        
        struct_stack.append((name, start_pos, body_end))
        
        # Extract struct body
        struct_body = clean_content[body_start + 1:body_end]
        
        # Find the line number of the struct body start in original content
        body_start_line = len(content[:body_start + 1].split('\n')) - 1
        
        # Extract struct comment
        struct_start_line = len(content[:start_pos].split('\n')) - 1
        comment = extract_comment(lines, struct_start_line)
        
        new_struct = Struct(name=name, namespace=schema.namespace, parent=parent, comment=comment)
        
        # Parse fields and union/group definitions
        _parse_struct_body(new_struct, struct_body, lines, body_start_line)
        
        # If this struct has a parent that is an interface, skip it here
        # (it will be attached to the interface as a nested type)
        # For now, we add all structs to the schema and filter later
        
        schema.structs.append(new_struct)
    
    # --- Parse interfaces ---
    interface_pattern = r'interface\s+(\w+)(?:\s+\$[^\{]+?)?(?:\s+extends\s*\(([^)]*)\))?\s*\{'
    for match in re.finditer(interface_pattern, clean_content):
        iface_name = match.group(1)
        extends_str = match.group(2)  # May be None
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
            body = clean_content[body_start + 1:body_end]
            
            # Find the line number for comment extraction
            iface_start_line = len(content[:start_pos].split('\n')) - 1
            comment = extract_comment(lines, iface_start_line)
            
            # Parse extends list
            extends = []
            if extends_str:
                extends = [e.strip() for e in extends_str.split(',') if e.strip()]
            
            interface = Interface(
                name=iface_name,
                namespace=schema.namespace,
                comment=comment,
                extends=extends
            )
            
            body_start_line = len(content[:body_start + 1].split('\n')) - 1
            
            # Parse nested types inside the interface body
            _parse_interface_body(interface, body, lines, body_start_line, schema, clean_content)
            
            schema.interfaces.append(interface)
    
    # --- Parse enums ---
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
            body = clean_content[body_start + 1:body_end]
            
            values = []
            value_pattern = r'(\w+)\s+@(\d+)'
            for value_match in re.finditer(value_pattern, body):
                values.append(value_match.group(1))
            
            enum_start_line = len(content[:start_pos].split('\n')) - 1
            comment = extract_comment(lines, enum_start_line)
            
            # Check if this enum is nested inside an interface
            parent_interface = None
            for iface in schema.interfaces:
                iface_body_start = len(content[:content.find('{', content.find(f'interface {iface.name}'))].split('\n')) - 1
                # Check if enum start is between interface body start and end
                if_iface_start = clean_content.find(f'interface {iface_name}')
                # Simple check: does the enum appear inside this interface's braces?
                # We check by finding the interface's body bounds
                iface_match = re.search(rf'interface\s+{re.escape(iface.name)}(?:\s+extends\s*\([^)]*\))?\s*\{{', clean_content)
                if iface_match:
                    iface_body_start_pos = iface_match.start()
                    # Find interface end
                    ibc = 0
                    iface_body_end_pos = -1
                    for ci, ch in enumerate(clean_content[iface_body_start_pos:], iface_body_start_pos):
                        if ch == '{':
                            ibc += 1
                        elif ch == '}':
                            ibc -= 1
                            if ibc == 0:
                                iface_body_end_pos = ci
                                break
                    if iface_body_end_pos > 0 and start_pos > iface_body_start_pos and start_pos < iface_body_end_pos:
                        parent_interface = iface.name
                        break
            
            enum = Enum(
                name=name,
                namespace=schema.namespace,
                values=values,
                comment=comment,
                parent_interface=parent_interface
            )
            schema.enums.append(enum)
    
    # Remove structs/enums that are nested inside interfaces from the top-level lists
    # (they are attached to their parent Interface objects instead)
    nested_struct_names = set()
    for iface in schema.interfaces:
        for ns in iface.nested_structs:
            nested_struct_names.add(ns.name)
    schema.structs = [s for s in schema.structs if s.name not in nested_struct_names]
    
    nested_enum_names = set()
    for iface in schema.interfaces:
        for ne in iface.nested_enums:
            nested_enum_names.add(ne.name)
    schema.enums = [e for e in schema.enums if e.name not in nested_enum_names]
    
    return schema


def _parse_interface_body(interface: Interface, body: str, original_lines: List[str],
                           body_start_line: int, schema: Schema, clean_content: str):
    """Parse methods and nested types from an interface body."""
    lines = body.split('\n')
    i = 0
    
    # Track nested type definitions to parse them
    nested_struct_positions = []
    nested_enum_positions = []
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            i += 1
            continue
        
        # Detect nested struct definition
        if stripped.startswith('struct '):
            struct_name_match = re.match(r'struct\s+(\w+)', stripped)
            if struct_name_match:
                # Find the end by counting braces within the interface body lines
                brace_count = stripped.count('{') - stripped.count('}')
                j = i + 1
                struct_body_lines = [line]
                while j < len(lines) and brace_count > 0:
                    struct_body_lines.append(lines[j])
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                struct_body = '\n'.join(struct_body_lines)
                # Get the part between braces
                first_brace = struct_body.find('{')
                last_brace = struct_body.rfind('}')
                if first_brace >= 0 and last_brace > first_brace:
                    inner = struct_body[first_brace + 1:last_brace]
                    
                    nested_struct = Struct(
                        name=struct_name_match.group(1),
                        namespace=interface.namespace,
                        parent=interface.name
                    )
                    
                    nested_body_start_line = body_start_line + i
                    nested_start_line_in_orig = body_start_line + i
                    comment = extract_comment(original_lines, nested_start_line_in_orig)
                    nested_struct.comment = comment
                    
                    _parse_struct_body(nested_struct, inner, original_lines, nested_body_start_line)
                    interface.nested_structs.append(nested_struct)
                
                i = j
                continue
        
        # Detect nested enum definition
        if stripped.startswith('enum '):
            enum_name_match = re.match(r'enum\s+(\w+)', stripped)
            if enum_name_match:
                brace_count = stripped.count('{') - stripped.count('}')
                j = i + 1
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                # Parse the enum values from the lines between braces
                enum_lines = lines[i:j]
                values = []
                for el in enum_lines:
                    val_match = re.search(r'(\w+)\s+@(\d+)', el)
                    if val_match:
                        values.append(val_match.group(1))
                
                enum_start_line_in_orig = body_start_line + i
                comment = extract_comment(original_lines, enum_start_line_in_orig)
                
                nested_enum = Enum(
                    name=enum_name_match.group(1),
                    namespace=interface.namespace,
                    values=values,
                    comment=comment,
                    parent_interface=interface.name
                )
                interface.nested_enums.append(nested_enum)
                
                i = j
                continue
        
        # Parse method definition: name @index (params) -> return_type;
        # Also handles: name @index -> return_type;  (no params)
        arrow_pos = stripped.find('->')
        if arrow_pos < 0:
            i += 1
            continue
        
        before_arrow = stripped[:arrow_pos].strip()
        after_arrow = stripped[arrow_pos + 2:].strip().rstrip(';').strip()
        
        # Must have @index
        at_match = re.search(r'@\s*(\d+)', before_arrow)
        if not at_match:
            i += 1
            continue
        
        # Extract method name (everything before @index)
        name_part = before_arrow[:at_match.start()].strip()
        # Remove parameter parentheses if present in name_part
        paren_start = name_part.find('(')
        if paren_start >= 0:
            method_name = name_part[:paren_start].strip()
        else:
            method_name = name_part
        
        if not method_name or not re.match(r'^\w+$', method_name):
            i += 1
            continue
        
        method_idx = int(at_match.group(1))
        
        # Extract parameters from before_arrow
        paren_match = re.search(r'\(', before_arrow)
        params = ""
        if paren_match:
            paren_start_pos = paren_match.start()
            paren_count = 1
            paren_end_pos = paren_start_pos + 1
            while paren_end_pos < len(before_arrow) and paren_count > 0:
                if before_arrow[paren_end_pos] == '(':
                    paren_count += 1
                elif before_arrow[paren_end_pos] == ')':
                    paren_count -= 1
                paren_end_pos += 1
            
            if paren_count == 0:
                params = before_arrow[paren_start_pos + 1:paren_end_pos - 1]
        
        # Extract method comment from original lines
        method_comment = None
        orig_line_idx = body_start_line + i
        # Check for trailing comment lines
        for ci in range(orig_line_idx + 1, min(orig_line_idx + 10, len(original_lines))):
            s = original_lines[ci].strip()
            if s.startswith('#'):
                content = s[1:].strip()
                if content:
                    method_comment = content
                    break
            elif s == '':
                continue
            else:
                break
        
        method = Method(
            name=method_name,
            index=method_idx,
            return_type=after_arrow,
            comment=_strip_internal_markers(method_comment)
        )
        
        # Parse parameters
        if params.strip():
            # Handle generic params like [T] separately
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
        i += 1


def parse_schema_file(filepath: str) -> Schema:
    """Parse a Cap'n'Proto schema file."""
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
                "fields": [{"name": f.name, "type": f.type_, "index": f.index, "comment": f.comment, "is_union_member": f.is_union_member} for f in s.fields],
                "unions": [
                    {
                        "name": u.name,
                        "kind": u.kind,
                        "fields": [{"name": f.name, "type": f.type_, "comment": f.comment} for f in u.fields],
                        "nested_groups": [
                            {
                                "name": ng.name,
                                "kind": ng.kind,
                                "fields": [{"name": f.name, "type": f.type_} for f in ng.fields]
                            }
                            for ng in u.nested_groups
                        ]
                    }
                    for u in s.unions
                ]
            }
            for s in schema.structs
        ],
        "interfaces": [
            {
                "name": i.name,
                "extends": i.extends,
                "methods": [
                    {
                        "name": m.name,
                        "parameters": [{"name": p.name, "type": p.type_} for p in m.parameters],
                        "return_type": m.return_type,
                        "comment": m.comment
                    }
                    for m in i.methods
                ],
                "nested_structs": [{"name": ns.name} for ns in i.nested_structs],
                "nested_enums": [{"name": ne.name} for ne in i.nested_enums]
            }
            for i in schema.interfaces
        ],
        "enums": [
            {
                "name": e.name,
                "values": e.values,
                "parent_interface": e.parent_interface
            }
            for e in schema.enums
        ]
    }
    
    print(json.dumps(output, indent=2))
