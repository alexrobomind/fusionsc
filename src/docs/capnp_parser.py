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


def _split_top_level(s: str, delimiter: str = ",") -> List[str]:
    """Split a string on a delimiter, but only at the top level (not inside parens/brackets/angle brackets).
    
    This is used to split parameter/return type lists like:
      "pos : List(Float64), axis : Data.Float64Tensor, meanField : Float64"
    into individual "name : type" entries without splitting at the comma inside List(Float64).
    """
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch in ('(', '[', '<'):
            depth += 1
            current.append(ch)
        elif ch in (')', ']', '>'):
            depth -= 1
            current.append(ch)
        elif ch == delimiter and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    remainder = ''.join(current).strip()
    if remainder:
        parts.append(remainder)
    return parts


def _parse_name_type_pairs(text: str) -> List[tuple]:
    """Parse 'name : Type, name2 : Type2' pairs respecting nested parens.
    
    Returns list of (name, type_str) tuples.
    """
    parts = _split_top_level(text, ",")
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        colon_match = re.match(r'(\w+)\s*:\s*(.+)', part)
        if colon_match:
            result.append((colon_match.group(1), colon_match.group(2).strip()))
    return result


@dataclass
class Field:
    """Represents a struct field or method parameter.
    
    When used as a method parameter/return, is_struct_ref=True indicates
    that this is a bare struct name reference (e.g., FLTRequest) rather
    than an inline field with a name and type.
    """
    name: str
    index: int
    type_: str
    default: Optional[str] = None
    annotations: Dict[str, str] = field(default_factory=dict)
    comment: Optional[str] = None
    is_union_member: bool = False
    union_name: Optional[str] = None  # Name of the enclosing union (empty str for anonymous)
    group_name: Optional[str] = None  # Name of enclosing group (if nested in a group)
    is_struct_ref: bool = False  # True when this represents a bare struct name (method param/return)


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
    """Represents an interface method.
    
    Cap'n'Proto methods have two forms for parameters and return types:
    
    1. Named struct reference (bare name, no parens):
         trace @0 FLTRequest -> FLTResponse;
       Stored as a single Field with name="" and type_="FLTRequest"
       (is_struct_ref=True on the Field).
       
    2. Inline struct (parenthesized name:type pairs):
         findAxis @1 FindAxisRequest -> (pos : List(Float64), axis : Float64Tensor);
       Stored as individual Fields with their names and types.
    """
    name: str
    index: int
    parameters: List[Field] = field(default_factory=list)
    return_type: str = "()"  # Kept for backward compat; return_fields is preferred
    return_fields: List[Field] = field(default_factory=list)
    comment: Optional[str] = None


@dataclass
class Struct:
    """Represents a Cap'n'Proto struct."""
    name: str
    namespace: str
    parent: Optional[str] = None  # Fully qualified name of parent (e.g., "LoadBalancerConfig.Rule")
    fields: List[Field] = field(default_factory=list)  # Non-union regular fields
    unions: List[UnionGroup] = field(default_factory=list)  # Top-level unions/groups
    comment: Optional[str] = None
    nested_structs: List['Struct'] = field(default_factory=list)
    nested_enums: List['Enum'] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        """Fully qualified name including parent path (e.g., LoadBalancerConfig.Backend)."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name


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
    nested_interfaces: List['Interface'] = field(default_factory=list)
    parent: Optional[str] = None  # Fully qualified name of parent (if nested in struct/interface)

    @property
    def qualified_name(self) -> str:
        """Fully qualified name including parent path."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name


@dataclass
class Enum:
    """Represents a Cap'n'Proto enum."""
    name: str
    namespace: str
    values: List[str] = field(default_factory=list)
    comment: Optional[str] = None
    parent: Optional[str] = None  # Fully qualified name of parent (e.g., "FieldCalculator" or "LoadBalancerConfig.Rule")

    @property
    def qualified_name(self) -> str:
        """Fully qualified name including parent path."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name


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
    """Remove C++ style comments from text (for structure parsing).
    
    Removes // style comments but preserves # style comments.
    The # comments are kept because they contain struct/interface documentation
    and are on their own lines, so they won't interfere with pattern matching
    (which now uses multiline mode to anchor to start of line).
    """
    #text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    #text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove #// style comments (often used for snippet markers)
    text = re.sub(r'\\#//.*?$', '', text, flags=re.MULTILINE)
    return text


def extract_comment(lines: List[str], start_idx: int) -> Optional[str]:
    """Extract comment block before a definition (looking backwards from start_idx).
    
    Only includes regular comment lines (starting with #) that look like documentation.
    Skips Doxygen snippet markers (starting with #//!) which are not documentation comments.
    """
    comment_lines = []
    for i in range(start_idx - 1, max(-1, start_idx - 30), -1):
        line = lines[i].strip()
        if line.startswith('#//!'):
            # Doxygen snippet marker - skip it (not a documentation comment)
            continue
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
    
    According to Cap'n'Proto convention, field comments are placed strictly
    BELOW the field definition. This function only checks for comments after
    the field line, not before.
    
    Example:
        fieldName @0 : Type;
        # Description of this field
    """
    # Only check for comments AFTER the field definition
    # A comment is only valid if it's directly after the field (no blank line gap)
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
    
    if after_comment_lines:
        result = ' '.join(after_comment_lines)
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
    lines = clean_content.split('\n')
    
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
    # Use multiline mode (^ anchors to start of line) to avoid matching in comments
    struct_pattern = r'(?m)^struct\s+(\w+)'
    struct_stack = []  # (qualified_name, start_pos, end_pos)
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
        
        # Determine parent (fully qualified name) by checking what enclosing
        # struct or interface contains this one.
        parent_qualified = None
        for stack_qname, stack_start, stack_end in struct_stack:
            if stack_start < start_pos and stack_end > body_end:
                parent_qualified = stack_qname
        
        # Build the qualified name for this struct
        if parent_qualified:
            qualified_name = f"{parent_qualified}.{name}"
        else:
            qualified_name = name
        
        struct_stack.append((qualified_name, start_pos, body_end))
        
        # Extract struct body
        struct_body = clean_content[body_start + 1:body_end]
        
        # Find the line number of the struct body start in original content
        body_start_line = len(content[:body_start + 1].split('\n')) - 1
        
        # Extract struct comment
        struct_start_line = len(content[:start_pos].split('\n')) - 1
        comment = extract_comment(lines, struct_start_line)
        
        new_struct = Struct(name=name, namespace=schema.namespace, parent=parent_qualified, comment=comment)
        
        # Parse fields and union/group definitions
        _parse_struct_body(new_struct, struct_body, lines, body_start_line)
        
        # Parse nested types (structs, enums) inside this struct body
        _parse_nested_types_in_struct(new_struct, struct_body, lines, body_start_line, schema)
        
        schema.structs.append(new_struct)
    
    # --- Parse interfaces ---
    # Use multiline mode (^ anchors to start of line) to avoid matching in comments
    interface_pattern = r'(?m)^interface\s+(\w+)(?:\s*\([^)]*\))?(?:\s+@0x[0-9a-fA-F]+)?(?:\s+\$[^\{]+?)?(?:\s+extends\s*\(([^)]*)\))?\s*\{'
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
            
            # Determine if this enum is nested inside a struct or interface
            # by checking position against all parsed types' body ranges
            parent_qualified = None
            
            # Check structs (using the struct_stack for position info)
            for stack_qname, stack_start, stack_end in struct_stack:
                if stack_start < start_pos and stack_end > body_end:
                    parent_qualified = stack_qname
            
            # Check interfaces
            for iface in schema.interfaces:
                iface_match = re.search(rf'interface\s+{re.escape(iface.name)}(?:\s+extends\s*\([^)]*\))?\s*\{{', clean_content)
                if iface_match:
                    iface_body_start_pos = iface_match.start()
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
                        # Interface takes precedence or is the actual parent
                        parent_qualified = iface.qualified_name
                        break
            
            enum = Enum(
                name=name,
                namespace=schema.namespace,
                values=values,
                comment=comment,
                parent=parent_qualified
            )
            schema.enums.append(enum)
    
    # --- Reorganize: move nested types into their parents ---
    # Build lookup by qualified name for structs
    struct_by_qname = {}
    for s in schema.structs:
        struct_by_qname[s.qualified_name] = s
    
    # Move child structs into their parent struct's nested_structs
    # Process in reverse order so deeper-nested structs get moved first
    top_level_structs = []
    for s in schema.structs:
        if s.parent:
            # Find the parent struct or interface
            if s.parent in struct_by_qname:
                parent_struct = struct_by_qname[s.parent]
                parent_struct.nested_structs.append(s)
            else:
                # Parent might be an interface
                found_iface = False
                for iface in schema.interfaces:
                    if iface.qualified_name == s.parent:
                        # Check if this struct was already parsed by _parse_interface_body
                        already_nested = any(ns.name == s.name for ns in iface.nested_structs)
                        if not already_nested:
                            iface.nested_structs.append(s)
                        found_iface = True
                        break
                if not found_iface:
                    # Unknown parent, keep at top level
                    top_level_structs.append(s)
        else:
            top_level_structs.append(s)
    schema.structs = top_level_structs
    
    # Move child enums into their parent struct's or interface's nested_enums
    top_level_enums = []
    for e in schema.enums:
        if e.parent:
            if e.parent in struct_by_qname:
                parent_struct = struct_by_qname[e.parent]
                parent_struct.nested_enums.append(e)
            else:
                found_parent = False
                for iface in schema.interfaces:
                    if iface.qualified_name == e.parent:
                        already_nested = any(ne.name == e.name for ne in iface.nested_enums)
                        if not already_nested:
                            iface.nested_enums.append(e)
                        found_parent = True
                        break
                if not found_parent:
                    top_level_enums.append(e)
        else:
            top_level_enums.append(e)
    schema.enums = top_level_enums
    
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
                        parent=interface.qualified_name
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
                    parent=interface.qualified_name
                )
                interface.nested_enums.append(nested_enum)
                
                i = j
                continue
        
        # Detect nested interface definition
        if stripped.startswith('interface '):
            iface_name_match = re.match(r'interface\s+(\w+)', stripped)
            if iface_name_match:
                brace_count = stripped.count('{') - stripped.count('}')
                j = i + 1
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                # Parse the interface body lines
                iface_lines = lines[i:j]
                iface_body = '\n'.join(iface_lines)
                
                # Extract the content between braces
                brace_start = iface_body.find('{')
                brace_end = -1
                if brace_start >= 0:
                    bc = 0
                    for idx, ch in enumerate(iface_body[brace_start:], brace_start):
                        if ch == '{':
                            bc += 1
                        elif ch == '}':
                            bc -= 1
                            if bc == 0:
                                brace_end = idx
                                break
                
                if brace_end > 0:
                    inner = iface_body[brace_start + 1:brace_end]
                    
                    iface_start_line_in_orig = body_start_line + i
                    comment = extract_comment(original_lines, iface_start_line_in_orig)
                    
                    nested_interface = Interface(
                        name=iface_name_match.group(1),
                        namespace=interface.namespace,
                        comment=comment,
                        parent=interface.qualified_name
                    )
                    
                    # Parse the nested interface body
                    nested_body_start_line = body_start_line + i
                    _parse_interface_body(nested_interface, inner, original_lines, nested_body_start_line, schema, clean_content)
                    
                    interface.nested_interfaces.append(nested_interface)
                
                i = j
                continue
        
        # Parse method definition: name @index (params) -> return_type;
        # Also handles: name @index -> return_type;  (no params)
        # Also handles: name @index StructName -> StructName;  (bare struct refs)
        # Also handles: name @index StructName -> (field : Type, ...);  (mixed)
        # Also handles: name @index (field : Type, ...) -> StructName;  (mixed)
        # Also handles: name @index (params);  (no return type, defaults to empty struct)
        arrow_pos = stripped.find('->')
        
        if arrow_pos < 0:
            # Check if this is a method without return type (ends with ; and has @index)
            # e.g., "transmit @2 (start : UInt64, end : UInt64, receiver : Receiver);"
            if stripped.endswith(';') and '@' in stripped:
                # Try to parse as method without return type
                before_arrow = stripped[:-1].strip()  # Remove trailing ; and whitespace
                after_arrow = ''  # No return type - defaults to empty struct
            else:
                i += 1
                continue
        else:
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
        
        # Extract the parameter portion: everything after @index and before ->
        after_at = before_arrow[at_match.end():].strip()
        
        # Extract parameters from after_at
        # Two forms:
        #   1. Bare struct name: after_at = "FLTRequest"  (no parens)
        #   2. Inline params:    after_at = "(field : Type, ...)"
        paren_match = re.search(r'\(', after_at)
        params = ""
        param_struct_ref = ""  # For bare struct name like FLTRequest
        if paren_match:
            paren_start_pos = paren_match.start()
            paren_count = 1
            paren_end_pos = paren_start_pos + 1
            while paren_end_pos < len(after_at) and paren_count > 0:
                if after_at[paren_end_pos] == '(':
                    paren_count += 1
                elif after_at[paren_end_pos] == ')':
                    paren_count -= 1
                paren_end_pos += 1
            
            if paren_count == 0:
                params = after_at[paren_start_pos + 1:paren_end_pos - 1]
        else:
            # Bare struct name reference (e.g., "FLTRequest")
            # Could also be empty (no params)
            candidate = after_at.strip()
            if candidate and re.match(r'^[\w.]+$', candidate):
                param_struct_ref = candidate
        
        # Parse return type
        # Two forms:
        #   1. Bare struct name: after_arrow = "FLTResponse"  (no parens)
        #   2. Inline return:    after_arrow = "(field : Type, ...)"
        return_struct_ref = ""  # For bare struct name
        return_inline = ""     # For inline return fields
        if after_arrow.startswith('(') and after_arrow.endswith(')'):
            return_inline = after_arrow[1:-1]
        elif after_arrow and re.match(r'^[\w.]+$', after_arrow):
            # Bare struct name reference (e.g., "FLTResponse")
            return_struct_ref = after_arrow
        
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
        if param_struct_ref:
            # Bare struct reference as parameter
            method.parameters.append(Field(
                name="",
                index=0,
                type_=param_struct_ref,
                is_struct_ref=True
            ))
        elif params.strip():
            # Inline parameter fields (using paren-aware splitting)
            for param_name, param_type in _parse_name_type_pairs(params):
                method.parameters.append(Field(
                    name=param_name,
                    index=0,
                    type_=param_type
                ))
        
        # Parse return fields
        if return_struct_ref:
            method.return_fields.append(Field(
                name="",
                index=0,
                type_=return_struct_ref,
                is_struct_ref=True
            ))
        elif return_inline.strip():
            # Inline return fields (using paren-aware splitting)
            for ret_name, ret_type in _parse_name_type_pairs(return_inline):
                method.return_fields.append(Field(
                    name=ret_name,
                    index=0,
                    type_=ret_type
                ))
        else:
            # No explicit return (or couldn't parse) - keep return_type as-is
            pass
        
        interface.methods.append(method)
        i += 1


def _parse_nested_types_in_struct(parent_struct: Struct, body: str, original_lines: List[str],
                                    body_start_line: int, schema: Schema):
    """No-op: nested types are handled by the reorganization pass in parse_schema().
    
    The top-level struct/enum regex parsing in parse_schema() discovers all types,
    and the reorganization code at the end moves child types into their parents'
    nested_structs/nested_enums lists. This function exists for structural
    consistency with _parse_interface_body but doesn't need to do additional work.
    """
    pass


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
                "parent": e.parent
            }
            for e in schema.enums
        ]
    }
    
    print(json.dumps(output, indent=2))
