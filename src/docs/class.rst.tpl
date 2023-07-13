{% macro withTypeParams(type) %}{{ type.name }}{% if type.arguments %}<{{ fullTypeName(type.arguments[0].type) }}{% for i in range(1, len(type.arguments)) %}, {{ fullTypeName(type.arguments[i].type) }}{% endfor %}>{% endif %}{% endmacro %}

{% macro fullTypeName(type) %}{% if isinstance(type, javalang.tree.ReferenceType) and type.sub_type.name %}{{ withTypeParams(type) }}.{{ fullTypeName(type.sub_type) }}{% else %}{{ withTypeParams(type) }}{% endif %}{% endmacro %}

{% macro funcParam(param) %}{{ fullTypeName(param.type) }} {{ param.name }}{% endmacro %}
{% macro funcParams(params) %}{% if len(params) != 0 %}{{ funcParam(params[0]) }}{% for i in range(1, len(params)) %}, {{ funcParam(params[i]) }}{% endfor %}{% endif %}{% endmacro %}

{% if isinstance(cls, javalang.tree.ClassDeclaration) or isinstance(cls, javalang.tree.InterfaceDeclaration) %}

.. _cls_{{ fullName }}:
{% if isinstance(cls, javalang.tree.ClassDeclaration) %}Class{% else %}Interface{% endif %} {{ cls.name }}
====================

Full name {{ fullName}}

{% if cls.extends and isinstance(cls, javalang.tree.ClassDeclaration) -%}
Extends {{ fullTypeName(cls.extends) }}
{% elif cls.extends and isinstance(cls, javalang.tree.InterfaceDeclaration) -%}
Extends:
{% for base in cls.extends %}
- {{ fullTypeName(base) }}
{% endfor %}
{% endif %}

{% if cls.implements -%}
Implements:
{% for base in cls.implements %}
- {{ fullTypeName(base) }}
{% endfor %}
{% endif %}

{% if cls.type_parameters -%}
Type parameters:
{% for param in cls.type_parameters %}
- {{ param.name }}{% if param.extends %} extends {{ param.extends }}{% endif %}
{% endfor %}
{% endif %}

Member classes:

.. toctree::
  :maxdepth: 2

{% for member in cls.body %}{% if isinstance(member, javalang.tree.ClassDeclaration) or isinstance(member, javalang.tree.InterfaceDeclaration) %}
  {{ fullName }}.{{ member.name }}.class
{% endif %}{% endfor %}

Method information:

{% if len(cls.constructors) > 0 -%}
.. list-table:: Constructors
  :header-rows: 1
  
  * - Call signature

{% for m in cls.constructors %}
  * - {{ m.name }}({{ funcParams(m.parameters) }}){% endfor %}
{% endif %}

{% if len(cls.methods) > 0 -%}
.. list-table:: Methods
  :header-rows: 1
  
  * - Return type
    - Call signature

{% for m in cls.methods %}
  * - {{ fullTypeName(m.return_type) }}
    - {{ m.name }}({{ funcParams(m.parameters) }}){% endfor %}
{% endif %}

{% elif isinstance(cls, javalang.tree.EnumDeclaration) %}

Enum {{ cls.name }}
========================

{% endif %}