Package {{ packageName }}
=========================

Member classes:

.. toctree::
  :maxdepth: 2

{% for name, cls in members.items() -%}
{% if isinstance(cls, javalang.tree.ClassDeclaration) %}
  {{ packageName }}.{{ cls.name }}.class
{% endif %}{% if isinstance(cls, javalang.tree.InterfaceDeclaration) %}
  {{ packageName }}.{{ cls.name }}.class
{% endif %}
{% endfor %}
