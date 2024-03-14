{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }}

.. autosummary::
{% for item in methods %}
  ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% if "Client" in members %}
This class represents a capability type. To use it, use the type {{ objname }}.Client (see below).
{% endif %}

{% if "Reader" in members %}
This class represents a struct type. The accessor types are .Reader and .Builder (see below)
{% endif %}

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}

.. autosummary::
{% for item in attributes %}
	~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methodsdescs %}
{% if methods %}
.. rubric:: Methods

{% for item in methods %}
.. automethod:: {{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% if "Client" in members %}
.. rubric:: Client class
.. autoclass:: {{ module }}::{{ objname }}.Client
	:members:
{% endif %}

{% if "newMessage" in methods %}
.. autosummary::
  :recursive:
  :template: summary-accessor.rst
  :toctree:
  
  {{ objname }}.Builder
  {{ objname }}.Reader
  {{ objname }}.Pipeline
 
.. toctree::
   {{ fullname }}.Builder
   {{ fullname }}.Reader
   {{ fullname }}.Pipeline
{% endif %}