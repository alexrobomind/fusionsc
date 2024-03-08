{{ fullname | escape | underline}}

This class is an accessor for a Cap'n'proto struct embedded in a message.

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}

The message object has the following attributes:

.. autosummary::
{% for item in attributes %}
	~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

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