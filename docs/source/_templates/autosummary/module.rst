{% extends "!autosummary/module.rst" %}

{# This file is almost the same as the default, but adds :toctree: to the autosummary directives.
   The original can be found at `sphinx/ext/autosummary/templates/autosummary/module.rst`. #}


{% block functions %}
{%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:

   {% for item in functions %}
      {{ item }}
   {%- endfor %}
{% endif %}
{%- endblock %}

{% block classes %}
{% if classes %}
   .. rubric:: {{ _('Classes') }}
   .. autosummary::
      :toctree:
      :nosignatures:

   {% for item in classes %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
{% endif %}
   {% endblock %}

{%- block modules %}
{#{%- if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}#}
{%- endblock %}