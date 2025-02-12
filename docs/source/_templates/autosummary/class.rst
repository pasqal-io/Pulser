{{ objname| escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:
   :member-order: groupwise

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :nosignatures:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods and methods != ['__init__'] %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') or item in ['__call__'] %}
         ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}

   .. rubric:: {{ _('Signatures') }}
   {% endif %}
   {% endblock %}