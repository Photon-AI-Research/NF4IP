from cement import Controller, ex


class Tools(Controller):
    class Meta:
        label = 'tools'
        stacked_on = 'base'
        stacked_type = 'embedded'

    @ex(
        help='prints current configuration',
    )
    def config(self):
        config = self.app.config.get_dict()

        config = {k: v for k, v in config.items() if not k.startswith('controller.') and not k == 'mail.dummy'}



        print(self.app.template.render(
            """
{% for section, configs in config.items() %}
    {{ section }}:
    {% for key, val in configs.items() %}\
    {{ "%-30s" | format(key,) }}:\t{{ "%-20s" | format(val,) }} ({{ type(val).__name__ }})
    {% endfor %}
{% endfor %}
        """,
            {'config': config, 'type': type, }
        ))
