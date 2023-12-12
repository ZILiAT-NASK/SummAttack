import gin


def make_configurable(cls, module: str):
    return gin.external_configurable(cls, module=module)
