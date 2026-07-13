from .spec import ModelSpec
from .vm import VelocityModule
from .straightpcf import StraightPCFModule


def get_model(model_config, **kwargs) -> ModelSpec:
    model_map = {
        "VelocityModule": VelocityModule,
        "StraightPCFModule": StraightPCFModule,
    }
    target = model_config["__target__"]
    del model_config["__target__"]
    assert target in model_map, (
        f"expect: [{','.join(model_map.keys())}], found: {target}"
    )
    return model_map[target](model_config=model_config, **kwargs)
