from typing import Any, Dict

from rektgbm.base import MethodName
from rektgbm.objective import ObjectiveName


def set_additional_params(
    objective: ObjectiveName,
    method: MethodName,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if objective == ObjectiveName.quantile:
        if method == MethodName.lightgbm and "quantile_alpha" in params.keys():
            params["alpha"] = params.pop("quantile_alpha")
        elif method == MethodName.xgboost and "alpha" in params.keys():
            params["quantile_alpha"] = params.pop("alpha")
    elif objective == ObjectiveName.huber:
        if method == MethodName.lightgbm and "huber_slope" in params.keys():
            params["alpha"] = params.pop("quantile_alpha")
        elif method == MethodName.xgboost and "alpha" in params.keys():
            params["huber_slope"] = params.pop("alpha")
    return params
