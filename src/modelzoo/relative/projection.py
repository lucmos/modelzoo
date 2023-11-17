import functools
import logging
from typing import Callable, Optional, Sequence, Tuple, Type

import torch
from deprecated import deprecated
from latentis.relative.projection import Projections, RelativeProjector
from latentis.transforms import Transforms
from torch import nn

pylogger = logging.getLogger(__name__)

PROJECTION_NAME2PROJECTION = {
    "cosine": Projections.COSINE,
    None: None,
}

TRANSFORM_NAME2TRANSFORM = {
    "centering": Transforms.Centering(),
    None: None,
}


@deprecated(version="0.1.0", reason="Use RelativeBlock instead")
class RelativeModule(nn.Module):
    def __init__(self, projection, abs_transforms, rel_transforms):
        super().__init__()
        projection = PROJECTION_NAME2PROJECTION[projection]

        if abs_transforms is not None:
            if not isinstance(abs_transforms, list):
                abs_transforms = [abs_transforms]
            abs_transforms = [TRANSFORM_NAME2TRANSFORM[transform] for transform in abs_transforms]

        if rel_transforms is not None:
            if not isinstance(rel_transforms, list):
                rel_transforms = [rel_transforms]
            rel_transforms = [TRANSFORM_NAME2TRANSFORM[transform] for transform in rel_transforms]

        self.relative_projector = RelativeProjector(
            projection=projection,
            abs_transforms=abs_transforms,
            rel_transforms=rel_transforms,
        )

    def forward(self, x, anchors):
        return self.relative_projector(x=x, anchors=anchors)



def abs_to_rel(
    anchors: torch.Tensor,
    points: torch.Tensor,
    normalizing_func: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
    dist_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    relative_points = []

    if normalizing_func is not None:
        anchors, points = normalizing_func(anchors=anchors, points=points)

    for point in points:
        current_rel_point = []
        for anchor in anchors:
            current_rel_point.append(dist_func(point=point, anchor=anchor))
        relative_points.append(current_rel_point)
    return torch.as_tensor(relative_points, dtype=anchors.dtype)



def abs_to_rel_cosine(
    anchors: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    norm_anchors = torch.nn.functional.normalize(anchors, dim=-1)
    norm_points = torch.nn.functional.normalize(points, dim=-1)

    return norm_points @ norm_anchors.T



def abs_to_rel_lp(
    anchors: torch.Tensor,
    points: torch.Tensor,
    p: int,
) -> torch.Tensor:
    return torch.cdist(points, anchors, p=p)




SIMPLE_PROJECTION_TYPE = {
    "Cosine": abs_to_rel_cosine,
    "Euclidean": functools.partial(abs_to_rel_lp, p=2),
    "L1": functools.partial(abs_to_rel_lp, p=1),
    "Linf": functools.partial(abs_to_rel_lp, p=torch.inf),
    "Absolute": lambda points, **kwargs: points,
}



class RelativeBlock(torch.nn.Module):
    def __init__(
        self,
        projection_names: Sequence[str],
        aggregation_module: torch.nn.Module,
    ):
        super().__init__()
        self.projection_names = projection_names
        self.projection_funcs: Sequence[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = [
            SIMPLE_PROJECTION_TYPE[x] for x in projection_names
        ]
        self.aggregation_module = aggregation_module

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        pylogger.warning("RelativeBlock used in forward pass! If you want to use it in training, use encode and decode!")
        rel_x = self.encode(x, anchors)
        return self.decode(rel_x)

    def encode(self, x, anchors) -> torch.Tensor:
        return torch.cat([fun(anchors=anchors, points=x) for fun in self.projection_funcs], dim=-1)

    def decode(self, rel_x: torch.Tensor) -> torch.Tensor:
        return self.aggregation_module(rel_x)

    def __repr__(self):
        return f"RelativeBlock({self.projection_names}, {self.aggregation_module})"

    def __str__(self):
        return f"RelativeBlock({self.projection_names=}, {self.aggregation_module=})"
