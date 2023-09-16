from latentis.relative.projection import Projections, RelativeProjector
from latentis.transforms import Transforms
from torch import nn

PROJECTION_NAME2PROJECTION = {
    "cosine": Projections.COSINE,
    None: None,
}

TRANSFORM_NAME2TRANSFORM = {
    "centering": Transforms.Centering(),
    None: None,
}


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
