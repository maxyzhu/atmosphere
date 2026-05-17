import json
import requests
from atmosphere.config import get_mapillary_token
from atmosphere.retrieval.mapillary import fetch_mapillary_images

images = fetch_mapillary_images(
    lat=47.8059, lon=-122.3392, radius_m=100,
    target_count=3, download_thumbnails=False,
)
image_id = images[0].mapillary_id
print(f"Probing image_id={image_id}")

fields = ",".join([
    "id",
    "camera_parameters",
    "camera_type",
    "computed_geometry",
    "computed_altitude",
    "computed_compass_angle",
    "computed_rotation",
    "compass_angle",
    "width",
    "height",
    "atomic_scale",
])

r = requests.get(
    f"https://graph.mapillary.com/{image_id}",
    params={"fields": fields},
    headers={"Authorization": f"OAuth {get_mapillary_token()}"},
    timeout=15,
)
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))