# Run WebApp
$ cd Zephyro

$ pip install -r requirements.txt

$ python app.py

# Zephyro
Zephyro is a web application for real-time air quality analysis and visualization, based on NASA TEMPO satellite data and other environmental sources.
API Features
| Parameter      | Type   | Req. | Default  | Description                                                                                                                                   |
| -------------- | ------ | :--: | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `bbox`         | string |   ✅  | —        | Bounding box `minLon,minLat,maxLon,maxLat` (e.g. `-125,24,-66,49`).                                                                           |
| `auto_step`    | bool   |   ❌  | `1`      | If enabled, **automatically** calculates the step (cell side length, in degrees) to obtain ~`target_cells` cells. If `0`, uses manual `step`. |
| `target_cells` | int    |   ❌  | `900`    | Target number of cells when `auto_step=1`. Higher ⇒ **more detail**.                                                                          |
| `step`         | float  |   ❌  | —        | Cell side length (in degrees). Used only if `auto_step=0` or you force it.                                                                    |
| `min_aqi`      | float  |   ❌  | `51`     | **Server-side filter:** excludes AQI < `min_aqi` (by default shows **yellow and above**).                                                     |
| `agg`          | string |   ❌  | `max`    | Cell aggregation: `max` (conservative) or `mean`.                                                                                             |
| `parameters`   | string |   ❌  | dynamic  | AirNow pollutants (CSV, e.g. `PM25,O3`). For large bbox default is `PM25`; otherwise `PM25,O3`. If provided, it won’t be overridden.          |
| `dataType`     | string |   ❌  | `A`      | AirNow `/aq/data` type: `A` (aggregated) or `B` (raw, heavy).                                                                                 |
| `startDate`    | string |   ❌  | —        | Start time (UTC), `YYYY-MM-DDTHH`.                                                                                                            |
| `endDate`      | string |   ❌  | —        | End time (UTC), `YYYY-MM-DDTHH`.                                                                                                              |
| `hours`        | int    |   ❌  | adaptive | Used **only** if `startDate`/`endDate` are missing. Recent time window in hours, chosen based on bbox size (1–24).                            |


# Adaptive Behaviors (if not forced)

Step: calculated to get ~target_cells cells in the bbox (with min/max limits).

Time window: recent time range chosen based on bbox size:

Very large BBOX ⇒ 1h

Macro-region ⇒ 3h

Large state/area ⇒ 6h

City ⇒ 24h

Parameters: default PM25 for huge areas, PM25,O3 for smaller ones.
