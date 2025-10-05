# Run WebApp
$ cd Zephyro
$ python app.py

# Zephyro
Zephyro è un’app web per l’analisi e la visualizzazione della qualità dell’aria in tempo reale, basata sui dati satellitari NASA TEMPO e altre fonti ambientali.

Funzionalità API
| Parametro      |   Tipo | Obbl. | Default  | Descrizione                                                                                                                           |
| -------------- | -----: | :---: | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `bbox`         | string |   ✅   | —        | Bounding box `minLon,minLat,maxLon,maxLat` (es. `-125,24,-66,49`).                                                                    |
| `auto_step`    |   bool |   ❌   | `1`      | Se attivo, calcola **automaticamente** lo step (lato cella, in gradi) per ottenere ~`target_cells` celle. Se `0`, usa `step` manuale. |
| `target_cells` |    int |   ❌   | `900`    | Celle “target” quando `auto_step=1`. Più alto ⇒ **più dettaglio**.                                                                    |
| `step`         |  float |   ❌   | —        | Lato cella (in gradi). Usato solo se `auto_step=0` o lo forzi.                                                                        |
| `min_aqi`      |  float |   ❌   | `51`     | **Filtro** lato server: esclude AQI < `min_aqi` (di default mostra **dal giallo in su**).                                             |
| `agg`          | string |   ❌   | `max`    | Aggregazione nella cella: `max` (conservativo) o `mean`.                                                                              |
| `parameters`   | string |   ❌   | dinamico | Inquinanti AirNow (CSV, es. `PM25,O3`). Su bbox grandi default `PM25`, altrimenti `PM25,O3`. Se lo passi, non viene sovrascritto.     |
| `dataType`     | string |   ❌   | `A`      | Tipo AirNow `/aq/data`: `A` (aggregato) o `B` (raw, pesante).                                                                         |
| `startDate`    | string |   ❌   | —        | Inizio (UTC), `YYYY-MM-DDTHH`.                                                                                                        |
| `endDate`      | string |   ❌   | —        | Fine (UTC), `YYYY-MM-DDTHH`.                                                                                                          |
| `hours`        |    int |   ❌   | adattivo | Usato **solo** se mancano `startDate`/`endDate`. Finestra recente in ore, scelta in base alla grandezza del bbox (1–24).              |

Comportamenti adattivi (se non forzati)

Step: calcolato per ottenere ~target_cells celle nel bbox (con limiti min/max).

Tempo: finestra recente scelta in base al bbox:

BBOX molto grande ⇒ 1h

Macro-regione ⇒ 3h

Stato/area grande ⇒ 6h

Città ⇒ 24h

Parameters: default PM25 su aree enormi, PM25,O3 sulle altre.
