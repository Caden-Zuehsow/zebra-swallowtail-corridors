# Zebra Swallowtail Corridor Modeling

This project models potential movement corridors for
**Zebra Swallowtail butterflies (Eurytides marcellus)**
based on:

- Pawpaw (Asimina triloba) host plant distribution
- Riparian / river navigation corridors
- Landscape resistance (land cover)

## Hypothesis

Zebra swallowtails use a combination of **pawpaw groves**
and **riparian zones** as navigation corridors, which
structure population connectivity and distribution.

## Outputs

- `resistance.tif` — movement cost surface
- `corridors.tif` — corridor intensity raster
- `corridors.shp` — vectorized corridor polygons

All outputs are GIS-ready (QGIS / ArcGIS).

## Requirements

```bash
pip install -r requirements.txt
