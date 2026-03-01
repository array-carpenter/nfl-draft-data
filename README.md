# NFL Draft Combine & Pro Day Data

Since 2022 I've been manually charting NFL Draft Combine and Pro Day data because I can't find a uniform source online that is fast. NFL Combine and Pro Day measurements for every prospect from 2007 to 2026. Two datasets: a coalesced version that fills in pro day numbers where official combine data is missing, and a combine-only version with scouting grades from NFL.com.

## Datasets

| File | Rows | Description |
|------|------|-------------|
| `data/combine_pro_day.csv` | 8,463 | Combine + pro day measurements (most complete) |
| `data/combine_pro_day.parquet` | 8,463 | Same data, Parquet format |
| `data/combine_official.csv` | 6,883 | Official combine only, includes scouting grades |
| `data/combine_official.parquet` | 6,883 | Same data, Parquet format |

Both datasets include height, weight, arm length, hand size, 40-yard dash, vertical jump, broad jump, 3-cone, shuttle, and bench press. The official dataset adds NFL.com prospect grades, NGS draft grades, draft projections, and player comparisons.

Full column descriptions in [docs/SCHEMAS.md](docs/SCHEMAS.md).

## Quick Start

```python
import pandas as pd

# Load the coalesced dataset (combine + pro day)
df = pd.read_parquet("data/combine_pro_day.parquet")

# All WRs who ran sub-4.4
fast_wrs = df[(df["POS_GP"] == "WR") & (df["40 Yard"] < 4.4)]
```

Or in R:

```r
library(arrow)
df <- read_parquet("data/combine_pro_day.parquet")
```

## Sources

- NFL Combine API (official measurements, grades, projections)
- Pro day results from various public sources
- ESPN athlete IDs for cross-referencing with CFBD stats
- NFLCombineResults.com

Please go ahead and star, fork, share, and reach out if you see anything out of the ordinary. And please send over anything interesting you create with this data, I'd love to see it. Thanks - RC