# Data Schemas

## `data/combine_pro_day.csv` / `.parquet`

The main coalesced dataset — every combine and pro day measurement from 2007 to 2026. Pro day numbers fill in where official combine numbers are missing, so this gives you the most complete picture of each prospect's athleticism.

8,463 rows as of March 2026.

| Column | Type | Description |
|--------|------|-------------|
| `Year` | int | Combine/pro day year (2007–2026) |
| `player` | string | Player full name |
| `College` | string | College/university |
| `POS_GP` | string | Position group (QB, RB, WR, TE, OL, DL, LB, DB, FB, P, LS) |
| `POS` | string | Specific position (DE, DT, CB, FS, SS, etc.) |
| `athlete_id` | float | ESPN athlete ID — primary key for joining with CFBD stats. Null for some older/obscure players. |
| `Height (in)` | float | Height in inches. Uses Daniel Jeremiah format internally (e.g. 6022 = 6'2 2/8"). |
| `Weight (lbs)` | float | Weight in pounds |
| `Arm Length (in)` | float | Arm length in inches |
| `Hand Size (in)` | float | Hand size in inches |
| `40 Yard` | float | 40-yard dash in seconds |
| `10-Yard Split` | float | 10-yard split in seconds (sparse before 2020) |
| `Vert Leap (in)` | float | Vertical jump in inches |
| `Broad Jump (in)` | float | Broad jump in inches |
| `3Cone` | float | 3-cone drill in seconds |
| `Shuttle` | float | 20-yard shuttle in seconds |
| `Bench Press` | float | Bench press reps at 225 lbs |
| `Wingspan (in)` | float | Wingspan in inches (very sparse — only 12 non-null) |
| `nfl_person_id` | string | NFL.com person UUID |

## `data/combine_official.csv` / `.parquet`

Official NFL Combine measurements only — no pro day data mixed in. Sourced directly from the NFL Combine API. Includes scouting grades and projections that aren't in the coalesced dataset.

6,883 rows as of March 2026.

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Combine year |
| `player` | string | Player full name |
| `first_name` | string | First name |
| `last_name` | string | Last name |
| `college` | string | College/university |
| `position` | string | Specific position |
| `position_group` | string | Position group |
| `height` | float | Height in inches |
| `weight` | float | Weight in pounds |
| `arm_length` | float | Arm length in inches |
| `hand_size` | float | Hand size in inches |
| `forty_yard_dash` | float | 40-yard dash in seconds |
| `ten_yard_split` | float | 10-yard split in seconds |
| `bench_press` | float | Bench press reps at 225 lbs |
| `vertical_jump` | float | Vertical jump in inches |
| `broad_jump` | float | Broad jump in inches |
| `three_cone_drill` | float | 3-cone drill in seconds |
| `twenty_yard_shuttle` | float | 20-yard shuttle in seconds |
| `person_id` | string | NFL.com person UUID |
| `grade` | float | NFL.com prospect grade (5.0–8.0 scale) |
| `draft_grade` | float | NGS draft grade (0–100) |
| `draft_projection` | string | Projected draft round (e.g. "Round 2") |
| `nfl_comparison` | string | NFL player comparison from scouts |

## Notes

- **Joining the two datasets**: Both have an NFL person ID (`nfl_person_id` / `person_id`) and player name. The coalesced dataset also has ESPN `athlete_id` for joining with CFBD stats.
- **Null handling**: Missing measurements are null/NaN, not zero. A null 40-yard dash means the player didn't run it, not that they ran a 0.00.
- **Height encoding**: Already converted to decimal inches in both files. 6'2 2/8" = 74.25 inches.
