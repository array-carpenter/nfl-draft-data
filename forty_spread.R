library(ggplot2)
library(ggbeeswarm)
library(dplyr)
library(showtext)
library(magick)
library(paletteer)

font_add_google("Karla", "karla")
showtext_auto()
showtext_opts(dpi = 300)

theme_spade <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      text = element_text(family = "karla", color = "#333333"),
      plot.background = element_rect(fill = "#DDEBEC", color = NA),
      panel.background = element_rect(fill = "#DDEBEC", color = NA),
      plot.title = element_text(face = "bold", size = 36, margin = margin(b = 5)),
      plot.title.position = "plot",
      plot.subtitle = element_text(size = 22, color = "#666666", margin = margin(b = 15)),
      plot.caption = element_text(size = 18, color = "#888888", hjust = 0),
      plot.caption.position = "plot",
      panel.grid.major = element_line(color = "#cccccc", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      legend.background = element_rect(fill = "#DDEBEC", color = NA),
      legend.key = element_rect(fill = "#DDEBEC", color = NA),
      plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
    )
}

# Okabe-Ito from paletteer (colorblind-safe) + 3 extensions
oi <- paletteer_d("colorblindr::OkabeIto")
pos_colors <- c(
  "DB"  = oi[6],   # blue
  "WR"  = oi[2],   # orange
  "RB"  = oi[3],   # green
  "LB"  = oi[5],   # vermillion
  "TE"  = oi[7],   # pink
  "FB"  = oi[4],   # yellow
  "QB"  = oi[8],   # sky blue
  "P"   = oi[1],   # black
  "DL"  = "#E83562",
  "LS"  = "#7A5195",
  "OL"  = "#1AFF1A"
)

df <- read.csv("combine_data_unique_athlete_id_step4.csv") |>
  filter(!is.na(X40.Yard), X40.Yard >= 4.2, X40.Yard <= 5.5) |>
  mutate(POS_GP = if_else(POS_GP %in% c("EDGE", "DT"), "DL", POS_GP)) |>
  filter(POS_GP %in% names(pos_colors)) |>
  mutate(POS_GP = factor(POS_GP, levels = names(pos_colors)))

n_obs <- nrow(df)

p <- ggplot(df, aes(x = X40.Yard, y = 0, color = POS_GP)) +
  geom_quasirandom(
    groupOnX = TRUE, width = 0.4,
    size = 1, alpha = 0.6
  ) +
  scale_color_manual(values = pos_colors, name = NULL) +
  labs(
    title = "NFL Combine & Pro Day 40-Yard Dash Times by Position",
    subtitle = paste0("Every recorded 40-yard dash from 2007-2026 (n = ", format(n_obs, big.mark = ","), ")"),
    x = "40-Yard Dash (seconds)",
    y = NULL,
    caption = "Ray Carpenter | TheSpade.Substack.com | @csv_enjoyer | data: Various sources"
  ) +
  scale_x_continuous(breaks = seq(4.2, 5.4, 0.2)) +
  theme_spade() +
  theme(
    axis.text.x = element_text(size = 18),
    axis.title.x = element_text(size = 20),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position = "top",
    legend.text = element_text(size = 16, face = "bold"),
    legend.key.size = unit(1.2, "lines")
  ) +
  guides(color = guide_legend(nrow = 1, override.aes = list(size = 4, alpha = 1)))

tmp <- tempfile(fileext = ".png")
ggsave(tmp, plot = p, width = 16, height = 9, dpi = 300, bg = "#DDEBEC")

plot_img <- image_read(tmp)
plot_info <- image_info(plot_img)

logo <- image_read("/Users/raycarpenter/Documents/GitHub/nfl-assets/spade_logo.png") |>
  image_trim() |>
  image_scale(geometry_size_pixels(height = round(plot_info$height * 0.05)))

logo_info <- image_info(logo)
margin_px <- round(20 / 72 * 300)  # 20pt margin at 300dpi
x_offset <- plot_info$width - logo_info$width - margin_px
y_offset <- margin_px

final <- image_composite(plot_img, logo, offset = paste0("+", x_offset, "+", y_offset))
image_write(final, "forty_spread.png")
cat("Saved forty_spread.png\n")
