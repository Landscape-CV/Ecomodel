#!/usr/bin/env Rscript
# Headless aRchi QSM builder for the PyTLidar benchmark.
# Usage:
#   Rscript run_archi_qsm.R input.xyz output.csv [D] [cl_dist] [max_d] [sec_length]
#
# Install (once): remotes::install_github("umr-amap/aRchi")

user_lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_lib)) {
  user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library", "4.6")
}
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_archi_qsm.R input.xyz output.csv [D] [cl_dist] [max_d] [sec_length]")
}

input_xyz <- args[[1]]
output_csv <- args[[2]]
D <- if (length(args) >= 3) as.numeric(args[[3]]) else 0.5
cl_dist <- if (length(args) >= 4) as.numeric(args[[4]]) else 0.2
max_d <- if (length(args) >= 5) as.numeric(args[[5]]) else 1.0
sec_length <- if (length(args) >= 6) as.numeric(args[[6]]) else 0.5

suppressPackageStartupMessages({
  if (!requireNamespace("aRchi", quietly = TRUE)) {
    stop("Package aRchi is not installed. Run: remotes::install_github('umr-amap/aRchi')")
  }
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package data.table is required by aRchi")
  }
  library(aRchi)
})

pts <- data.table::fread(input_xyz, header = FALSE)
if (ncol(pts) < 3) {
  stop("Input XYZ must have at least 3 columns")
}
data.table::setnames(pts, c("X", "Y", "Z"))
pts <- pts[, .(X, Y, Z)]

obj <- aRchi::build_aRchi()
obj <- aRchi::add_pointcloud(obj, point_cloud = pts)
obj <- aRchi::skeletonize_pc(obj, D = D, progressive = TRUE, cl_dist = cl_dist, max_d = max_d)
obj <- aRchi::smooth_skeleton(obj)
obj <- aRchi::add_radius(obj, sec_length = sec_length, by_axis = TRUE, method = "median")
qsm <- aRchi::get_QSM(obj)

needed <- c(
  "startX", "startY", "startZ",
  "endX", "endY", "endZ",
  "radius_cyl", "length", "branching_order"
)
missing <- setdiff(needed, names(qsm))
if (length(missing) > 0) {
  stop(paste("QSM missing columns:", paste(missing, collapse = ", ")))
}

utils::write.csv(qsm[, needed, with = FALSE], file = output_csv, row.names = FALSE)
message(sprintf("Wrote %d cylinders to %s", nrow(qsm), output_csv))
