library(lidR)

# =========================
# Settings
# =========================
input_file <- "3dtree_404_3596_segmentation.laz"
tree_field <- "preds_instance_segmentation"

tile_dir  <- "tiles"
parts_dir <- "tree_parts"
out_dir   <- "trees_split"

dir.create(tile_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(parts_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir,   showWarnings = FALSE, recursive = TRUE)

# =========================
# 1) Read catalog metadata (safe)
# =========================
ctg <- readLAScatalog(input_file)
print(ctg)

# =========================
# 2) Retile (chunked) - tune chunk size
# =========================
# For terrestrial data this dense, smaller chunk sizes keep tiles manageable.
# Start with 100m; if you get too many tiles, go 150–200m.
opt_chunk_size(ctg)   <- 100
opt_chunk_buffer(ctg) <- 0  # buffer not needed for instance ID splitting

# Keep what you need. Add other attributes if you want them preserved.
# Example: opt_select(ctg) <- paste("xyz intensity", tree_field)
opt_select(ctg) <- paste("xyz", tree_field)

opt_output_files(ctg) <- file.path(tile_dir, "tile_{XLEFT}_{YBOTTOM}")
catalog_retile(ctg)
cat("Tiling finished.\n")

# =========================
# 3) Stage 1: write per-tile tree parts (NO appending)
# =========================
tile_files <- list.files(tile_dir, pattern = "\\.la[sz]$", full.names = TRUE)
n_tiles <- length(tile_files)
cat("Found", n_tiles, "tile files.\n")

t0 <- Sys.time()

for (i in seq_along(tile_files)) {
  f <- tile_files[i]
  cat(sprintf("[parts %d/%d] %s (elapsed %s)\n",
              i, n_tiles, basename(f), format(Sys.time() - t0)))
  
  las <- readLAS(f)
  if (is.empty(las)) next
  
  if (!(tree_field %in% names(las@data))) {
    stop(paste("Field", tree_field, "not found in tile:", f))
  }
  
  ids <- unique(las@data[[tree_field]])
  ids <- ids[!is.na(ids) & ids >= 0]
  
  # Use a stable per-tile tag for filenames
  tile_tag <- tools::file_path_sans_ext(basename(f))
  
  for (tid in ids) {
    part <- filter_poi(las, las@data[[tree_field]] == tid)
    if (is.empty(part)) next
    
    # Write a "part" file for this tree from this tile
    part_file <- file.path(parts_dir, sprintf("tree_%05d__%s.las", tid, tile_tag))
    writeLAS(part, part_file)
  }
  
  rm(las); gc()
}

cat("Stage 1 done: wrote tree parts.\n")

# =========================
# 4) Stage 2: merge parts into one LAS per tree
# =========================
part_files <- list.files(parts_dir, pattern = "^tree_\\d+__.*\\.las$", full.names = TRUE)

# Extract tree IDs from filenames
tree_ids <- sub("^tree_(\\d{5})__.*$", "\\1", basename(part_files))
tree_ids <- unique(tree_ids)

cat("Merging", length(tree_ids), "trees...\n")
t1 <- Sys.time()

for (j in seq_along(tree_ids)) {
  tid_str <- tree_ids[j]
  tid <- as.integer(tid_str)
  
  files_for_tree <- part_files[grepl(paste0("^tree_", tid_str, "__"), basename(part_files))]
  
  cat(sprintf("[merge %d/%d] tree %s (%d parts) (elapsed %s)\n",
              j, length(tree_ids), tid_str, length(files_for_tree),
              format(Sys.time() - t1)))
  
  # Read and combine all parts
  las_list <- lapply(files_for_tree, readLAS)
  las_list <- Filter(function(x) !is.empty(x), las_list)
  if (length(las_list) == 0) next
  
  merged <- do.call(rbind, las_list)
  
  out_file <- file.path(out_dir, sprintf("tree_%05d.las", tid))
  writeLAS(merged, out_file)
  
  # Optional: delete parts after successful merge to save space
  # file.remove(files_for_tree)
  
  rm(las_list, merged); gc()
}

cat("All done. Output in:", out_dir, "\n")