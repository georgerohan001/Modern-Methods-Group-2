## --------------------------------------------------------------
## 0. Packages ----------------------------------------------------
## --------------------------------------------------------------
# If the packages are not yet installed, uncomment the line below:
# install.packages(c("lidR", "terra", "CspStandSegmentation"))

library(lidR)                 # point‑cloud I/O & many processing tools
library(terra)                # raster handling (used internally by lidR)
library(CspStandSegmentation) # CSP tree‑segmentation functions

## --------------------------------------------------------------
## 1. USER‑DEFINED SETTINGS --------------------------------------
## --------------------------------------------------------------
src_las   <- "group_321_GP.las"    # <-- path to your raw TLS file
out_dir   <- "segmented_trees"    # folder where the per‑tree LAS files will be written
eps_grid  <- 1.0                  # grid spacing (metres) for tree‑base detection
dtm_res   <- 0.1                  # raster resolution (metres) for DTM/CHM
min_tree_height <- 0.2           # points lower than this are discarded (optional)

## --------------------------------------------------------------
## 2. READ THE RAW POINT CLOUD ------------------------------------
## --------------------------------------------------------------
las <- readTLS(src_las)          # same as lidR::readTLS()

## --------------------------------------------------------------
## 3. CLASSIFY GROUND --------------------------------------------
## --------------------------------------------------------------
las <- classify_ground(las, csf())   # CSF works well for TLS data

## --------------------------------------------------------------
## 4. DTM (digital terrain model) -------------------------------
## --------------------------------------------------------------
dtm <- rasterize_terrain(las,
                         res = dtm_res,
                         algorithm = tin())

## --------------------------------------------------------------
## 5. NORMALISE HEIGHT -------------------------------------------
## --------------------------------------------------------------
las_norm <- las - dtm               # Z now = height above ground

## --------------------------------------------------------------
## 6. OPTIONAL: remove points that are essentially ground --------
## --------------------------------------------------------------
las_norm <- filter_poi(las_norm, Z >= min_tree_height)

## --------------------------------------------------------------
## 7. ADD THE geometric descriptors required by CSP ---------------
## --------------------------------------------------------------
# This step was missing in your original script – without it the
# segmentation algorithm prints a warning and may fail.
las_norm <- add_geometry(las_norm)   # adds V_w, L_w, S_w scalar fields

## --------------------------------------------------------------
## 8. FIND TREE‑BASE COORDINATES (raster) ------------------------
## --------------------------------------------------------------
base_map <- find_base_coordinates_raster(las_norm, eps = eps_grid)

# ---- sanity check -------------------------------------------------
if (nrow(base_map) == 0) {
  stop(paste0("No tree bases were detected (eps = ", eps_grid,
              "). Try a smaller eps or inspect the point cloud."))
} else {
  cat("Detected", nrow(base_map), "candidate tree bases.\n")
}

## --------------------------------------------------------------
## 9. CSP‑COST SEGMENTATION ---------------------------------------
## --------------------------------------------------------------
las_seg <- csp_cost_segmentation(las_norm, base_map)
# The function adds an integer column called “TreeID” to every point.

## --------------------------------------------------------------
## 10. FOREST INVENTORY (optional – you had it in the original) --
## --------------------------------------------------------------
inventory <- forest_inventory(las_seg)   # returns a data.frame, you can write it out if you like
# write.csv(inventory, file.path(out_dir, "forest_inventory.csv"), row.names = FALSE)

## --------------------------------------------------------------
## 11. POST‑PROCESSING: write ONE LAS per tree ------------------
## --------------------------------------------------------------
# --------------------------------------------------------------
# 11.1  Create output folder (if it does not exist)
# --------------------------------------------------------------
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------------------
# 11.2  Split the segmented cloud by TreeID
# --------------------------------------------------------------
# Points with TreeID == 0 are usually “noise / background”.  We drop them,
# but you can keep them by commenting the next line.
las_seg <- filter_poi(las_seg, TreeID > 0)

tree_list <- split(las_seg, las_seg$TreeID)   # a list, one element per tree

cat("Exporting", length(tree_list), "trees →", out_dir, "\n")

# --------------------------------------------------------------
# 11.3  Write each tree to its own .las file
# --------------------------------------------------------------
for (tid in names(tree_list)) {
  # Zero‑pad the ID so the files sort nicely (Tree_001.las, Tree_002.las, …)
  fname <- sprintf("Tree_%03d.las", as.integer(tid))
  writeLAS(tree_list[[tid]], file.path(out_dir, fname))
}

cat("All done! Open the folder '", out_dir,
    "' in CloudCompare and select all *.las files – each will appear as a separate layer.\n")