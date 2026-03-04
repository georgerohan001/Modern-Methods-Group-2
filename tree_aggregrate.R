# --------------------------------------------------------------
# 1️⃣  Load required libraries
# --------------------------------------------------------------
library(lidR)      # point‑cloud I/O and processing
library(xml2)      # read the CVAT XML export
library(jsonlite)  # read the metadata.json created when slicing
library(data.table) # optional – faster sub‑setting

# --------------------------------------------------------------
# 2️⃣  USER SETTINGS  (change paths if you moved the files)
# --------------------------------------------------------------
las_file       <- "group_321_GP.las"                     # original point cloud
xml_file       <- "group_321_GP/annotations.xml"       # CVAT export
metadata_file  <- "tree_slices/group_321_GP/metadata.json" # per‑slice metadata
output_file    <- "tree_3d_annotated_R.las"            # result
slice_height   <- 0.20                                 # must match the slicing script

# --------------------------------------------------------------
# 3️⃣  Helper: map textual label → integer code (you can adapt)
# --------------------------------------------------------------
label_codes <- c(
  trunk  = 1L,
  branch = 2L,
  twigs  = 3L,
  grass  = 4L
)

# --------------------------------------------------------------
# 4️⃣  Load the original LAS file
# --------------------------------------------------------------
cat("[INFO] Reading LAS file ...")
las <- readLAS(las_file)

if (is.empty(las)) {
  stop("[ERROR] LAS file is empty or could not be read.")
}
cat("  (", npoints(las), "points )\n")

# -----------------------------------------------------------------
# 5️⃣  Add a new column that will hold the annotation code
# -----------------------------------------------------------------
# Give every point a default label = 0 (un‑labelled)
las@data$label_id <- 0L   # integer column, 0 = no label

# -----------------------------------------------------------------
# 6️⃣  Load the CVAT XML and the metadata JSON
# -----------------------------------------------------------------
cat("[INFO] Loading XML annotations ...\n")
cvat_doc   <- read_xml(xml_file)
image_nodes <- xml_find_all(cvat_doc, ".//image")

cat("[INFO] Loading slice metadata ...\n")
meta_raw <- fromJSON(metadata_file, simplifyVector = FALSE)

# The JSON you wrote stores one entry per slice, e.g.
#   "group_321_GP_slice_000.png" : { "x_origin_global":..., ... }
# We keep it as a named list (meta_raw) for fast lookup.
# No further processing is needed – the values are already numeric.

# -----------------------------------------------------------------
# 7️⃣  Iterate over every <image> (= slice) and its <box> elements
# -----------------------------------------------------------------
pb <- txtProgressBar(min = 0, max = length(image_nodes), style = 3)
i  <- 0

for (img_node in image_nodes) {
  i <- i + 1
  setTxtProgressBar(pb, i)
  
  # ----- 7.1  Image name (slice filename) -----
  img_name <- xml_attr(img_node, "name")   # e.g. "group_321_GP_slice_024.png"
  
  # ----- 7.2  Find matching entry in the metadata list -----
  img_meta <- meta_raw[[img_name]]
  if (is.null(img_meta)) {
    warning("[WARN] No metadata for image ", img_name, " – skipping.")
    next
  }
  
  # Global slice geometry (all slices share the same origin & pixel size)
  x0      <- img_meta$x_origin_global        # world X of pixel (0,0)
  y0      <- img_meta$y_origin_global        # world Y of pixel (0,0)
  pix_sz  <- img_meta$pixel_size            # metres per pixel
  z_low   <- img_meta$z_layer               # bottom of this slice (metres)
  z_up    <- z_low + slice_height            # top of this slice
  
  # ----- 7.3  All <box> elements inside this image -----
  boxes <- xml_find_all(img_node, ".//box")
  
  if (length(boxes) == 0L) next   # nothing to do for an empty slice
  
  for (b in boxes) {
    # ----- 7.3.1  Label -----
    lbl <- xml_attr(b, "label")
    code <- label_codes[[lbl]]
    if (is.null(code)) {
      warning("[WARN] Unknown label '", lbl, "' – ignored.")
      next
    }
    
    # ----- 7.3.2  Pixel coordinates (they are stored as strings) -----
    xtl <- as.numeric(xml_attr(b, "xtl"))
    ytl <- as.numeric(xml_attr(b, "ytl"))
    xbr <- as.numeric(xml_attr(b, "xbr"))
    ybr <- as.numeric(xml_attr(b, "ybr"))
    
    # ----- 7.3.3  Transform to world coordinates (metres) -----
    #  X_world = X_origin + pixel_x * pixel_size
    #  Y_world = Y_origin + pixel_y * pixel_size
    #  NOTE: the PNGs were saved as raster.T (i.e. X = column, Y = row) – no flip needed.
    xmin <- x0 + xtl * pix_sz
    xmax <- x0 + xbr * pix_sz
    ymin <- y0 + ytl * pix_sz
    ymax <- y0 + ybr * pix_sz
    
    # ----- 7️⃣ 4️⃣  Assign points that fall inside this 3‑D box -----
    # The box is an axis‑aligned 3‑D slab:
    #   X ∈ [xmin,xmax] , Y ∈ [ymin,ymax] , Z ∈ [z_low, z_up]
    # Use data.table for fast logical indexing.
    # (If your LAS is huge you may want to process slice‑by‑slice with `filter_poi`,
    #  but the vectorised test below works comfortably for a few million points.)
    
    idx <- which(
      las@data$Z >= z_low & las@data$Z <  z_up &
        las@data$X >= xmin  & las@data$X <= xmax  &
        las@data$Y >= ymin  & las@data$Y <= ymax
    )
    
    if (length(idx) > 0) {
      # If a point is already labelled we keep the first (higher‑priority) label.
      # You can change the logic if you want later boxes to overwrite earlier ones.
      newly_unlabelled <- which(las@data$label_id[idx] == 0L)
      if (length(newly_unlabelled) > 0) {
        las@data$label_id[idx[newly_unlabelled]] <- code
      }
    }
  } # end boxes loop
} # end images loop
close(pb)

cat("\n[INFO] Labelling finished –", sum(las@data$label_id > 0L), "points labelled.\n")

# --------------------------------------------------------------
# 8️⃣  OPTIONAL – store the label also in the standard LAS Classification
# --------------------------------------------------------------
# The LAS Classification field accepts values 1‑31 (see ASPRS spec).
# We reuse the same integer codes; you can also keep the original
# classification and store yours in the UserData column if preferred.
las@data$Classification <- las@data$label_id

# --------------------------------------------------------------
# 9️⃣  Write the new LAS file
# --------------------------------------------------------------
cat("[INFO] Writing annotated LAS to", output_file, "...\n")
writeLAS(las, output_file)

cat("[DONE] Done! You can now read", output_file,
    "with any LAS viewer (e.g. CloudCompare) and colour‑code by Classification.\n")