#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(reticulate)
})

KEY_SEPARATOR <- "__"
METADATA_KEY <- "__metadata_json__"
DATASET_URLS <- c(
  kluger_census_income_strat = paste0(
    "https://raw.githubusercontent.com/DanKluger/",
    "Predict-Then-Debias_Bootstrap/main/Datasets/",
    "CensusStratIncomeFormatted.RData"
  ),
  kluger_housing_price = paste0(
    "https://raw.githubusercontent.com/DanKluger/",
    "Predict-Then-Debias_Bootstrap/main/Datasets/",
    "HousingPriceFormatted.RData"
  ),
  kluger_treecover = paste0(
    "https://raw.githubusercontent.com/DanKluger/",
    "Predict-Then-Debias_Bootstrap/main/Datasets/",
    "TreecoverFormatted.RData" 
  )
)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) {
    y
  } else {
    x
  }
}

get_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    raw_path <- sub("^--file=", "", file_arg[[1]])
    raw_path <- gsub("~\\+~", " ", raw_path, fixed = FALSE)
    return(normalizePath(raw_path, winslash = "/", mustWork = FALSE))
  }

  frame_paths <- vapply(
    sys.frames(),
    function(frame) {
      ofile <- frame$ofile %||% ""
      if (length(ofile) == 0) {
        ""
      } else {
        as.character(ofile[[1]])
      }
    },
    character(1)
  )
  frame_paths <- frame_paths[nzchar(frame_paths)]
  if (length(frame_paths) > 0) {
    return(normalizePath(tail(frame_paths, 1), winslash = "/", mustWork = FALSE))
  }

  candidates <- c(
    file.path(getwd(), "examples", "prepare_kluger_data.R"),
    file.path(getwd(), "prepare_kluger_data.R")
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing) > 0) {
    return(normalizePath(existing[[1]], winslash = "/", mustWork = FALSE))
  }

  NA_character_
}

SCRIPT_PATH <- get_script_path()
REPO_ROOT <- if (!is.na(SCRIPT_PATH) && nzchar(SCRIPT_PATH)) {
  normalizePath(file.path(dirname(SCRIPT_PATH), ".."), winslash = "/", mustWork = FALSE)
} else if (basename(getwd()) == "examples") {
  normalizePath(file.path(getwd(), ".."), winslash = "/", mustWork = FALSE)
} else {
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
DEFAULT_OUTPUT_DIR <- file.path(REPO_ROOT, "examples", "data")

print_help <- function() {
  cat(
    paste(
      "Usage: Rscript examples/prepare_kluger_data.R [options]",
      "",
      "Options:",
      "  --datasets NAME [NAME ...]  Subset of datasets to process.",
      sprintf("                             Valid names: %s", paste(names(DATASET_URLS), collapse = ", ")),
      "  --output-dir PATH           Directory where output NPZ files will be written.",
      "  --overwrite                 Overwrite existing NPZ files instead of skipping them.",
      "  -h, --help                  Show this message and exit.",
      sep = "\n"
    )
  )
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  parsed <- list(
    datasets = names(DATASET_URLS),
    output_dir = DEFAULT_OUTPUT_DIR,
    overwrite = FALSE
  )

  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      print_help()
      quit(save = "no", status = 0)
    }

    if (arg == "--datasets") {
      i <- i + 1
      if (i > length(args) || startsWith(args[[i]], "--")) {
        stop("--datasets requires at least one dataset name.")
      }
      datasets <- character()
      while (i <= length(args) && !startsWith(args[[i]], "--")) {
        datasets <- c(datasets, args[[i]])
        i <- i + 1
      }
      parsed$datasets <- datasets
      next
    }

    if (arg == "--output-dir") {
      i <- i + 1
      if (i > length(args)) {
        stop("--output-dir requires a value.")
      }
      parsed$output_dir <- args[[i]]
      i <- i + 1
      next
    }

    if (arg == "--overwrite") {
      parsed$overwrite <- TRUE
      i <- i + 1
      next
    }

    stop(sprintf("Unknown argument: %s", arg))
  }

  unknown <- setdiff(parsed$datasets, names(DATASET_URLS))
  if (length(unknown) > 0) {
    stop(sprintf("Unknown dataset(s): %s", paste(unknown, collapse = ", ")))
  }

  parsed
}

sanitize_key_part <- function(part) {
  raw <- trimws(as.character(part %||% ""))
  if (!nzchar(raw)) {
    raw <- "item"
  }
  sanitized <- gsub("[^[:alnum:]_.]+", "_", raw)
  sanitized <- gsub("^_+|_+$", "", sanitized)
  if (!nzchar(sanitized)) {
    "item"
  } else {
    sanitized
  }
}

make_unique_names <- function(parts) {
  used <- character()
  out <- character(length(parts))
  for (index in seq_along(parts)) {
    base <- sanitize_key_part(parts[[index]])
    candidate <- base
    suffix <- 1
    while (candidate %in% used) {
      candidate <- sprintf("%s_%d", base, suffix)
      suffix <- suffix + 1
    }
    used <- c(used, candidate)
    out[[index]] <- candidate
  }
  out
}

path_to_key <- function(path_parts) {
  paste(path_parts, collapse = KEY_SEPARATOR)
}

infer_numpy_dtype <- function(value) {
  if (is.null(value)) {
    return("<U1")
  }
  if (is.factor(value) || is.character(value) || inherits(value, c("Date", "POSIXct", "POSIXlt"))) {
    return("<U")
  }
  if (is.logical(value)) {
    if (anyNA(value)) {
      return("float64")
    }
    return("bool")
  }
  if (is.integer(value)) {
    if (anyNA(value)) {
      return("float64")
    }
    return("int32")
  }
  if (is.numeric(value)) {
    return("float64")
  }
  if (is.complex(value)) {
    return("complex128")
  }
  typeof(value)
}

scalar_summary <- function(key, value, source_type, from_dataframe_column = FALSE) {
  dims <- dim(value)
  shape <- if (is.null(dims)) list(as.integer(length(value))) else as.list(as.integer(dims))
  summary_kind <- if (length(value) == 1 && is.null(dim(value))) "scalar" else "vector"
  list(
    path = key,
    type = source_type,
    summary_kind = summary_kind,
    shape = shape,
    dtype = infer_numpy_dtype(value),
    length = as.integer(length(value)),
    from_dataframe_column = from_dataframe_column
  )
}

normalize_datetime <- function(value) {
  out <- format(as.POSIXct(value, tz = "UTC"), tz = "UTC", usetz = TRUE)
  out[is.na(out)] <- ""
  out
}

normalize_atomic <- function(value, path) {
  dims <- dim(value)
  if (is.null(value)) {
    return("")
  }
  if (is.factor(value)) {
    out <- as.character(value)
    out[is.na(out)] <- ""
    if (!is.null(dims)) {
      dim(out) <- dims
    }
    return(out)
  }
  if (inherits(value, "Date")) {
    out <- as.character(value)
    out[is.na(out)] <- ""
    if (!is.null(dims)) {
      dim(out) <- dims
    }
    return(out)
  }
  if (inherits(value, c("POSIXct", "POSIXlt"))) {
    out <- normalize_datetime(value)
    if (!is.null(dims)) {
      dim(out) <- dims
    }
    return(out)
  }
  if (inherits(value, "difftime")) {
    out <- as.numeric(value)
    if (!is.null(dims)) {
      dim(out) <- dims
    }
    return(out)
  }
  if (is.character(value)) {
    out <- value
    out[is.na(out)] <- ""
    if (!is.null(dims)) {
      dim(out) <- dims
    }
    return(out)
  }
  if (is.logical(value)) {
    out <- if (anyNA(value)) as.numeric(value) else value
    if (!is.null(dims)) {
      dim(out) <- dims
    }
    return(out)
  }
  if (is.integer(value)) {
    out <- if (anyNA(value)) as.numeric(value) else value
    if (!is.null(dims)) {
      dim(out) <- dims
    }
    return(out)
  }
  if (is.numeric(value) || is.complex(value)) {
    out <- value
    if (!is.null(dims)) {
      dim(out) <- dims
    }
    return(out)
  }

  stop(sprintf(
    "Unsupported atomic object at %s: %s",
    path,
    paste(class(value), collapse = ", ")
  ))
}

is_scalar_like <- function(value) {
  if (is.null(value)) {
    return(TRUE)
  }
  if (is.data.frame(value) || is.list(value)) {
    return(FALSE)
  }
  is.atomic(value) && is.null(dim(value)) && length(value) == 1
}

scalar_list_to_vector <- function(values, path) {
  non_null <- values[!vapply(values, is.null, logical(1))]
  if (length(non_null) == 0) {
    return("")
  }

  if (all(vapply(non_null, function(x) is.factor(x) || is.character(x) || inherits(x, c("Date", "POSIXct", "POSIXlt")), logical(1)))) {
    out <- vapply(values, function(x) {
      if (is.null(x) || is.na(x)) {
        ""
      } else if (inherits(x, c("POSIXct", "POSIXlt"))) {
        normalize_datetime(x)
      } else {
        as.character(x)
      }
    }, character(1))
    return(out)
  }

  if (all(vapply(non_null, is.logical, logical(1)))) {
    if (any(vapply(values, function(x) is.null(x) || is.na(x), logical(1)))) {
      return(vapply(values, function(x) if (is.null(x) || is.na(x)) NA_real_ else as.numeric(x), numeric(1)))
    }
    return(vapply(values, as.logical, logical(1)))
  }

  if (all(vapply(non_null, is.integer, logical(1)))) {
    if (any(vapply(values, function(x) is.null(x) || is.na(x), logical(1)))) {
      return(vapply(values, function(x) if (is.null(x) || is.na(x)) NA_real_ else as.numeric(x), numeric(1)))
    }
    return(vapply(values, as.integer, integer(1)))
  }

  if (all(vapply(non_null, function(x) is.numeric(x) || is.integer(x), logical(1)))) {
    return(vapply(values, function(x) if (is.null(x) || is.na(x)) NA_real_ else as.numeric(x), numeric(1)))
  }

  stop(sprintf("Unsupported scalar list at %s.", path))
}

add_leaf <- function(key, value, source_type, payload, metadata_objects, from_dataframe_column = FALSE) {
  payload[[key]] <- value
  metadata_objects[[length(metadata_objects) + 1]] <- scalar_summary(
    key,
    value,
    source_type,
    from_dataframe_column = from_dataframe_column
  )
  list(payload = payload, metadata_objects = metadata_objects)
}

flatten_object <- function(obj, path_parts, payload, metadata_objects) {
  key <- path_to_key(path_parts)

  if (is.data.frame(obj)) {
    metadata_objects[[length(metadata_objects) + 1]] <- list(
      path = key,
      type = paste(class(obj), collapse = ","),
      summary_kind = "table",
      rows = as.integer(nrow(obj)),
      cols = as.integer(ncol(obj)),
      columns = as.list(colnames(obj))
    )

    child_names <- make_unique_names(colnames(obj))
    for (index in seq_along(obj)) {
      child_key <- path_to_key(c(path_parts, child_names[[index]]))
      column_value <- normalize_atomic(obj[[index]], child_key)
      added <- add_leaf(
        child_key,
        column_value,
        paste(class(obj[[index]]), collapse = ","),
        payload,
        metadata_objects,
        from_dataframe_column = TRUE
      )
      payload <- added$payload
      metadata_objects <- added$metadata_objects
    }
    return(list(payload = payload, metadata_objects = metadata_objects))
  }

  if (is.null(obj) || is.atomic(obj)) {
    normalized <- normalize_atomic(obj, key)
    return(add_leaf(key, normalized, paste(class(obj), collapse = ","), payload, metadata_objects))
  }

  if (is.list(obj)) {
    names_vec <- names(obj)
    if (length(obj) == 0) {
      return(add_leaf(key, "", paste(class(obj), collapse = ","), payload, metadata_objects))
    }

    if (all(vapply(obj, is_scalar_like, logical(1)))) {
      normalized <- scalar_list_to_vector(obj, key)
      return(add_leaf(key, normalized, paste(class(obj), collapse = ","), payload, metadata_objects))
    }

    metadata_objects[[length(metadata_objects) + 1]] <- list(
      path = key,
      type = paste(class(obj), collapse = ","),
      summary_kind = "container",
      entries = as.integer(length(obj))
    )

    raw_names <- if (is.null(names_vec)) {
      sprintf("item_%d", seq_along(obj) - 1L)
    } else {
      ifelse(names_vec == "" | is.na(names_vec), sprintf("item_%d", seq_along(obj) - 1L), names_vec)
    }
    child_names <- make_unique_names(raw_names)
    for (index in seq_along(obj)) {
      flattened <- flatten_object(
        obj[[index]],
        c(path_parts, child_names[[index]]),
        payload,
        metadata_objects
      )
      payload <- flattened$payload
      metadata_objects <- flattened$metadata_objects
    }
    return(list(payload = payload, metadata_objects = metadata_objects))
  }

  stop(sprintf(
    "Unsupported object at %s: %s",
    key,
    paste(class(obj), collapse = ", ")
  ))
}

build_payload <- function(dataset_name, source_url, top_level_objects) {
  payload <- list()
  metadata_objects <- list()
  top_level_names <- names(top_level_objects)
  if (is.null(top_level_names)) {
    stop(sprintf("Top-level objects for %s must be named.", dataset_name))
  }

  unique_names <- make_unique_names(top_level_names)
  for (index in seq_along(top_level_objects)) {
    flattened <- flatten_object(
      top_level_objects[[index]],
      unique_names[[index]],
      payload,
      metadata_objects
    )
    payload <- flattened$payload
    metadata_objects <- flattened$metadata_objects
  }

  metadata <- list(
    dataset_name = dataset_name,
    source_url = source_url,
    top_level_objects = as.list(top_level_names),
    objects = metadata_objects
  )

  payload[[METADATA_KEY]] <- toJSON(metadata, auto_unbox = TRUE, null = "null", na = "null")
  list(payload = payload, metadata = metadata)
}

print_report <- function(metadata, output_path) {
  cat(sprintf("\n=== %s ===\n", metadata$dataset_name))
  cat(sprintf("Source: %s\n", metadata$source_url))
  cat(sprintf("Top-level objects: %s\n", paste(unlist(metadata$top_level_objects), collapse = ", ")))

  tables <- Filter(function(obj) identical(obj$summary_kind, "table"), metadata$objects)
  leaves <- Filter(
    function(obj) {
      obj$summary_kind %in% c("scalar", "vector") && !isTRUE(obj$from_dataframe_column)
    },
    metadata$objects
  )

  if (length(tables) > 0) {
    cat("Tabular objects:\n")
    for (table in tables) {
      cat(sprintf(
        "  - %s: rows=%s, cols=%s, columns=%s\n",
        table$path,
        table$rows,
        table$cols,
        paste(unlist(table$columns), collapse = ", ")
      ))
    }
  }

  if (length(leaves) > 0) {
    cat("Vectors/scalars:\n")
    for (leaf in leaves) {
      cat(sprintf(
        "  - %s: shape=(%s), dtype=%s\n",
        leaf$path,
        paste(unlist(leaf$shape), collapse = ", "),
        leaf$dtype
      ))
    }
  }

  cat(sprintf("Output: %s\n", output_path))
}

save_npz <- function(output_path, payload) {
  np <- import("numpy", convert = FALSE)
  args <- c(list(file = output_path), payload)
  do.call(np$savez_compressed, args)
}

process_dataset <- function(dataset_name, source_url, output_dir, overwrite) {
  output_path <- file.path(output_dir, sprintf("%s.npz", dataset_name))
  if (file.exists(output_path) && !overwrite) {
    cat(sprintf("Skipping %s: %s already exists.\n", dataset_name, output_path))
    return(output_path)
  }

  temp_dir <- tempfile(pattern = paste0(dataset_name, "_"))
  dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(temp_dir, recursive = TRUE, force = TRUE), add = TRUE)

  download_path <- file.path(temp_dir, basename(source_url))
  utils::download.file(source_url, download_path, mode = "wb", quiet = TRUE)

  load_env <- new.env(parent = emptyenv())
  loaded_names <- load(download_path, envir = load_env)
  top_level_objects <- mget(loaded_names, envir = load_env, inherits = FALSE)

  built <- build_payload(dataset_name, source_url, top_level_objects)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  save_npz(output_path, built$payload)
  print_report(built$metadata, output_path)
  output_path
}

main <- function() {
  args <- parse_args()
  failures <- list()

  for (dataset_name in args$datasets) {
    source_url <- DATASET_URLS[[dataset_name]]
    tryCatch(
      {
        process_dataset(dataset_name, source_url, args$output_dir, args$overwrite)
      },
      error = function(err) {
        failures[[dataset_name]] <<- conditionMessage(err)
        message(sprintf("[ERROR] %s: %s", dataset_name, conditionMessage(err)))
      }
    )
  }

  if (length(failures) > 0) {
    message(sprintf("Failed datasets: %s", paste(names(failures), collapse = ", ")))
    quit(save = "no", status = 1)
  }
}

if (sys.nframe() == 0) {
  main()
}
