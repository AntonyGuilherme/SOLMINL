# Source the main functions required by generateCOP
source("generateCOP.R")
source("LinearProg.R") # Assuming LinearProg.R is in the same directory
source("Zvalue.R")     # Assuming Zvalue.R is in the same directory

# Define fixed parameters as per your request
n_param <- 10
m_param <- 20
G_param <- "max"
typeDistance_param <- "K"

# Define the folder where your input files (permutations, distances, thetas) are located
input_files_folder <- "./../data/" 
output_dir <- "./../data/expected_outputs/" 

# Create the output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# --- Dynamic determination of instance numbers and types ---
# List all files in the input folder
all_files <- list.files(path = input_files_folder, full.names = FALSE)

# Filter for files that match the pattern "permutation_TYPE_X.txt"
# This pattern is used to identify unique instance types (easy, difficult, similar) and numbers (X)
permutation_files <- grep("^permutation_(easy|difficult)[0-9]+\\.txt$", all_files, value = TRUE)

# Extract the unique (type, number) pairs from the filenames
instance_info <- data.frame(
  filename = permutation_files,
  type = gsub("permutation_([a-z]+)[0-9]+\\.txt", "\\1", permutation_files),
  number = as.numeric(gsub("permutation_[a-z]+([0-9]+)\\.txt", "\\1", permutation_files))
)

# Get unique combinations of type and number
unique_instances <- unique(instance_info[, c("type", "number")])

# Sort the instances for consistent processing order (e.g., by type then by number)
unique_instances <- unique_instances[order(unique_instances$type, unique_instances$number), ]

if (nrow(unique_instances) == 0) {
  cat("No 'permutation_TYPE_X.txt' files (easy, difficult, similar) found in the specified folder.\n")
  cat("Please ensure input files are in:", input_files_folder, "\n")
} else {
  cat(sprintf("Found %d unique instances to process.\n", nrow(unique_instances)))
  cat("Processing instances:\n")
  for(k in 1:nrow(unique_instances)) {
    cat(sprintf("  - Type: %s, Number: %d\n", unique_instances$type[k], unique_instances$number[k]))
  }
  cat("\n")

  # Loop through the dynamically determined instance types and numbers
  for (k in 1:nrow(unique_instances)) {
    current_type <- unique_instances$type[k]
    current_number <- unique_instances$number[k]

    # Construct the input file paths dynamically
    file_sigma <- file.path(input_files_folder, paste0("permutation_", current_type, current_number, ".txt"))
    file_distances <- file.path(input_files_folder, paste0("permutation_", current_type, current_number, "_distance.txt"))
    file_theta <- file.path(input_files_folder, paste0("permutation_", current_type, current_number, "_theta.txt"))
    
    # Construct the output file path
    file_out <- file.path(output_dir, paste0(current_type, current_number, ".txt"))
    
    cat(sprintf("--- Calling generateCOP for instance Type: %s, Number: %d ---\n", current_type, current_number))
    cat(sprintf("  FileSigma: %s\n", file_sigma))
    cat(sprintf("  FileDistances: %s\n", file_distances))
    cat(sprintf("  FileTheta: %s\n", file_theta))
    cat(sprintf("  FileOut: %s\n", file_out))
    
    # Call the generateCOP function
    # Ensure that the input files actually exist before calling
    if (file.exists(file_sigma) && file.exists(file_distances) && file.exists(file_theta)) {

      if(current_type == "difficult"){
        G_param <- "min"
      }

      generateCOP(n = n_param, 
                  m = m_param, 
                  FileSigma = file_sigma, 
                  FileDistances = file_distances, 
                  FileTheta = file_theta, 
                  G = G_param, 
                  typeDistance = typeDistance_param, 
                  FileOut = file_out)
      cat(sprintf("Successfully processed instance Type: %s, Number: %d.\n", G_param, current_number))
    } else {
      cat(sprintf("Skipping instance Type: %s, Number: %d: One or more input files missing.\n", current_type, current_number))
      cat(sprintf("  Missing: Sigma (%s), Distances (%s), Theta (%s)\n", 
                  file.exists(file_sigma), file.exists(file_distances), file.exists(file_theta)))
    }
    cat("\n") # Add a newline for better readability between calls
  }
}

cat("--- Batch processing complete ---\n")
