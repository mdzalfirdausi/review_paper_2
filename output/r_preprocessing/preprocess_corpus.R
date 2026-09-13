
args <- commandArgs(trailingOnly = TRUE)

input_file  <- args[1]
output_file <- args[2]

suppressPackageStartupMessages({
    library(tm)
    library(SnowballC)
})

# ------------------------------------------------------------
# Load abstracts
# ------------------------------------------------------------

data <- read.csv(
    input_file,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
)

abstracts <- data$Abstract

# ------------------------------------------------------------
# Shared stopwords
# ------------------------------------------------------------

query_stops <- c(
    "power",
    "flow",
    "machine",
    "learning",
    "optimization",
    "optimisation"
)

all_stops <- unique(
    c(
        tm::stopwords("english"),
        query_stops
    )
)

# ------------------------------------------------------------
# tm preprocessing
# ------------------------------------------------------------

corpus <- VCorpus(
    VectorSource(abstracts)
)

corpus <- tm_map(
    corpus,
    content_transformer(tolower)
)

corpus <- tm_map(
    corpus,
    removePunctuation
)

corpus <- tm_map(
    corpus,
    removeNumbers
)

corpus <- tm_map(
    corpus,
    stripWhitespace
)

corpus <- tm_map(
    corpus,
    removeWords,
    all_stops
)

corpus <- tm_map(
    corpus,
    stemDocument,
    language = "english"
)

corpus <- tm_map(
    corpus,
    stripWhitespace
)

processed <- vapply(
    corpus,
    as.character,
    character(1)
)

result <- data.frame(
    document_id = seq_along(processed),
    processed_text = processed,
    stringsAsFactors = FALSE
)

write.csv(
    result,
    output_file,
    row.names = FALSE,
    fileEncoding = "UTF-8"
)
