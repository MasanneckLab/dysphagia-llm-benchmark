# Independent optional cross-check. Not executed locally: R is not installed.
# install.packages(c("ordinal", "emmeans")) before running this script.
# This uses Laplace estimation, so small differences from Python quadrature are expected.
library(ordinal)
library(emmeans)
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[1] else "."
d <- read.csv(file.path(root, "data", "response_level.csv"))
d$rating <- ordered(d$rating, levels = 0:2)
d$system <- relevel(factor(d$system), "GPT-5.2")
d$domain <- relevel(factor(d$domain), "Clinical Assessment")
d$question_type <- relevel(factor(d$question_type), "Knowledge")
d$guideline_id <- factor(d$guideline_id)
full <- tryCatch(clmm(rating ~ system + domain + question_type + guideline_id +
                      (1 | statement_id) + (1 | item_id), data = d, Hess = TRUE, nAGQ = 1),
                 error = function(e) e)
if (inherits(full, "error")) {
  message("Full fit failed as anticipated: ", conditionMessage(full))
} else {
  print(summary(full))
}
# Guideline 5 has all ratings Correct; the extra item variance is at the boundary.
# The finite reduced fit is the one used in the revision outputs.
reduced_data <- droplevels(subset(d, guideline_id != "5"))
reduced <- clmm(rating ~ system + domain + question_type + guideline_id +
                (1 | statement_id), data = reduced_data, Hess = TRUE, nAGQ = 25)
print(summary(reduced))
print(pairs(emmeans(reduced, ~ system, mode = "latent"), adjust = "holm"))
