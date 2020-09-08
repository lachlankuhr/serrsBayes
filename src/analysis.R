library(tidyverse)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
data <- read_csv("sample.csv", col_names = FALSE)
plot(hist(data$X1))

N <- 1:16
P <- seq(100, 2000, 10)
combos <- expand.grid(N, P)

formula <- (combos$Var1/combos$Var2 + log(combos$Var1)) / (combos$Var1/combos$Var2)
combos$results <- formula

ggplot(data = combos, mapping = aes(x = Var1,
                                                       y = Var2)) + 
  geom_raster(aes(fill = results), interpolate = TRUE) + 
  xlab(label = "Number Of Processing Units") +
  ylab(label = "Number Of Particles")
