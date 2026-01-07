############################################
# FINAL PROJECT REGRESI HARGA RUMAH
# File R Terpadu
############################################

# ===============================
# 1. LOAD LIBRARY
# ===============================
library(readr)
library(corrplot)
library(reshape2)
library(ggplot2)
library(MASS)
library(car)
library(lmtest)
library(performance)
library(caret)

# ===============================
# 2. IMPORT DATA
# ===============================
data <- read_csv("house_price_regression_dataset.csv")

# ===============================
# 3. PRE-PROCESSING
# ===============================
View(data)
summary(data)

# Cek missing values
colSums(is.na(data))

# ===============================
# 4. ANALISIS DESKRIPTIF
# ===============================

# Scatter Plot Matrix
pairs(data, main = "Scatter Plot Antar Variabel")

# ===============================
# 5. ANALISIS KORELASI
# ===============================
correlation_matrix <- cor(data)
corrplot(correlation_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 30,
         addCoef.col = "black", number.cex = 0.7)

title(main = "Correlation Plot Antar Variabel",
      col.main = "cyan", font.main = 4)

# Heatmap dengan ggplot
corr_mat_round <- round(correlation_matrix, 2)
melted_corr <- melt(corr_mat_round)

ggplot(data = melted_corr,
       aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value),
            color = "white", size = 4) +
  theme_minimal() +
  labs(title = "Heatmap Korelasi")

# ===============================
# 6. MODEL REGRESI LINEAR
# ===============================

# First Order Model
first_order_model <- lm(
  House_Price ~ Square_Footage + Num_Bedrooms + Num_Bathrooms +
    Year_Built + Lot_Size + Garage_Size + Neighborhood_Quality,
  data = data
)

summary(first_order_model)

# ===============================
# 7. STEPWISE REGRESSION
# ===============================
stepwise_model <- stepAIC(first_order_model,
                          direction = "both",
                          trace = TRUE)

summary(stepwise_model)

# Perbandingan Model
anova(stepwise_model, first_order_model)

# ===============================
# 8. UJI ASUMSI KLASIK
# ===============================

# 1. Linearitas
plot(stepwise_model, which = 1)
abline(h = 0, col = "red", lty = 2)

# 2. Independensi Residual
dw_test <- durbinWatsonTest(stepwise_model)
print(dw_test)

# 3. Homoskedastisitas
plot(stepwise_model, which = 3)
bp_test <- bptest(stepwise_model)
print(bp_test)

# 4. Normalitas Residual
plot(stepwise_model, which = 2)
shapiro_test <- shapiro.test(residuals(stepwise_model))
print(shapiro_test)

# 5. Multikolinearitas
vif_values <- vif(stepwise_model)
print(vif_values)

# ===============================
# 9. VALIDASI MODEL (CROSS VALIDATION)
# ===============================
set.seed(123)

cv_control <- trainControl(
  method = "cv",
  number = 10
)

cv_model <- train(
  House_Price ~ Square_Footage + Num_Bedrooms + Num_Bathrooms +
    Year_Built + Lot_Size + Garage_Size,
  data = data,
  method = "lm",
  trControl = cv_control
)

print(cv_model)

cat("RMSE (Cross-validation):", cv_model$results$RMSE, "\n")
cat("R-squared (Cross-validation):", cv_model$results$Rsquared, "\n")

############################################
# END OF FILE
############################################
