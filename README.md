# Forest-Cover-Prediction
Target is to create a predictive model which helps to predict seven different cover types in four different wilderness areas of the Forest with the best accuracy
# Forest-Cover-Prediction
Target is to create a predictive model which helps to predict seven different cover types in four different wilderness areas of the Forest with the best accuracy
# Dataset
The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.
This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.¶

# Dataset Description
Elevation - Elevation in meters
Aspect - Aspect in degrees azimuth
Slope - Slope in degrees
Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation
The wilderness areas are:
1 - Rawah Wilderness Area
2 - Neota Wilderness Area
3 - Comanche Peak Wilderness Area
4 - Cache la Poudre Wilderness Area
The soil types are:
1 Cathedral family - Rock outcrop complex, extremely stony.
2 Vanet - Ratake families complex, very stony.
3 Haploborolis - Rock outcrop complex, rubbly.
4 Ratake family - Rock outcrop complex, rubbly.
5 Vanet family - Rock outcrop complex complex, rubbly.
6 Vanet - Wetmore families - Rock outcrop complex, stony.
7 Gothic family.
8 Supervisor - Limber families complex.
9 Troutville family, very stony.
10 Bullwark - Catamount families - Rock outcrop complex, rubbly.
11 Bullwark - Catamount families - Rock land complex, rubbly.
12 Legault family - Rock land complex, stony.
13 Catamount family - Rock land - Bullwark family complex, rubbly.
14 Pachic Argiborolis - Aquolis complex.
15 unspecified in the USFS Soil and ELU Survey.
16 Cryaquolis - Cryoborolis complex.
17 Gateview family - Cryaquolis complex.
18 Rogert family, very stony.
19 Typic Cryaquolis - Borohemists complex.
20 Typic Cryaquepts - Typic Cryaquolls complex.
21 Typic Cryaquolls - Leighcan family, till substratum complex.
22 Leighcan family, till substratum, extremely bouldery.
23 Leighcan family, till substratum - Typic Cryaquolls complex.
24 Leighcan family, extremely stony.
25 Leighcan family, warm, extremely stony.
26 Granile - Catamount families complex, very stony.
27 Leighcan family, warm - Rock outcrop complex, extremely stony.
28 Leighcan family - Rock outcrop complex, extremely stony.
29 Como - Legault families complex, extremely stony.
30 Como family - Rock land - Legault family complex, extremely stony.
31 Leighcan - Catamount families complex, extremely stony.
32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.
34 Cryorthents - Rock land complex, extremely stony.
35 Cryumbrepts - Rock outcrop - Cryaquepts complex.
36 Bross family - Rock land - Cryumbrepts complex, extremely stony.
37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
38 Leighcan - Moran families - Cryaquolls complex, extremely stony.
39 Moran family - Cryorthents - Leighcan family complex, extremely stony.
40 Moran family - Cryorthents - Rock land complex, extremely stony.

# Insights From Exploratory Data Analysis
Data set have balanced labels, resulting in equal number of cover types. This will be an advantage when it comes to apply classification ML models because, the model will have good chance to learn patterns of all labels, eliminating the probability of underfitting.

Different wilderness areas consist of some specific trees. Interestingly, there is one fantastic tree, Cottonwood/Willow, specifically likes to grow in wilderness area 4. While cover types 1, 2, 5 and 6 can grow in any soil type, other cover types grows more with specific soil types.

Soil types are reverse-one-hot-encoded, meaning they are going to be included as numeric data in the training set and one-hot-encoded soil type columns will be excluded. With that way, there is a stronger correlation between soil type and Cover_Type. Numeric soil type column and other variables have pearson coefficients in the range of [-0.2, 0.1].

Hillshade columns are collinear within each other and Hillshade_9am has the least importance in determining Cover_Type. Thus this column will be dropped in Part 3 for better interpretability of the future model.

# Based on the comparison of models trained on the forest cover prediction dataset, we can draw several insights and conclusions:

# Model Performance:

The Random Forest and XGBoost models achieved the highest accuracy among all the models considered. Decision Tree and k-NN models achieved moderate accuracy, while the SVM model lagged slightly behind. The relative performance of models may vary depending on the specific dataset and problem at hand. Complexity and Interpretability:

Decision Tree and k-NN models tend to be simpler and more interpretable compared to ensemble methods like Random Forest and XGBoost, which are more complex and often referred to as "black-box" models. SVM falls somewhere in between, offering good accuracy with a moderate level of interpretability.

# Scalability:

k-NN and Decision Tree models are relatively straightforward and may scale well with larger datasets. SVM and ensemble methods like Random Forest and XGBoost can be computationally intensive and may require more resources for training and inference.

# Robustness:

Ensemble methods like Random Forest and XGBoost are generally more robust to noise and overfitting compared to simpler models like Decision Tree and k-NN. SVM can also be robust, especially with proper tuning of hyperparameters, but may require careful preprocessing and feature scaling.

# Hyperparameter Sensitivity:

Ensemble methods like Random Forest and XGBoost have several hyperparameters to tune, but they tend to be less sensitive to the choice of hyperparameters compared to models like SVM. SVM's performance can be highly dependent on the choice of hyperparameters such as the choice of kernel, regularization parameter (C), and kernel coefficient (gamma).

# Overall Recommendation:

Based on the comparison, if we prioritize accuracy and robustness and are less concerned about interpretability, ensemble methods like Random Forest and XGBoost are preferable. If interpretability is a significant concern or computational resources are limited, simpler models like Decision Tree or k-NN can still provide reasonable performance. The choice of model ultimately depends on the specific requirements of the problem, computational resources available, and the balance between model complexity and interpretability.
