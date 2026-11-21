NASA Lithium-Ion Battery Aging Dataset

1. Dataset Description
The dataset used for this project is the NASA Lithium-Ion Battery Aging Dataset. It was collected by NASA Ames Research Center through the Prognostics Center of Excellence.

It contains experimental measurements from lithium-ion batteries subjected to repeated charging, discharging, and electrochemical impedance tests. These experiments were conducted to investigate battery degradation and changes in electrical behavior over time.

Four battery datasets have been selected for this project:
B0005.mat
B0006.mat
B0007.mat
B0018.mat

Each file contains actual measurements from an individual 18650 lithium-ion battery. The batteries were tested under controlled laboratory conditions, allowing their behavior and degradation characteristics to be compared.

2. Dataset Features
The dataset contains the following features:

Feature     - Voltage_measured
Description - Measured battery terminal voltage (V)

Feature     - Current_measured
Description - Measured battery current (A)

Feature     - Temperature_measured
Description - Measured battery temperature (°C)

Feature     - Time
Description - Time elapsed during an experiment (s)

Feature     - Capacity
Description - Measured battery discharge capacity (Ah)

Feature     - Voltage_charge
Description - Voltage measured at the charger or load (V)

Feature     - Current_charge
Description - Current measured at the charger or load (A)

Feature     - Ambient_temperature
Description - Ambient temperature during testing (°C)

Feature     - Battery_impedance
Description - Measured battery impedance (Ω)

Feature     - Re
Description - Estimated electrolyte resistance (Ω)

Feature     - Rct
Description - Estimated charge-transfer resistance (Ω)

Measurements are organized by charge, discharge, and impedance cycles, so not every feature is available for every test. Also, the datasets are provided in MATLAB .mat format and can be processed using MATLAB or Python.

3. Prediction Labels
The primary prediction targets will be battery capacity, State of Health (SOH), and State of Charge (SOC). Battery capacity is provided in the dataset and can be used to calculate SOH. SOC reference values can be estimated using coulomb counting, assuming a known initial SOC and battery capacity.

4. Problem Statement
The goal of this project is to investigate how different machine learning algorithms perform when analyzing lithium-ion battery data from different batteries. Models will be trained to estimate battery capacity, SOC and SOH using experimental measurements and will be evaluated on different battery datasets.

The project will investigate whether algorithms trained on one battery can accurately predict the behavior of another battery and explain possible differences in performance.

The performance of different algorithms will be compared using metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

5.Engineering Significance
As batteries age, their capacity decreases and their electrical characteristics change. These changes affect battery performance, reliability, and available energy.

Understanding how well estimation algorithms generalize across different batteries is crucial. It will help engineers develop more reliable battery monitoring systems while reducing the need for extensive model changes.

6. Dataset Source and Access

Original source: NASA Ames Research Center, Prognostics Center of Excellence (PCoE).

Dataset name: NASA Lithium-Ion Battery Aging Dataset

Original dataset repository:
https://c3.ndc.nasa.gov/dashlink/resources/133/

Direct download:
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

File format: MATLAB .mat

Google Drive Link:
https://drive.google.com/drive/folders/1AYLG9hDgs28WYZDJ40H1Kam94imI6kj4?usp=drive_link

The dataset was downloaded from NASA's publicly available repository. The selected battery files (B0005, B0006, B0007, and B0018) were extracted and uploaded to Google Drive for storage.
