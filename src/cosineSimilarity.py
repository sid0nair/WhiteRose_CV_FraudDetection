# Step 3: Apply Min-Max Scaling on the original data
minmax_scaler = MinMaxScaler()
risk_factors_minmax_scaled = minmax_scaler.fit_transform(risk_factors_final_df[columns_to_process])

# Apply Isolation Forest on the Min-Max scaled data
iso_forest_step3 = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest_step3.fit(risk_factors_minmax_scaled)

# Predict anomalies
iso_forest_labels_step3 = iso_forest_step3.predict(risk_factors_minmax_scaled)

# Identify anomalies: Isolation Forest marks anomalies with -1
risk_factors_final_df['is_anomaly'] = (iso_forest_labels_step3 == -1).astype(int)

# Visualize using PCA
risk_factors_pca_step3 = pca.fit_transform(risk_factors_minmax_scaled)

# Visualization: Scatter plot of Isolation Forest results in 2D PCA space (Min-Max Scaling)
plt.figure(figsize=(10, 6))
# Plot normal points
plt.scatter(risk_factors_pca_step3[iso_forest_labels_step3 == 1, 0], risk_factors_pca_step3[iso_forest_labels_step3 == 1, 1], 
            c='blue', alpha=0.5, label='Normal Points')
# Plot anomalies
plt.scatter(risk_factors_pca_step3[iso_forest_labels_step3 == -1, 0], risk_factors_pca_step3[iso_forest_labels_step3 == -1, 1], 
            color='red', edgecolor='k', label='Anomalies')

plt.title('Isolation Forest: Anomaly Detection (Min-Max Scaled Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
