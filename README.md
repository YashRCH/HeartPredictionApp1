# Heart Disease Risk Prediction System

An end-to-end machine learning pipeline that predicts cardiovascular risk factors using patient health metrics, deployed as an Android application with on-device inference.

## System Architecture
1. **Data Pipeline**  
   - Ingests structured health data (age, BP, cholesterol, etc.)  
   - Automated preprocessing: Missing value imputation, categorical encoding, feature scaling  

2. **Machine Learning Core**  
   - XGBoost classifier trained on clinical datasets  
   - Model achieves 89% accuracy (F1-score: 0.88)  
   - Hyperparameter optimization via grid search  

3. **Edge Deployment**  
   - Model converted to ONNX format for cross-platform compatibility  
   - Scaler objects serialized using Pickle  
   - ONNX Runtime Mobile for efficient inference on Android devices  

4. **Mobile Application**  
   - Java-based Android frontend  
   - Dynamic risk visualization (traffic-light color coding)  
   - Complete offline operation - no data leaves device  

## Technical Highlights
 Privacy-focused design (no cloud dependencies)  
 <100ms inference latency on mid-range smartphones  
 Comprehensive preprocessing pipeline  
 Explainable risk factors through feature importance  

Built with: Python • Scikit-learn • XGBoost • ONNX • Java • Android SDK
