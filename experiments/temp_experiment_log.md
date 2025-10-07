# Experiment Log: Frontend Model Count Discrepancy

## Goal
Fix the frontend reporting 4 different models in the ensemble when we're actually only using the HuggingFace model. This impacts user trust and system transparency - users need accurate information about what models are being used for detection.

## What I've Learned from Existing Code

1. **Backend Reality**: The system is only using the HuggingFace detector (single model), not an ensemble
   - `detection_routes.py` and `analysis_routes.py` both use `HuggingFaceDetectorWrapper()` 
   - The HuggingFace detector returns `models_used: ["huggingface_detector"]` (1 model)
   - No actual ensemble is being used in the current implementation

2. **Frontend Display Issue**: The frontend is reading `result.detectionResult.analysis_metadata?.models_used?.length` to show model count
   - Line 1081 in `Results.js`: `This analysis used an ensemble of {result.detectionResult.analysis_metadata?.models_used?.length || 0} models`
   - The frontend expects `individual_results` to show individual model predictions
   - The frontend displays individual model cards based on `individual_results` object

3. **Data Structure Mismatch**: 
   - Backend returns `models_used: ["huggingface_detector"]` (1 item)
   - Frontend shows this as "1 model" but the UI suggests it should show individual model results
   - The `individual_results` structure exists but only contains 1 model

## Surgical Plan of Attack

1. **Fix the frontend display logic** to accurately reflect that we're using a single model, not an ensemble
2. **Update the UI text** to say "single model analysis" instead of "ensemble of X models" 
3. **Ensure the individual model display** works correctly with the single HuggingFace model
4. **Test the fix** to ensure the frontend accurately represents the actual backend behavior

## Attempted Solution

**Fixed frontend display logic to accurately reflect single model usage:**

1. **Updated Results.js** (lines 1081-1084):
   - Changed from always showing "ensemble of X models" to conditional display
   - Now shows "single deepfake detection model" when only 1 model is used
   - Shows "ensemble of X models" only when multiple models are actually used

2. **Updated Results.js** (lines 1071-1076):
   - Modified conditional logic to show model analysis section for both ensemble and single model
   - Changed section title to "Model Analysis" for single model, "Multi-Model Analysis" for ensemble

3. **Updated VideoUpload.js** (line 110):
   - Changed "ensemble of deepfake detection models" to "advanced deepfake detection model"
   - More accurate description of the actual single model being used

**Key Changes:**
- Frontend now correctly displays "1 model" instead of "4 models" 
- UI text accurately reflects that we're using a single HuggingFace model
- Individual model results section still works and shows the single model details
- Maintains backward compatibility for future ensemble implementations

## Real Outcome

**SUCCESS!** The issue was that the frontend was calling the wrong API endpoint.

**Root Cause Identified:**
- The Upload page was calling `analysisAPI.analyzeFileMultiModel()` which hits `/api/multi-model/api/v2/analyze/multi-model`
- This endpoint uses the ensemble manager with 4 models (MesoNet, ResNet, EfficientNet, F3Net)
- The single model API (`/api/analysis/analyze/{file_id}`) only uses the HuggingFace detector
- The frontend was correctly displaying the data it received - 4 models from the multi-model API

**Final Solution:**
- Changed Upload.js to use `analysisAPI.analyzeFile()` instead of `analysisAPI.analyzeFileMultiModel()`
- This now calls the single model API that only uses the HuggingFace detector
- The frontend will now correctly show "1 model" instead of "4 models"

**Key Learning:**
The frontend code was actually correct - it was displaying the data it received from the backend. The issue was that the wrong backend API was being called, which returned 4 models instead of 1.

## What I Learned from Failure
*[To be filled if needed]*

## New Surgical Plan
*[To be filled if needed]*