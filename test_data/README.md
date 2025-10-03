# Deepfake Detection Test Dataset

This directory contains a comprehensive test dataset for demonstrating deepfake detection capabilities.

## Dataset Structure

```
test_data/
├── real/
│   ├── images/          # Real person photos
│   └── videos/          # Real video clips (placeholders)
├── fake/
│   ├── images/          # Deepfake/fake images
│   └── videos/          # Deepfake/fake videos (placeholders)
├── dataset_metadata.json # Complete dataset information
├── batch_test.py        # Batch testing script
└── README.md            # This file
```

## Content Overview

### Real Content (3 samples)
- **Images**: High-quality photos of real people from verified sources
- **Videos**: Placeholder files (replace with actual video content)
- **Sources**: Wikipedia Commons, official portraits, verified content

### Fake Content (5 samples)
- **Images**: Synthetic deepfake images with various artifacts
- **Videos**: Placeholder files (replace with actual deepfake videos)
- **Sources**: Generated samples, research dataset simulations

## Expected Results

Each sample includes expected classification results:
- **REAL**: Should be classified as genuine content
- **FAKE**: Should be classified as deepfake/manipulated content

Confidence scores are provided as ranges to account for model variations.

## Usage

### 1. Manual Testing
Upload individual files through your deepfake detection system's interface.

### 2. Batch Testing
Run the automated batch testing script:
```bash
cd test_data
python batch_test.py
```

### 3. Integration Testing
Use the metadata file to integrate with your testing framework.

## File Requirements

- **Images**: JPG format, 512x512 resolution
- **Videos**: MP4 format, 10-30 seconds duration (replace placeholders)
- **Size**: All files under 50MB for easy testing

## Important Notes

- **Video Placeholders**: Some video files are text placeholders. Replace them with actual video files for full testing.
- **Real Videos**: Use clips from news interviews, official statements, or verified content.
- **Fake Videos**: Use samples from deepfake detection challenges or research datasets.

## Getting Real Video Content

For real video content, consider:
1. News interview clips (YouTube, news websites)
2. Official statements from verified sources
3. Public domain educational content

For fake video content, consider:
1. Deepfake detection challenge datasets
2. Academic research paper supplementary materials
3. Publicly available deepfake examples

## Support

For questions about this test dataset, refer to your deepfake detection system documentation.
