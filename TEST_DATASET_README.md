# Deepfake Detection Test Dataset

A comprehensive test dataset for demonstrating and validating deepfake detection systems. This dataset contains a curated collection of real and fake content (images and videos) designed to test the accuracy and robustness of deepfake detection algorithms.

## 🎯 Purpose

This test dataset serves multiple purposes:
- **Demonstration**: Showcase your deepfake detection system's capabilities
- **Validation**: Test detection accuracy on known real/fake content
- **Development**: Provide consistent test cases during system development
- **Benchmarking**: Compare different detection models and approaches

## 📁 Project Structure

```
deepfake/
├── test_data/                    # Generated test dataset
│   ├── real/
│   │   ├── images/              # Real person photos
│   │   └── videos/              # Real video clips
│   ├── fake/
│   │   ├── images/              # Deepfake/fake images
│   │   └── videos/              # Deepfake/fake videos
│   ├── dataset_metadata.json    # Complete dataset information
│   ├── batch_test.py            # Automated testing script
│   └── README.md                # Dataset documentation
├── scripts/
│   ├── create_test_dataset.py           # Full dataset creator (requires OpenCV)
│   ├── create_test_dataset_simple.py    # Simplified creator (no OpenCV)
│   ├── requirements_test_dataset.txt     # Full requirements
│   └── requirements_simple.txt           # Minimal requirements
└── TEST_DATASET_README.md       # This file
```

## 🚀 Quick Start

### Option 1: Simple Setup (Recommended)

1. **Install minimal dependencies**:
   ```bash
   pip install -r scripts/requirements_simple.txt
   ```

2. **Create test dataset**:
   ```bash
   python scripts/create_test_dataset_simple.py
   ```

3. **Review the dataset**:
   ```bash
   ls -la test_data/
   ```

### Option 2: Full Setup (with Video Generation)

1. **Install full dependencies**:
   ```bash
   pip install -r scripts/requirements_test_dataset.txt
   ```

2. **Create test dataset**:
   ```bash
   python scripts/create_test_dataset.py
   ```

## 📊 Dataset Content

### Real Content (Expected: REAL classification)
- **Images**: High-quality photos from verified sources (Wikipedia Commons)
  - Celebrity portraits (Elon Musk, Mark Zuckerberg, Barack Obama)
  - Synthetic natural-looking faces
- **Videos**: Placeholder files (replace with actual content)
  - News interview clips
  - Official statements
  - Verified public domain content

### Fake Content (Expected: FAKE classification)
- **Images**: Synthetic deepfake images with artifacts
  - High-frequency artifact patterns
  - Compression artifacts
  - AI-generated face characteristics
- **Videos**: Placeholder files (replace with actual content)
  - Deepfake detection challenge samples
  - Research dataset examples
  - AI-generated talking heads

## 🧪 Testing Your System

### Manual Testing
1. Upload individual files through your deepfake detection interface
2. Compare results with expected classifications in `dataset_metadata.json`
3. Verify confidence scores fall within expected ranges

### Automated Batch Testing
1. **Configure your API endpoint** in `test_data/batch_test.py`:
   ```python
   API_BASE_URL = "http://localhost:8000"  # Adjust to your API URL
   ```

2. **Run batch tests**:
   ```bash
   cd test_data
   python batch_test.py
   ```

3. **Review results** in `test_results.json`

### Integration Testing
- Use the metadata file to integrate with your testing framework
- Parse `dataset_metadata.json` for test case information
- Compare actual vs. expected results programmatically

## 📋 Expected Results

Each sample includes expected detection results:

```json
{
  "expected_classification": "FAKE",
  "expected_confidence": {
    "min": 0.7,
    "max": 0.95
  },
  "notes": "Should be classified as FAKE based on synthetic deepfake with artifacts"
}
```

**Confidence Score Guidelines**:
- **REAL content**: 0.8 - 0.98 (high confidence in real classification)
- **FAKE content**: 0.7 - 0.95 (high confidence in fake classification)

## 🔧 Customization

### Adding Your Own Samples
1. **Real content**: Add verified, public domain images/videos
2. **Fake content**: Add known deepfake examples or create synthetic ones
3. **Update metadata**: Modify `dataset_metadata.json` to include new samples

### Modifying Expected Results
- Adjust confidence score ranges based on your model's performance
- Update expected classifications if you have ground truth data
- Add custom notes for specific test cases

### Extending the Dataset
- Add more artifact types for fake content
- Include different image/video formats
- Add metadata fields for specific use cases

## 📚 File Requirements

### Images
- **Format**: JPG (recommended) or PNG
- **Resolution**: 512x512 pixels (standardized)
- **Size**: Under 10MB per file
- **Quality**: High quality for real content, controlled artifacts for fake

### Videos
- **Format**: MP4 (recommended)
- **Duration**: 10-30 seconds
- **Resolution**: 640x480 or higher
- **Size**: Under 50MB per file
- **Content**: Clear faces, good audio quality

## 🌐 Content Sources

### Real Content
- **Wikipedia Commons**: Public domain celebrity photos
- **News websites**: Verified interview clips
- **Official sources**: Government/public figure statements
- **Public domain**: Educational and documentary content

### Fake Content
- **Research datasets**: FaceForensics++, Celeb-DF, DFDC
- **Academic papers**: Supplementary materials
- **Deepfake challenges**: Competition submissions
- **Synthetic generation**: Artificially created examples

## ⚠️ Important Notes

### Video Placeholders
- Some video files are text placeholders
- Replace with actual video content for full testing
- Placeholders explain what content should be added

### Legal Considerations
- Only use content you have rights to use
- Respect copyright and licensing requirements
- Use public domain or properly licensed materials

### Ethical Use
- This dataset is for testing and research purposes
- Do not use for malicious deepfake creation
- Respect privacy and consent requirements

## 🐛 Troubleshooting

### Common Issues

1. **Download failures**:
   - Check internet connection
   - Verify URLs are accessible
   - Check file permissions

2. **Image processing errors**:
   - Ensure Pillow is properly installed
   - Check image file integrity
   - Verify sufficient disk space

3. **API testing failures**:
   - Verify your API is running
   - Check endpoint URLs
   - Ensure proper authentication

### Performance Tips

- **Large datasets**: Process in batches
- **Memory usage**: Monitor during image generation
- **Network**: Use stable connections for downloads

## 🔗 Integration Examples

### Python Integration
```python
import json
from pathlib import Path

# Load test dataset metadata
with open('test_data/dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

# Access sample information
for name, info in metadata['samples'].items():
    print(f"{name}: {info['description']}")
    print(f"Expected: {info['expected']}")
    print(f"Path: {info['path']}")
```

### API Testing
```python
import requests

# Test a single image
with open('test_data/real/images/real_celebrity_1.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/detect', files=files)
    result = response.json()
    print(f"Detection result: {result}")
```

## 📈 Performance Metrics

### Accuracy Metrics
- **Classification accuracy**: % of correct real/fake classifications
- **Confidence correlation**: How well confidence scores align with expectations
- **False positive rate**: % of real content classified as fake
- **False negative rate**: % of fake content classified as real

### Benchmarking
- Compare your model against baseline results
- Track performance improvements over time
- Identify challenging test cases

## 🤝 Contributing

### Adding New Test Cases
1. Create new samples following the established format
2. Update metadata with proper descriptions
3. Test with your detection system
4. Document any special considerations

### Improving the Dataset
- Suggest better content sources
- Propose new artifact types
- Enhance metadata structure
- Improve testing automation

## 📞 Support

### Documentation
- Review this README for usage instructions
- Check individual script documentation
- Examine generated metadata files

### Issues
- Report bugs in the dataset creation scripts
- Suggest improvements to the test cases
- Request additional content types

### Community
- Share your testing results
- Contribute additional test cases
- Discuss integration approaches

## 📄 License

This test dataset is provided for educational and research purposes. Please ensure you have proper rights to use any content you add to the dataset.

## 🎉 Success Stories

Share how this test dataset helped you:
- Validate your deepfake detection system
- Improve detection accuracy
- Demonstrate capabilities to stakeholders
- Benchmark against other approaches

---

**Happy Testing! 🧪✨**

This test dataset is designed to make deepfake detection validation straightforward and comprehensive. Use it to build confidence in your system's capabilities and identify areas for improvement.





