# Advanced Frontend Components for Multi-Model Deepfake Detection

This document provides comprehensive documentation for the advanced React components designed for the multi-model deepfake detection system. These components offer production-quality UI patterns with modern design, responsive layouts, and comprehensive visualization capabilities.

## 🎯 **Component Overview**

### 1. **MultiModelDashboard** (`MultiModelDashboard.js`)
- **Purpose**: Comprehensive results dashboard with comparison views and model agreement visualization
- **Features**: Real-time progress tracking, ensemble analysis, performance metrics, and interactive charts
- **Key Capabilities**: Model comparison, agreement analysis, performance benchmarking

### 2. **HeatmapOverlay** (`HeatmapOverlay.js`)
- **Purpose**: Interactive heatmap visualization with zoom/pan capabilities
- **Features**: Attention maps, Grad-CAM overlays, frequency analysis visualization
- **Key Capabilities**: Zoom/pan controls, color scheme selection, export functionality

### 3. **RealTimeProgress** (`RealTimeProgress.js`)
- **Purpose**: Real-time analysis progress tracking for ensemble processing
- **Features**: Individual model progress, ensemble status, live logs, time estimation
- **Key Capabilities**: Start/pause/resume controls, detailed progress tracking

### 4. **FrequencyAnalysisDisplay** (`FrequencyAnalysisDisplay.js`)
- **Purpose**: Specialized F3Net frequency analysis visualization
- **Features**: DCT coefficients, frequency spectrum, artifact detection
- **Key Capabilities**: Tabbed interface, interactive charts, insights generation

## 🚀 **Quick Start**

### Installation

```bash
# Install required dependencies
npm install react-bootstrap recharts react-icons

# Import components
import MultiModelDashboard from './components/MultiModelDashboard';
import HeatmapOverlay from './components/HeatmapOverlay';
import RealTimeProgress from './components/RealTimeProgress';
import FrequencyAnalysisDisplay from './components/FrequencyAnalysisDisplay';
```

### Basic Usage

```jsx
import React, { useState } from 'react';
import MultiModelDashboard from './components/MultiModelDashboard';
import RealTimeProgress from './components/RealTimeProgress';

function App() {
  const [analysisId] = useState('analysis_123');
  const [models] = useState([
    { id: 'xception', name: 'Xception', estimatedTime: 5000 },
    { id: 'efficientnet', name: 'EfficientNet-B4', estimatedTime: 3000 },
    { id: 'f3net', name: 'F3Net', estimatedTime: 4000 }
  ]);

  const handleAnalysisComplete = (results) => {
    console.log('Analysis completed:', results);
  };

  return (
    <div className="App">
      <RealTimeProgress
        analysisId={analysisId}
        models={models}
        onComplete={handleAnalysisComplete}
        autoStart={true}
      />
    </div>
  );
}
```

## 📊 **Component Details**

### MultiModelDashboard

**Purpose**: Displays comprehensive analysis results with model comparison and ensemble insights.

```jsx
<MultiModelDashboard
  analysisId="analysis_123"
  onModelSelect={(modelKey, modelData) => {
    console.log('Selected model:', modelKey, modelData);
  }}
/>
```

**Key Features**:
- Ensemble prediction summary with confidence scores
- Individual model results with performance metrics
- Interactive charts for accuracy and confidence comparison
- Model agreement analysis with consensus strength
- Performance metrics and system statistics

**Props**:
- `analysisId` (string): Unique analysis identifier
- `onModelSelect` (function): Callback for model selection

### HeatmapOverlay

**Purpose**: Interactive visualization of attention maps and frequency analysis results.

```jsx
<HeatmapOverlay
  originalImage="/path/to/image.jpg"
  heatmapData={heatmapData}
  modelName="Xception"
  analysisType="attention"
  onClose={() => setShowHeatmap(false)}
  onExport={(dataUrl) => downloadImage(dataUrl)}
/>
```

**Key Features**:
- Zoom and pan controls for detailed inspection
- Multiple color schemes (jet, hot, cool, viridis, plasma)
- Toggle between original image and heatmap overlay
- Fullscreen mode for immersive analysis
- Export functionality for saving visualizations

**Props**:
- `originalImage` (string): URL of the original image
- `heatmapData` (object): Heatmap data with width, height, and pixel values
- `modelName` (string): Name of the model generating the heatmap
- `analysisType` (string): Type of analysis (attention, frequency, etc.)
- `onClose` (function): Callback for closing the overlay
- `onExport` (function): Callback for exporting the visualization

### RealTimeProgress

**Purpose**: Real-time tracking of multi-model analysis progress with detailed status updates.

```jsx
<RealTimeProgress
  analysisId="analysis_123"
  models={[
    { id: 'xception', name: 'Xception', estimatedTime: 5000 },
    { id: 'efficientnet', name: 'EfficientNet-B4', estimatedTime: 3000 },
    { id: 'f3net', name: 'F3Net', estimatedTime: 4000 }
  ]}
  onComplete={handleAnalysisComplete}
  onError={handleAnalysisError}
  onCancel={handleAnalysisCancel}
  autoStart={true}
/>
```

**Key Features**:
- Individual model progress tracking with realistic timing
- Ensemble processing status and fusion method display
- Real-time logs with timestamps and message types
- Estimated time remaining calculations
- Start/pause/resume/cancel controls

**Props**:
- `analysisId` (string): Unique analysis identifier
- `models` (array): Array of model configurations
- `onComplete` (function): Callback when analysis completes
- `onError` (function): Callback for error handling
- `onCancel` (function): Callback for cancellation
- `autoStart` (boolean): Whether to start analysis automatically

### FrequencyAnalysisDisplay

**Purpose**: Specialized visualization for F3Net frequency domain analysis results.

```jsx
<FrequencyAnalysisDisplay
  f3netResults={f3netResults}
  originalImage="/path/to/image.jpg"
  onExport={() => exportAnalysis()}
  onHeatmapView={() => setShowHeatmap(true)}
/>
```

**Key Features**:
- Tabbed interface for different analysis aspects
- DCT coefficient visualization with magnitude charts
- Frequency spectrum analysis with interactive controls
- Artifact detection with scoring and breakdown
- Insights generation based on analysis metrics

**Props**:
- `f3netResults` (object): F3Net analysis results
- `originalImage` (string): URL of the original image
- `onExport` (function): Callback for exporting analysis
- `onHeatmapView` (function): Callback for viewing heatmap

## 🎨 **Styling and Theming**

### CSS Files
Each component has its own CSS file for modular styling:
- `MultiModelDashboard.css`
- `HeatmapOverlay.css`
- `RealTimeProgress.css`
- `FrequencyAnalysisDisplay.css`

### Design Features
- **Modern UI**: Gradient backgrounds, rounded corners, subtle shadows
- **Responsive Design**: Mobile-first approach with breakpoints
- **Dark Mode Support**: Automatic dark mode detection and styling
- **Accessibility**: ARIA labels, keyboard navigation, reduced motion support
- **Animations**: Smooth transitions and hover effects

### Customization
```css
/* Custom color scheme */
:root {
  --primary-color: #667eea;
  --secondary-color: #764ba2;
  --success-color: #28a745;
  --danger-color: #dc3545;
  --warning-color: #ffc107;
  --info-color: #17a2b8;
}

/* Custom component styling */
.multi-model-dashboard {
  --dashboard-bg: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}
```

## 📱 **Responsive Design**

### Breakpoints
- **Desktop**: 1200px and above
- **Tablet**: 768px - 1199px
- **Mobile**: 576px - 767px
- **Small Mobile**: Below 576px

### Mobile Optimizations
- Collapsible navigation and controls
- Touch-friendly button sizes
- Simplified layouts for small screens
- Optimized chart rendering

## 🔧 **Integration Examples**

### Complete Analysis Flow

```jsx
import React, { useState } from 'react';
import MultiModelDashboard from './components/MultiModelDashboard';
import RealTimeProgress from './components/RealTimeProgress';
import HeatmapOverlay from './components/HeatmapOverlay';
import FrequencyAnalysisDisplay from './components/FrequencyAnalysisDisplay';

function AnalysisApp() {
  const [currentStep, setCurrentStep] = useState('progress');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);

  const models = [
    { id: 'xception', name: 'Xception', estimatedTime: 5000 },
    { id: 'efficientnet', name: 'EfficientNet-B4', estimatedTime: 3000 },
    { id: 'f3net', name: 'F3Net', estimatedTime: 4000 }
  ];

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setCurrentStep('results');
  };

  const handleModelSelect = (modelKey, modelData) => {
    setSelectedModel({ key: modelKey, data: modelData });
    setShowHeatmap(true);
  };

  return (
    <div className="analysis-app">
      {currentStep === 'progress' && (
        <RealTimeProgress
          analysisId="analysis_123"
          models={models}
          onComplete={handleAnalysisComplete}
          autoStart={true}
        />
      )}

      {currentStep === 'results' && analysisResults && (
        <MultiModelDashboard
          analysisId="analysis_123"
          onModelSelect={handleModelSelect}
        />
      )}

      {showHeatmap && selectedModel && (
        <HeatmapOverlay
          originalImage="/path/to/image.jpg"
          heatmapData={selectedModel.data.heatmap}
          modelName={selectedModel.data.name}
          analysisType="attention"
          onClose={() => setShowHeatmap(false)}
          onExport={(dataUrl) => downloadImage(dataUrl)}
        />
      )}

      {analysisResults?.f3net && (
        <FrequencyAnalysisDisplay
          f3netResults={analysisResults.f3net}
          originalImage="/path/to/image.jpg"
          onExport={() => exportAnalysis()}
          onHeatmapView={() => setShowHeatmap(true)}
        />
      )}
    </div>
  );
}
```

### API Integration

```jsx
// API service for analysis
const analysisAPI = {
  async startAnalysis(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await fetch('/api/analysis/start', {
      method: 'POST',
      body: formData
    });
    
    return response.json();
  },

  async getAnalysisStatus(analysisId) {
    const response = await fetch(`/api/analysis/${analysisId}/status`);
    return response.json();
  },

  async getAnalysisResults(analysisId) {
    const response = await fetch(`/api/analysis/${analysisId}/results`);
    return response.json();
  }
};

// Usage in component
useEffect(() => {
  const pollStatus = async () => {
    const status = await analysisAPI.getAnalysisStatus(analysisId);
    setAnalysisStatus(status);
    
    if (status.completed) {
      const results = await analysisAPI.getAnalysisResults(analysisId);
      setAnalysisResults(results);
    }
  };

  const interval = setInterval(pollStatus, 1000);
  return () => clearInterval(interval);
}, [analysisId]);
```

## 🧪 **Testing**

### Component Testing
```jsx
import { render, screen, fireEvent } from '@testing-library/react';
import MultiModelDashboard from './MultiModelDashboard';

test('renders dashboard with analysis data', () => {
  const mockAnalysisData = {
    id: 'test_123',
    models: {
      xception: { name: 'Xception', prediction: 'FAKE', confidence: 0.87 }
    }
  };

  render(<MultiModelDashboard analysisId="test_123" />);
  
  expect(screen.getByText('Multi-Model Analysis Results')).toBeInTheDocument();
  expect(screen.getByText('Xception')).toBeInTheDocument();
});
```

### Integration Testing
```jsx
test('complete analysis flow', async () => {
  render(<AnalysisApp />);
  
  // Start analysis
  fireEvent.click(screen.getByText('Start Analysis'));
  
  // Wait for completion
  await waitFor(() => {
    expect(screen.getByText('Analysis completed')).toBeInTheDocument();
  });
  
  // View results
  expect(screen.getByText('Ensemble Prediction')).toBeInTheDocument();
});
```

## 🚀 **Performance Optimization**

### Best Practices
1. **Lazy Loading**: Load components only when needed
2. **Memoization**: Use React.memo for expensive components
3. **Virtualization**: For large datasets, use virtualized lists
4. **Image Optimization**: Compress and optimize images for web
5. **Bundle Splitting**: Split large components into separate chunks

### Performance Monitoring
```jsx
// Performance monitoring hook
const usePerformanceMonitor = (componentName) => {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      console.log(`${componentName} render time: ${endTime - startTime}ms`);
    };
  });
};
```

## 🔒 **Security Considerations**

### Data Handling
- Sanitize user inputs and API responses
- Validate image files before processing
- Implement proper CORS policies
- Use HTTPS for all API communications

### Privacy
- Don't store sensitive analysis data in localStorage
- Implement proper session management
- Clear sensitive data on logout
- Use secure file upload endpoints

## 📚 **Additional Resources**

### Dependencies
- **React Bootstrap**: UI component library
- **Recharts**: Chart and visualization library
- **React Icons**: Icon library
- **React Router**: Navigation (if needed)

### Documentation
- [React Bootstrap Documentation](https://react-bootstrap.github.io/)
- [Recharts Documentation](https://recharts.org/)
- [React Icons](https://react-icons.github.io/react-icons/)

### Browser Support
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🤝 **Contributing**

### Development Setup
```bash
# Clone repository
git clone <repository-url>

# Install dependencies
npm install

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

### Code Style
- Use functional components with hooks
- Follow React best practices
- Maintain consistent naming conventions
- Add proper TypeScript types (if using TS)
- Include comprehensive documentation

### Testing Strategy
- Unit tests for individual components
- Integration tests for component interactions
- E2E tests for complete user flows
- Performance testing for large datasets

---

This documentation provides a comprehensive guide to using the advanced frontend components for the multi-model deepfake detection system. For additional support or questions, please refer to the component source code or contact the development team. 