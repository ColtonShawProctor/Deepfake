"""
Video Visualization and Timeline Analysis

Advanced visualization system for video deepfake detection results,
providing temporal analysis, confidence timelines, and interactive
exploration of detection patterns.

Visualization Components:
1. Confidence Timeline with annotations
2. Temporal Consistency Heatmaps
3. Frame-by-frame Analysis Views
4. Optical Flow Visualizations
5. Alert Timeline and Patterns
6. Performance Metrics Dashboard
7. Interactive Video Scrubbing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

from .video_analysis_core import VideoAnalysisResult, FrameInfo
from .realtime_video_processor import StreamAlert, StreamStats, AlertLevel

@dataclass
class VisualizationConfig:
    """Configuration for video visualizations"""
    # Plot styling
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 100
    color_scheme: str = "viridis"
    dark_theme: bool = True
    
    # Timeline settings
    timeline_height: int = 200
    confidence_threshold: float = 50.0
    alert_marker_size: int = 10
    
    # Heatmap settings
    heatmap_resolution: int = 100
    temporal_window: int = 30
    
    # Animation settings
    animation_fps: int = 10
    frame_display_size: Tuple[int, int] = (224, 224)
    
    # Export settings
    export_format: str = "html"  # html, png, mp4
    interactive: bool = True

class VideoTimelineVisualizer:
    """Create timeline visualizations for video analysis results"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup matplotlib and plotly styling"""
        if self.config.dark_theme:
            plt.style.use('dark_background')
            self.plotly_template = "plotly_dark"
        else:
            plt.style.use('default')
            self.plotly_template = "plotly_white"
    
    def create_confidence_timeline(self, result: VideoAnalysisResult, 
                                 save_path: Optional[str] = None) -> go.Figure:
        """Create interactive confidence timeline"""
        
        # Extract timeline data
        timestamps = [r['timestamp'] for r in result.frame_results]
        confidences = [r['confidence'] for r in result.frame_results]
        is_deepfake = [r['is_deepfake'] for r in result.frame_results]
        
        # Create main timeline plot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Confidence Timeline', 'Detection States', 'Frame Analysis'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.2, 0.3]
        )
        
        # Confidence line plot
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='cyan', width=2),
                marker=dict(size=4),
                hovertemplate='Time: %{x:.2f}s<br>Confidence: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Threshold line
        fig.add_hline(
            y=self.config.confidence_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
            row=1, col=1
        )
        
        # Detection state indicators
        deepfake_times = [t for t, df in zip(timestamps, is_deepfake) if df]
        deepfake_confs = [c for c, df in zip(confidences, is_deepfake) if df]
        
        if deepfake_times:
            fig.add_trace(
                go.Scatter(
                    x=deepfake_times,
                    y=[1] * len(deepfake_times),
                    mode='markers',
                    name='Deepfake Detected',
                    marker=dict(color='red', size=8, symbol='diamond'),
                    hovertemplate='Deepfake at %{x:.2f}s<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Frame analysis scatter
        motion_scores = [r.get('motion_score', 0) for r in result.frame_results]
        keyframes = [r.get('is_keyframe', False) for r in result.frame_results]
        
        # Regular frames
        regular_mask = [not kf for kf in keyframes]
        if any(regular_mask):
            fig.add_trace(
                go.Scatter(
                    x=[t for t, m in zip(timestamps, regular_mask) if m],
                    y=[ms for ms, m in zip(motion_scores, regular_mask) if m],
                    mode='markers',
                    name='Regular Frames',
                    marker=dict(color='blue', size=4),
                    hovertemplate='Time: %{x:.2f}s<br>Motion: %{y:.3f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Keyframes
        keyframe_mask = keyframes
        if any(keyframe_mask):
            fig.add_trace(
                go.Scatter(
                    x=[t for t, m in zip(timestamps, keyframe_mask) if m],
                    y=[ms for ms, m in zip(motion_scores, keyframe_mask) if m],
                    mode='markers',
                    name='Key Frames',
                    marker=dict(color='gold', size=8, symbol='star'),
                    hovertemplate='Keyframe at %{x:.2f}s<br>Motion: %{y:.3f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'Video Analysis Timeline - {Path(result.video_path).name}',
            template=self.plotly_template,
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)")
        fig.update_yaxes(title_text="Confidence (%)", row=1, col=1)
        fig.update_yaxes(title_text="Detection", row=2, col=1, range=[-0.1, 1.1])
        fig.update_yaxes(title_text="Motion Score", row=3, col=1)
        
        if save_path:
            if self.config.export_format == "html":
                fig.write_html(save_path)
            elif self.config.export_format == "png":
                fig.write_image(save_path, width=1200, height=800)
        
        return fig
    
    def create_temporal_consistency_heatmap(self, result: VideoAnalysisResult,
                                          save_path: Optional[str] = None) -> go.Figure:
        """Create temporal consistency heatmap"""
        
        confidences = result.confidence_timeline
        if len(confidences) < self.config.temporal_window:
            # Pad with zeros if too short
            confidences = confidences + [0] * (self.config.temporal_window - len(confidences))
        
        # Create sliding window matrix
        window_size = min(self.config.temporal_window, len(confidences))
        n_windows = len(confidences) - window_size + 1
        
        heatmap_data = []
        for i in range(n_windows):
            window = confidences[i:i + window_size]
            heatmap_data.append(window)
        
        heatmap_data = np.array(heatmap_data)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            colorscale='RdYlBu_r',
            hovertemplate='Window: %{y}<br>Frame: %{x}<br>Confidence: %{z:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Temporal Consistency Heatmap',
            xaxis_title='Frame in Window',
            yaxis_title='Time Window',
            template=self.plotly_template
        )
        
        if save_path:
            if self.config.export_format == "html":
                fig.write_html(save_path)
            elif self.config.export_format == "png":
                fig.write_image(save_path)
        
        return fig
    
    def create_frame_analysis_grid(self, result: VideoAnalysisResult, 
                                  max_frames: int = 16,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create grid showing key frames with analysis results"""
        
        # Select key frames to display
        frame_results = result.frame_results
        if len(frame_results) > max_frames:
            # Select evenly distributed frames
            indices = np.linspace(0, len(frame_results) - 1, max_frames, dtype=int)
            selected_results = [frame_results[i] for i in indices]
        else:
            selected_results = frame_results
        
        # Calculate grid dimensions
        n_frames = len(selected_results)
        n_cols = int(np.ceil(np.sqrt(n_frames)))
        n_rows = int(np.ceil(n_frames / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, frame_result in enumerate(selected_results):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Create placeholder frame (in real implementation, would load actual frame)
            placeholder = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Color border based on detection
            border_color = 'red' if frame_result['is_deepfake'] else 'green'
            confidence = frame_result['confidence']
            
            ax.imshow(placeholder)
            ax.set_title(f"Frame {frame_result['frame_idx']}\n"
                        f"Conf: {confidence:.1f}%\n"
                        f"Time: {frame_result['timestamp']:.1f}s",
                        fontsize=8, color=border_color)
            ax.axis('off')
            
            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
        
        # Hide unused subplots
        for i in range(n_frames, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Key Frames Analysis - {Path(result.video_path).name}', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def create_optical_flow_visualization(self, result: VideoAnalysisResult,
                                        save_path: Optional[str] = None) -> go.Figure:
        """Create optical flow analysis visualization"""
        
        flow_data = result.optical_flow_analysis
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Flow Magnitudes', 'Flow Variance', 
                          'Inconsistencies', 'Flow Statistics'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Flow magnitudes over time
        if 'flow_magnitudes' in flow_data:
            magnitudes = flow_data['flow_magnitudes']
            fig.add_trace(
                go.Scatter(
                    y=magnitudes,
                    mode='lines+markers',
                    name='Flow Magnitude',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Flow variance
        if 'flow_magnitudes' in flow_data:
            # Calculate rolling variance
            window_size = 5
            variances = []
            magnitudes = flow_data['flow_magnitudes']
            for i in range(len(magnitudes)):
                start_idx = max(0, i - window_size)
                window = magnitudes[start_idx:i+1]
                variances.append(np.var(window))
            
            fig.add_trace(
                go.Scatter(
                    y=variances,
                    mode='lines',
                    name='Flow Variance',
                    line=dict(color='orange')
                ),
                row=1, col=2
            )
        
        # Inconsistencies bar chart
        if 'inconsistencies' in flow_data:
            inconsistencies = flow_data['inconsistencies']
            fig.add_trace(
                go.Bar(
                    x=list(range(len(inconsistencies))),
                    y=[1] * len(inconsistencies),
                    name='Inconsistencies',
                    marker_color='red'
                ),
                row=2, col=1
            )
        
        # Statistics table
        stats_data = [
            ['Average Flow', f"{flow_data.get('average_flow', 0):.4f}"],
            ['Flow Variance', f"{flow_data.get('flow_variance', 0):.4f}"],
            ['Inconsistency Ratio', f"{flow_data.get('inconsistency_ratio', 0):.2%}"],
            ['Total Inconsistencies', str(len(flow_data.get('inconsistencies', [])))]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[row[0] for row in stats_data],
                                 [row[1] for row in stats_data]])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Optical Flow Analysis',
            template=self.plotly_template,
            height=600
        )
        
        if save_path:
            if self.config.export_format == "html":
                fig.write_html(save_path)
            elif self.config.export_format == "png":
                fig.write_image(save_path)
        
        return fig

class RealTimeVisualizationDashboard:
    """Real-time visualization dashboard for stream processing"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.alert_history = []
        self.stats_history = []
        self.confidence_buffer = []
        
    def create_realtime_dashboard(self, stats: StreamStats, 
                                recent_alerts: List[StreamAlert]) -> go.Figure:
        """Create real-time monitoring dashboard"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Processing Statistics', 'Memory & CPU Usage',
                'Alert Timeline', 'Detection Rate',
                'FPS & Latency', 'System Health'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "indicator"}]
            ]
        )
        
        # Processing statistics
        stat_names = ['Frames Processed', 'Frames Dropped', 'Deepfake Detections', 'Alerts']
        stat_values = [stats.frames_processed, stats.frames_dropped, 
                      stats.deepfake_detections, stats.alerts_generated]
        
        fig.add_trace(
            go.Bar(x=stat_names, y=stat_values, name='Statistics',
                  marker_color=['blue', 'orange', 'red', 'purple']),
            row=1, col=1
        )
        
        # Memory & CPU usage over time
        self.stats_history.append({
            'timestamp': datetime.now(),
            'memory': stats.memory_usage,
            'cpu': stats.cpu_usage
        })
        
        # Keep only recent history
        if len(self.stats_history) > 100:
            self.stats_history = self.stats_history[-100:]
        
        if len(self.stats_history) > 1:
            timestamps = [s['timestamp'] for s in self.stats_history]
            memory_usage = [s['memory'] for s in self.stats_history]
            cpu_usage = [s['cpu'] for s in self.stats_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=memory_usage, name='Memory (GB)',
                          line=dict(color='green')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=timestamps, y=cpu_usage, name='CPU (%)',
                          line=dict(color='blue'), yaxis='y2'),
                row=1, col=2
            )
        
        # Alert timeline
        if recent_alerts:
            alert_times = [alert.timestamp for alert in recent_alerts]
            alert_levels = [1 if alert.level == AlertLevel.CRITICAL else 0.5 
                          for alert in recent_alerts]
            alert_colors = ['red' if alert.level == AlertLevel.CRITICAL else 'orange'
                          for alert in recent_alerts]
            
            fig.add_trace(
                go.Scatter(
                    x=alert_times, y=alert_levels,
                    mode='markers',
                    name='Alerts',
                    marker=dict(color=alert_colors, size=10),
                    hovertemplate='%{text}<extra></extra>',
                    text=[alert.message for alert in recent_alerts]
                ),
                row=2, col=1
            )
        
        # Detection rate indicator
        detection_rate = (stats.deepfake_detections / max(stats.frames_processed, 1)) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=detection_rate,
                title={'text': "Detection Rate (%)"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "darkblue"},
                      'steps': [{'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 100], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=2
        )
        
        # FPS & Latency
        fig.add_trace(
            go.Scatter(
                x=['FPS', 'Latency (ms)'],
                y=[stats.average_fps, stats.average_latency],
                mode='markers+text',
                text=[f'{stats.average_fps:.1f}', f'{stats.average_latency:.1f}'],
                textposition='top center',
                marker=dict(size=30, color=['green', 'orange'])
            ),
            row=3, col=1
        )
        
        # System health indicator
        health_score = min(100, (stats.uptime / 3600) * 10)  # Score based on uptime
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={'text': "System Health"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "green"},
                      'steps': [{'range': [0, 50], 'color': "red"},
                               {'range': [50, 80], 'color': "yellow"},
                               {'range': [80, 100], 'color': "lightgreen"}]}
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title='Real-Time Video Processing Dashboard',
            template=self.plotly_template,
            height=900,
            showlegend=False
        )
        
        return fig
    
    def create_alert_analysis(self, alerts: List[StreamAlert]) -> go.Figure:
        """Create alert pattern analysis"""
        
        if not alerts:
            return go.Figure().add_annotation(
                text="No alerts to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Group alerts by hour
        alert_df = pd.DataFrame([
            {
                'timestamp': datetime.fromtimestamp(alert.timestamp),
                'level': alert.level,
                'confidence': alert.confidence,
                'hour': datetime.fromtimestamp(alert.timestamp).hour
            }
            for alert in alerts
        ])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Alerts by Hour', 'Alert Levels', 
                          'Confidence Distribution', 'Alert Timeline')
        )
        
        # Alerts by hour
        hourly_counts = alert_df.groupby('hour').size()
        fig.add_trace(
            go.Bar(x=hourly_counts.index, y=hourly_counts.values,
                  name='Alerts per Hour'),
            row=1, col=1
        )
        
        # Alert levels pie chart
        level_counts = alert_df['level'].value_counts()
        fig.add_trace(
            go.Pie(labels=level_counts.index, values=level_counts.values,
                  name='Alert Levels'),
            row=1, col=2
        )
        
        # Confidence distribution
        fig.add_trace(
            go.Histogram(x=alert_df['confidence'], name='Confidence Distribution',
                        nbinsx=20),
            row=2, col=1
        )
        
        # Alert timeline
        fig.add_trace(
            go.Scatter(
                x=alert_df['timestamp'],
                y=alert_df['confidence'],
                mode='markers',
                marker=dict(
                    color=['red' if level == AlertLevel.CRITICAL else 'orange' 
                          for level in alert_df['level']],
                    size=8
                ),
                name='Alerts Over Time'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Alert Pattern Analysis',
            template=self.plotly_template,
            height=600
        )
        
        return fig

class VideoExportUtilities:
    """Utilities for exporting video analysis results"""
    
    @staticmethod
    def export_analysis_report(result: VideoAnalysisResult, 
                             output_path: str,
                             include_visualizations: bool = True):
        """Export comprehensive analysis report"""
        
        report = {
            'video_info': {
                'path': result.video_path,
                'total_frames': result.total_frames,
                'analyzed_frames': result.analyzed_frames,
                'processing_time': result.processing_time,
                'memory_usage': result.memory_usage
            },
            'analysis_results': {
                'overall_confidence': result.overall_confidence,
                'is_deepfake': result.is_deepfake,
                'temporal_consistency': result.temporal_consistency
            },
            'temporal_patterns': result.temporal_patterns,
            'optical_flow_analysis': result.optical_flow_analysis,
            'frame_results': result.frame_results,
            'confidence_timeline': result.confidence_timeline,
            'config': result.analysis_config.__dict__
        }
        
        # Export as JSON
        with open(f"{output_path}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create visualizations if requested
        if include_visualizations:
            visualizer = VideoTimelineVisualizer()
            
            # Timeline
            timeline_fig = visualizer.create_confidence_timeline(result)
            timeline_fig.write_html(f"{output_path}_timeline.html")
            
            # Heatmap
            heatmap_fig = visualizer.create_temporal_consistency_heatmap(result)
            heatmap_fig.write_html(f"{output_path}_heatmap.html")
            
            # Optical flow
            flow_fig = visualizer.create_optical_flow_visualization(result)
            flow_fig.write_html(f"{output_path}_optical_flow.html")
    
    @staticmethod
    def create_video_summary_animation(result: VideoAnalysisResult,
                                     output_path: str):
        """Create animated summary of video analysis"""
        
        # This would create an MP4 animation showing:
        # - Key frames with detection results
        # - Confidence timeline
        # - Real-time analysis progression
        
        # Placeholder implementation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        def animate(frame_idx):
            ax1.clear()
            ax2.clear()
            
            # Show confidence up to current frame
            current_confidences = result.confidence_timeline[:frame_idx+1]
            current_timestamps = [r['timestamp'] for r in result.frame_results[:frame_idx+1]]
            
            ax1.plot(current_timestamps, current_confidences, 'b-')
            ax1.axhline(y=50, color='r', linestyle='--', alpha=0.7)
            ax1.set_ylabel('Confidence (%)')
            ax1.set_title(f'Video Analysis Progress - Frame {frame_idx+1}/{len(result.frame_results)}')
            
            # Show current frame info
            if frame_idx < len(result.frame_results):
                current_result = result.frame_results[frame_idx]
                ax2.text(0.1, 0.8, f"Frame: {current_result['frame_idx']}", transform=ax2.transAxes)
                ax2.text(0.1, 0.6, f"Time: {current_result['timestamp']:.2f}s", transform=ax2.transAxes)
                ax2.text(0.1, 0.4, f"Confidence: {current_result['confidence']:.1f}%", transform=ax2.transAxes)
                ax2.text(0.1, 0.2, f"Deepfake: {'Yes' if current_result['is_deepfake'] else 'No'}", 
                        transform=ax2.transAxes)
            
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(result.frame_results),
                                     interval=200, blit=False)
        
        # Save animation
        Writer = animation.writers['pillow']
        writer = Writer(fps=5, metadata=dict(artist='Deepfake Detector'), bitrate=1800)
        anim.save(f"{output_path}.gif", writer=writer)
        
        plt.close(fig)

def create_comprehensive_visualization_suite(result: VideoAnalysisResult,
                                           output_dir: str,
                                           config: VisualizationConfig = None) -> Dict[str, str]:
    """Create complete visualization suite for video analysis"""
    
    config = config or VisualizationConfig()
    visualizer = VideoTimelineVisualizer(config)
    
    output_files = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(result.video_path).stem
    
    # Timeline visualization
    timeline_fig = visualizer.create_confidence_timeline(result)
    timeline_path = output_path / f"{base_name}_timeline.html"
    timeline_fig.write_html(str(timeline_path))
    output_files['timeline'] = str(timeline_path)
    
    # Consistency heatmap
    heatmap_fig = visualizer.create_temporal_consistency_heatmap(result)
    heatmap_path = output_path / f"{base_name}_heatmap.html"
    heatmap_fig.write_html(str(heatmap_path))
    output_files['heatmap'] = str(heatmap_path)
    
    # Optical flow analysis
    flow_fig = visualizer.create_optical_flow_visualization(result)
    flow_path = output_path / f"{base_name}_optical_flow.html"
    flow_fig.write_html(str(flow_path))
    output_files['optical_flow'] = str(flow_path)
    
    # Frame analysis grid
    frame_fig = visualizer.create_frame_analysis_grid(result)
    frame_path = output_path / f"{base_name}_frames.png"
    frame_fig.savefig(str(frame_path), dpi=config.dpi, bbox_inches='tight')
    plt.close(frame_fig)
    output_files['frames'] = str(frame_path)
    
    # Export comprehensive report
    report_path = output_path / f"{base_name}_report"
    VideoExportUtilities.export_analysis_report(result, str(report_path), False)
    output_files['report'] = f"{report_path}.json"
    
    return output_files