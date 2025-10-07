import React, { forwardRef } from 'react';

const InteractiveImageViewer = forwardRef(({ 
  image, 
  zoomLevel = 1, 
  position = { x: 0, y: 0 }, 
  onPositionChange = () => {}, 
  onHover = () => {},
  className = '',
  children 
}, ref) => {
  const handleMouseMove = (e) => {
    if (onHover) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      onHover({ x, y });
    }
  };

  const handleWheel = (e) => {
    e.preventDefault();
    // Basic zoom handling - could be enhanced
  };

  return (
    <div 
      ref={ref}
      className={`interactive-image-viewer ${className}`}
      onMouseMove={handleMouseMove}
      onWheel={handleWheel}
      style={{
        position: 'relative',
        overflow: 'hidden',
        transform: `scale(${zoomLevel})`,
        transformOrigin: 'center center'
      }}
    >
      {image && (
        <img 
          src={image} 
          alt="Analysis target"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            transform: `translate(${position.x}px, ${position.y}px)`
          }}
        />
      )}
      {children}
    </div>
  );
});

InteractiveImageViewer.displayName = 'InteractiveImageViewer';

export default InteractiveImageViewer;
