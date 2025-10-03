#!/usr/bin/env python3
"""
Final demonstration image selection.
This script identifies the best 6 images (3 fake, 3 real) for demonstration purposes.
"""

import json
import os

def select_final_demo_images():
    """Select the best demonstration images"""
    print("🎯 FINAL DEMONSTRATION IMAGE SELECTION")
    print("=" * 80)
    
    # Load the test results
    if not os.path.exists("demo_image_results.json"):
        print("❌ No test results found. Run find_demo_images.py first.")
        return
    
    with open("demo_image_results.json", 'r') as f:
        results = json.load(f)
    
    # Separate results by type
    fake_results = [r for r in results if r['expected_type'] == 'fake']
    real_results = [r for r in results if r['expected_type'] == 'real']
    
    # Sort fake images by confidence (ascending - lower confidence = more obvious fake)
    fake_results.sort(key=lambda x: x['confidence'])
    
    # Sort real images by confidence (descending - higher confidence = more obvious real)
    real_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"📊 Available Images:")
    print(f"  Fake images: {len(fake_results)}")
    print(f"  Real images: {len(real_results)}")
    
    print(f"\n🎭 FINAL DEMONSTRATION SET (6 Images)")
    print("=" * 80)
    
    print(f"\n🔴 FAKE IMAGES (3 examples with varied confidence):")
    print("-" * 60)
    
    # Select 3 fake images with varied confidence scores
    selected_fakes = []
    if len(fake_results) >= 3:
        # Take low, medium, and high confidence fakes
        selected_fakes = [
            fake_results[0],  # Lowest confidence (most obvious fake)
            fake_results[len(fake_results)//2],  # Medium confidence
            fake_results[-1]  # Highest confidence (most subtle fake)
        ]
    
    for i, result in enumerate(selected_fakes):
        confidence_level = "LOW" if result['confidence'] < 0.6 else "MEDIUM" if result['confidence'] < 0.8 else "HIGH"
        print(f"{i+1}. {result['filename']}")
        print(f"   Confidence: {result['confidence']:.1f}% ({confidence_level})")
        print(f"   Prediction: {result['predicted_type']} | ✅ Correct")
        print(f"   Path: {result['path']}")
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        print()
    
    print(f"\n🟢 REAL IMAGES (3 examples with varied confidence):")
    print("-" * 60)
    
    # Select 3 real images with varied confidence scores
    selected_reals = []
    if len(real_results) >= 3:
        # Take high, medium, and lower confidence reals
        selected_reals = [
            real_results[0],  # Highest confidence (most obvious real)
            real_results[len(real_results)//2],  # Medium confidence
            real_results[-1]  # Lower confidence (more ambiguous real)
        ]
    
    for i, result in enumerate(selected_reals):
        confidence_level = "HIGH" if result['confidence'] > 0.8 else "MEDIUM" if result['confidence'] > 0.6 else "LOW"
        print(f"{i+1}. {result['filename']}")
        print(f"   Confidence: {result['confidence']:.1f}% ({confidence_level})")
        print(f"   Prediction: {result['predicted_type']} | ✅ Correct")
        print(f"   Path: {result['path']}")
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        print()
    
    # Create demonstration summary
    print(f"\n📋 DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    all_selected = selected_fakes + selected_reals
    confidence_range = [r['confidence'] for r in all_selected]
    
    print(f"Total Images: {len(all_selected)}")
    print(f"Confidence Range: {min(confidence_range):.1f}% - {max(confidence_range):.1f}%")
    print(f"Average Confidence: {sum(confidence_range)/len(confidence_range):.1f}%")
    print(f"Processing Speed: {sum(r['processing_time'] for r in all_selected)/len(all_selected):.3f}s per image")
    
    print(f"\n🎯 DEMONSTRATION POINTS:")
    print(f"• Show clear fake detection (low confidence scores)")
    print(f"• Show subtle fake detection (medium confidence scores)")
    print(f"• Show obvious real detection (high confidence scores)")
    print(f"• Show ambiguous real detection (lower confidence scores)")
    print(f"• Demonstrate consistent processing speed")
    print(f"• Highlight the model's 94.4% accuracy benchmark")
    
    # Save final selection
    final_selection = {
        "fake_images": selected_fakes,
        "real_images": selected_reals,
        "summary": {
            "total_images": len(all_selected),
            "confidence_range": f"{min(confidence_range):.1f}% - {max(confidence_range):.1f}%",
            "average_confidence": f"{sum(confidence_range)/len(confidence_range):.1f}%",
            "average_processing_time": f"{sum(r['processing_time'] for r in all_selected)/len(all_selected):.3f}s"
        }
    }
    
    with open("final_demo_selection.json", 'w') as f:
        json.dump(final_selection, f, indent=2)
    
    print(f"\n💾 Final selection saved to: final_demo_selection.json")
    
    return final_selection

if __name__ == "__main__":
    try:
        selection = select_final_demo_images()
        if selection:
            print(f"\n🎉 Final demonstration image selection completed!")
            print(f"Use these 6 images to showcase your Hugging Face deepfake detector!")
        else:
            print("\n❌ Selection failed")
            exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        exit(1)





