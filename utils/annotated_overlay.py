"""
AcneVision: Annotated Face Overlay Module
==========================================
Draws colored detection markers directly on uploaded face images.
Creates professional dermatology-style visualizations.
"""

import cv2
import numpy as np
import random

def draw_acne_overlay(face_image, skin_features, save_path):
    """
    Draw colored circles and labels on detected acne spots.
    
    Parameters:
    - face_image: RGB numpy array of the face
    - skin_features: Dictionary from analyze_skin_features()
    - save_path: Where to save the annotated image
    
    Returns:
    - save_path if successful, None if failed
    """
    # Convert RGB to BGR for OpenCV drawing
    img = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # Create a copy to draw on
    overlay = img.copy()
    
    # Get detection regions from skin_features
    blackheads = skin_features.get('blackheads', {}).get('regions', [])
    whiteheads = skin_features.get('whiteheads', {}).get('regions', [])
    pustules = skin_features.get('pustules', {}).get('regions', [])
    papules = skin_features.get('papules', {}).get('regions', [])
    hyperpig = skin_features.get('hyperpigmentation', {}).get('regions', [])
    
    # Draw each type with different colors
    # Pustules = RED (inflamed, pus-filled)
    for i, (x, y, bw, bh) in enumerate(pustules[:8]):  # Limit to 8 spots
        center = (x + bw//2, y + bh//2)
        radius = max(bw, bh) // 2 + 5
        cv2.circle(overlay, center, radius, (0, 0, 255), 2)  # Red outline
        cv2.circle(overlay, center, radius-2, (0, 0, 255), -1)  # Red fill (transparent)
        
        # Add label with line
        label_x = x - 80 if x > w//2 else x + bw + 10
        label_y = y - 20 if y > h//2 else y + bh + 30
        draw_label_with_line(overlay, center, (label_x, label_y), f"Pustule {i+1}", (0, 0, 255))
    
    # Papules = ORANGE-RED (inflamed, no pus)
    for i, (x, y, bw, bh) in enumerate(papules[:8]):
        center = (x + bw//2, y + bh//2)
        radius = max(bw, bh) // 2 + 5
        cv2.circle(overlay, center, radius, (0, 100, 255), 2)
        
        label_x = x - 80 if x > w//2 else x + bw + 10
        label_y = y - 30 if y > h//2 else y + bh + 40
        draw_label_with_line(overlay, center, (label_x, label_y), f"Papule {i+1}", (0, 100, 255))
    
    # Blackheads = GREEN (dark spots)
    for i, (x, y, bw, bh) in enumerate(blackheads[:10]):
        center = (x + bw//2, y + bh//2)
        radius = max(bw, bh) // 2 + 3
        cv2.circle(overlay, center, radius, (0, 255, 0), 2)
        
        label_x = x - 90 if x > w//2 else x + bw + 10
        label_y = y + bh + 20
        draw_label_with_line(overlay, center, (label_x, label_y), f"Blackhead {i+1}", (0, 255, 0))
    
    # Whiteheads = YELLOW-GREEN (light spots)
    for i, (x, y, bw, bh) in enumerate(whiteheads[:10]):
        center = (x + bw//2, y + bh//2)
        radius = max(bw, bh) // 2 + 3
        cv2.circle(overlay, center, radius, (0, 255, 200), 2)
        
        label_x = x - 100 if x > w//2 else x + bw + 10
        label_y = y - 15
        draw_label_with_line(overlay, center, (label_x, label_y), f"Whitehead {i+1}", (0, 255, 200))
    
    # Hyperpigmentation = PURPLE (dark marks)
    for i, (x, y, bw, bh) in enumerate(hyperpig[:6]):
        center = (x + bw//2, y + bh//2)
        radius = max(bw, bh) // 2 + 5
        cv2.circle(overlay, center, radius, (255, 0, 255), 2)
        
        label_x = x - 120 if x > w//2 else x + bw + 10
        label_y = y + bh + 50
        draw_label_with_line(overlay, center, (label_x, label_y), f"Dark Spot {i+1}", (255, 0, 255))
    
    # Add semi-transparent overlay effect
    alpha = 0.7
    overlay = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Add title bar at top
    add_title_bar(overlay, skin_features)
    
    # Add stats panel on left side
    add_stats_panel(overlay, skin_features)
    
    # Save result
    cv2.imwrite(save_path, overlay)
    print(f"[INFO] Annotated image saved: {save_path}")
    
    return save_path


def draw_label_with_line(img, start_point, end_point, text, color):
    """Draw a line from circle to text label."""
    # Draw line
    cv2.line(img, start_point, end_point, color, 1, cv2.LINE_AA)
    
    # Draw text background
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    padding = 3
    cv2.rectangle(img, 
                  (end_point[0] - padding, end_point[1] - text_h - padding),
                  (end_point[0] + text_w + padding, end_point[1] + padding),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(img, text, (end_point[0], end_point[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def add_title_bar(img, skin_features):
    """Add title bar at top of image."""
    h, w = img.shape[:2]
    bar_height = 40
    
    # Draw semi-transparent black bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    
    # Title text
    severity = skin_features.get('overall_severity', 'Analysis')
    title = f"AcneVision AI Analysis - {severity}"
    cv2.putText(img, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2, cv2.LINE_AA)


def add_stats_panel(img, skin_features):
    """Add statistics panel on left side of image."""
    h, w = img.shape[:2]
    panel_width = 200
    panel_height = 280
    x = 10
    y = h - panel_height - 10
    
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
    
    # Border
    cv2.rectangle(img, (x, y), (x + panel_width, y + panel_height), (100, 100, 100), 1)
    
    # Stats text
    stats = [
        ("DIAGNOSIS RESULTS", (255, 255, 255), 0.5, 2),
        ("", (255, 255, 255), 0.4, 1),
        (f"Pustules: {skin_features.get('pustules', {}).get('count', 0)}", (0, 100, 255), 0.4, 1),
        (f"Papules: {skin_features.get('papules', {}).get('count', 0)}", (0, 150, 255), 0.4, 1),
        (f"Blackheads: {skin_features.get('blackheads', {}).get('count', 0)}", (0, 255, 0), 0.4, 1),
        (f"Whiteheads: {skin_features.get('whiteheads', {}).get('count', 0)}", (0, 255, 200), 0.4, 1),
        (f"Dark Spots: {skin_features.get('hyperpigmentation', {}).get('count', 0)}", (255, 0, 255), 0.4, 1),
        ("", (255, 255, 255), 0.4, 1),
        (f"Redness: {skin_features.get('redness', {}).get('severity', 'N/A')}", (0, 100, 255), 0.4, 1),
        (f"Texture: {skin_features.get('texture_roughness', {}).get('smoothness', 'N/A')}", (200, 200, 200), 0.4, 1),
        ("", (255, 255, 255), 0.4, 1),
        (f"Health Score: {skin_features.get('skin_health_score', 0)}/100", (255, 255, 0), 0.45, 1),
    ]
    
    line_y = y + 30
    for text, color, scale, thickness in stats:
        cv2.putText(img, text, (x + 10, line_y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        line_y += 22


def create_annotated_image(face_image, skin_features, save_path):
    """
    Main function to create annotated face image.
    Wrapper for draw_acne_overlay.
    """
    try:
        result_path = draw_acne_overlay(face_image, skin_features, save_path)
        return result_path
    except Exception as e:
        print(f"[ERROR] Failed to create annotated image: {e}")
        import traceback
        traceback.print_exc()
        return None