import cv2
import numpy as np

def load_image(file_path):
    """Load an image from the specified file path."""
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Error: Could not load image from {file_path}. Check the file path.")
    return image

def convert_to_hsv(image):
    """Convert the image to HSV color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def create_mask(hsi_image, lower_hue, upper_hue):
    """Create a mask for the specified color range."""
    return cv2.inRange(hsi_image, lower_hue, upper_hue)

def perform_morphological_operations(mask):
    """Perform morphological operations to clean up the mask."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def calculate_distance(rect1, rect2):
    """Calculate distance between two rectangles (bounding boxes)."""
    center1 = ((rect1[0] + rect1[2] // 2), (rect1[1] + rect1[3] // 2))
    center2 = ((rect2[0] + rect2[2] // 2), (rect2[1] + rect2[3] // 2))
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def find_and_draw_groups(image, mask, max_group_size=1000, overlap_threshold=500):
    """Find contours and draw bounding boxes around grouped flowers."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]  # Minimum area filter

    # List to hold bounding boxes of detected groups
    group_boxes = []
    group_id = 1

    # Iterate through contours and group them based on bounding box overlap
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        new_group = True

        # Check for overlap with existing group boxes
        for group in group_boxes:
            group_rect = group['rect']
            if calculate_distance((x, y, w, h), group_rect) < overlap_threshold:
                # Merge with existing group
                group['contours'].append(cnt)
                group['rect'] = (min(group_rect[0], x), 
                                 min(group_rect[1], y), 
                                 max(group_rect[0] + group_rect[2], x + w) - min(group_rect[0], x), 
                                 max(group_rect[1] + group_rect[3], y + h) - min(group_rect[1], y))
                new_group = False
                break

        if new_group:
            # Create a new group
            group_boxes.append({'rect': (x, y, w, h), 'contours': [cnt]})
            draw_group(image, [cnt], group_id)
            group_id += 1

    # Draw updated groups after merging
    for group in group_boxes:
        draw_group(image, group['contours'], group_id)
        group_id += 1

def draw_group(image, contours, group_id):
    """Draw a bounding box around all contours in a group and annotate with group ID."""
    x_min = min(cv2.boundingRect(cnt)[0] for cnt in contours)
    y_min = min(cv2.boundingRect(cnt)[1] for cnt in contours)
    x_max = max(cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt in contours)
    y_max = max(cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in contours)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle
    centroid_x = (x_min + x_max) // 2
    centroid_y = (y_min + y_max) // 2

    # Display group ID and centroid coordinates
    cv2.putText(image, f"Group {group_id} - Centroid: ({centroid_x}, {centroid_y})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main(file_path, lower_hue, upper_hue):
    """Main function to load an image, detect objects, and save/display the result."""
    try:
        image = load_image(file_path)
        hsi_image = convert_to_hsv(image)
        mask = create_mask(hsi_image, lower_hue, upper_hue)
        mask = perform_morphological_operations(mask)

        # Detect and group saffron flowers
        find_and_draw_groups(image, mask)

        output_file_path = 'E:/saffronImageProcess/Grouped_Saffron_Flowers_NoOverlap.jpg'
        cv2.imwrite(output_file_path, image)
        print(f"Output image saved to {output_file_path}")
    except Exception as e:
        print(e)

# Define color ranges for saffron flower
saffron_lower_hue = np.array([120, 100, 100])  # Adjusted lower hue range
saffron_upper_hue = np.array([140, 255, 255])  # Adjusted upper hue range

# Call the main function for saffron
main('E:/saffronImageProcess/Source/10.jpg', saffron_lower_hue, saffron_upper_hue)
