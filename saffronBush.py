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

def find_and_draw_groups(image, mask):  
    """Find contours and draw a bounding box around all detected flowers as one group."""  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]  # Adjusted minimum area  

    if filtered_contours:  
        draw_group(image, filtered_contours)  # Draw a single group for all flowers  

def draw_group(image, contours):  
    """Draw a bounding box around all contours and annotate."""  
    x_min = min(cv2.boundingRect(cnt)[0] for cnt in contours)  
    y_min = min(cv2.boundingRect(cnt)[1] for cnt in contours)  
    x_max = max(cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt in contours)  
    y_max = max(cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in contours)  

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle  
    count = len(contours)  
    centroid_x = (x_min + x_max) // 2  
    centroid_y = (y_min + y_max) // 2  

    cv2.putText(image, f"Count: {count}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
    cv2.putText(image, f"Centroid: ({centroid_x}, {centroid_y})", (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  

def main(file_path, lower_hue, upper_hue):  
    """Main function to load an image, detect objects, and save/display the result."""  
    try:  
        image = load_image(file_path)  
        hsi_image = convert_to_hsv(image)  
        mask = create_mask(hsi_image, lower_hue, upper_hue)  
        mask = perform_morphological_operations(mask)  

        find_and_draw_groups(image, mask)  
        
        output_file_path = 'E:/saffronImageProcess/Grouped_Saffron_Flowers.jpg'  
        cv2.imwrite(output_file_path, image)  
        print(f"Output image saved to {output_file_path}")  
    except Exception as e:  
        print(e)  

# Define color ranges for saffron flower  
saffron_lower_hue = np.array([120, 100, 100])  # Adjusted lower hue range  
saffron_upper_hue = np.array([140, 255, 255])  # Adjusted upper hue range  

# Call the main function for saffron  
main('E:/saffronImageProcess/Source/3.jpg', saffron_lower_hue, saffron_upper_hue)