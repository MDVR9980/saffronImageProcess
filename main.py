import cv2  
import numpy as np  

def load_image(file_path):  
    """Load an image from the specified file path."""  
    image = cv2.imread(file_path)  
    if image is None:  
        print(f"Error: Could not load image from {file_path}. Check the file path.")  
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

def find_and_draw_contours(image, mask, z_value):  
    """Find contours in the mask, draw them on the image, and annotate with coordinates."""  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  
    
    for cnt in filtered_contours:
        # Draw a bounding box around each detected contour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle
        
        # Calculate centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Annotate with the centroid coordinates
            cv2.putText(image, f"({cX}, {cY}, {z_value})", (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
    return filtered_contours  

def get_centroids(contours):  
    """Calculate the centroids of the detected contours."""  
    centroids = []  
    for cnt in contours:  
        M = cv2.moments(cnt)  
        if M["m00"] != 0:  
            cX = int(M["m10"] / M["m00"])  
            cY = int(M["m01"] / M["m00"])  
            centroids.append((cX, cY))  
    return centroids  

def main(file_path, lower_hue, upper_hue, z_value):  
    """Main function to load an image, detect objects, and save/display the result."""  
    image = load_image(file_path)  
    if image is not None:  
        hsi_image = convert_to_hsv(image)  
        mask = create_mask(hsi_image, lower_hue, upper_hue)  
        mask = perform_morphological_operations(mask)  
        contours = find_and_draw_contours(image, mask, z_value)  
        
        centroids = get_centroids(contours)  
        
        for idx, (x, y) in enumerate(centroids):  
            print(f"Contour {idx + 1}: Centroid at (x: {x} pixels, y: {y} pixels, z: {z_value} units)")  
        
        # Save the result image
        output_file_path = 'E:/Saffron/Detected_Saffron_Flowers.jpg'
        cv2.imwrite(output_file_path, image)
        print(f"Output image saved to {output_file_path}")

# Define color ranges for saffron flower  
saffron_lower_hue = np.array([120, 50, 50])  # Adjust these values  
saffron_upper_hue = np.array([140, 255, 255])  # Adjust these values  

# Define a constant z-value (you can adjust this based on your context)  
z_value = 10  # Example z-coordinate in arbitrary units  

# Call the main function for saffron  
main('E:/Saffron/R.jpg', saffron_lower_hue, saffron_upper_hue, z_value)