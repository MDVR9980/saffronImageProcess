import cv2  
import numpy as np  

def load_image(file_path):  
    """Load an image from the specified file path."""  
    image = cv2.imread(file_path)  
    if image is None:  
        print(f"Error: Could not load image from {file_path}. Check the file path.")  
    return image  

def apply_gaussian_filter(image):  
    """Apply Gaussian filter to reduce noise."""  
    return cv2.GaussianBlur(image, (5, 5), 0)  

def apply_median_filter(image):  
    """Apply Median filter to reduce noise."""  
    return cv2.medianBlur(image, 5)  

def adaptive_color_segmentation(image):  
    """Perform adaptive color segmentation using K-means clustering."""  
    pixel_values = image.reshape((-1, 3))  
    pixel_values = np.float32(pixel_values)  
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  
    k = 2  # Number of clusters  
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  
    
    centers = np.uint8(centers)  
    labels = labels.flatten()  
    
    segmented_image = centers[labels.flatten()]  
    segmented_image = segmented_image.reshape(image.shape)  
    
    return segmented_image  

def create_mask(image):  
    """Create a mask from the segmented image based on a color range."""  
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
    mask = cv2.inRange(hsv_image, (120, 50, 50), (140, 255, 255))  # Example hue range  
    return mask  

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
        x, y, w, h = cv2.boundingRect(cnt)  
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle  
        
        M = cv2.moments(cnt)  
        if M["m00"] != 0:  
            cX = int(M["m10"] / M["m00"])  
            cY = int(M["m01"] / M["m00"])  
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

def compute_z_value(depth_map, cX, cY):  
    """Compute Z value using a depth map."""  
    return depth_map[cY, cX]  

def edge_detection(image):  
    """Perform Canny edge detection."""  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    edges = cv2.Canny(gray_image, 100, 200)  # Perform Canny edge detection  
    return edges  

def main(file_path, depth_map_path):  
    """Main function to load an image, detect objects, and save/display the result."""  
    image = load_image(file_path)  
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)  # Load depth map  
    
    if image is not None and depth_map is not None:  
        # Apply noise reduction  
        filtered_image = apply_gaussian_filter(image)  
        filtered_image = apply_median_filter(filtered_image)  
        
        # Adaptive segmentation  
        segmented_image = adaptive_color_segmentation(filtered_image)  
        
        # Create mask for the segmented image  
        mask = create_mask(segmented_image)  
        mask = perform_morphological_operations(mask)  
        
        # Find and draw contours  
        contours = find_and_draw_contours(image, mask, z_value=10)  
        centroids = get_centroids(contours)  
        
        # Calculate Z-value and print centroid info  
        for idx, (x, y) in enumerate(centroids):  
            z_value = compute_z_value(depth_map, x, y)  
            print(f"Contour {idx + 1}: Centroid at (x: {x}, y: {y}, z: {z_value})")  
        
        # Perform edge detection  
        edges = edge_detection(image)  
        
        # Combine original image with edges for visualization  
        combined_image = cv2.addWeighted(image, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)  
        
        # Save the result image  
        output_file_path = 'E:/Saffron/Detected_Saffron_Flowers_Enhanced.jpg'  
        cv2.imwrite(output_file_path, combined_image)  
        print(f"Output image saved to {output_file_path}")  

# Example usage  
main('E:/Saffron/R.jpg', 'E:/Saffron/depth_map.jpg')