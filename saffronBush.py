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

def find_and_draw_groups(image, mask):  
    """Find contours, group them, and draw a bounding box around each group."""  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    # Filter contours based on area  
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  

    groups = []  
    visited = [False] * len(filtered_contours)  

    for i in range(len(filtered_contours)):  
        if not visited[i]:  
            group = [filtered_contours[i]]  
            visited[i] = True  

            for j in range(i + 1, len(filtered_contours)):  
                if not visited[j]:  
                    # Check if any contour in group is close to contour j  
                    if is_contour_near(group, filtered_contours[j]):  
                        group.append(filtered_contours[j])  
                        visited[j] = True  

            groups.append(group)  

    # Draw groups and report counts  
    for group in groups:  
        draw_group(image, group)  

    return groups  

def is_contour_near(group, contour):  
    """Determine if a contour is near any contour in the existing group."""  
    group_centroids = [cv2.boundingRect(cnt) for cnt in group]   
    x, y, w, h = cv2.boundingRect(contour)  
    centroid = (x + w // 2, y + h // 2)  # center of the current contour  

    for gx, gy, gw, gh in group_centroids:  
        group_centroid = (gx + gw // 2, gy + gh // 2)  
        if np.linalg.norm(np.array(centroid) - np.array(group_centroid)) < 100:  # Adjust distance threshold  
            return True  

    return False  

def draw_group(image, group):  
    """Draw a bounding box around the group and annotate."""  
    x_min = min(cv2.boundingRect(cnt)[0] for cnt in group)  
    y_min = min(cv2.boundingRect(cnt)[1] for cnt in group)  
    x_max = max(cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt in group)  
    y_max = max(cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in group)  

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # Yellow rectangle  
    count = len(group)  
    centroid_x = (x_min + x_max) // 2  
    centroid_y = (y_min + y_max) // 2  

    cv2.putText(image, f"Count: {count}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  
    cv2.putText(image, f"Centroid: ({centroid_x}, {centroid_y})", (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  

def main(file_path, lower_hue, upper_hue):  
    """Main function to load an image, detect objects, and save/display the result."""  
    image = load_image(file_path)  
    if image is not None:  
        hsi_image = convert_to_hsv(image)  
        mask = create_mask(hsi_image, lower_hue, upper_hue)  
        mask = perform_morphological_operations(mask)  

        groups = find_and_draw_groups(image, mask)  
        
        # Save the result image  
        output_file_path = 'E:/saffronImageProcess/Grouped_Saffron_Flowers.jpg'  
        cv2.imwrite(output_file_path, image)  
        print(f"Output image saved to {output_file_path}")  

# Define color ranges for saffron flower  
saffron_lower_hue = np.array([120, 100, 100])  # Adjust these values  
saffron_upper_hue = np.array([140, 255, 255])  # Adjust these values  

# Call the main function for saffron  
main('E:/saffronImageProcess/Source/3.jpg', saffron_lower_hue, saffron_upper_hue)