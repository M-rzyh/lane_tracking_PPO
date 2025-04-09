import cv2
import numpy as np
import statistics
# from vehicle import Driver
# from controller import Camera, Display, GPS, Keyboard, Lidar

previous_left_distance = []
previous_right_distance = []

# left_distances = []
# right_distances = []

def process_image(image):#(image, camera):
    
    height, width = 100,300 #camera.getHeight(), camera.getWidth()
    image = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Debug: Show the edge detection result
    cv2.imshow("Edges", edges)
    cv2.waitKey(1)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=50)
    return lines

def classify_and_measure_distances(lines, image_width):
    left_distances = []
    right_distances = []
    
    # global left_distances
    # global right_distances
    
    global previous_left_distance
    global previous_right_distance
    
    center_x = image_width // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0  # Avoid division by zero
            # Calculate the distance to the midpoint of the line
            midpoint_x = (x1 + x2) / 2
            distance = abs(midpoint_x - center_x)

            if slope < 0:  # Assuming negative slope for left lines
                left_distances.append(distance)
            elif slope > 0:  # Assuming positive slope for right lines
                right_distances.append(distance)

    # Return the average or minimum distance


    
    # Use min or max based on what is suitable for your use case
    # left_distance = min(left_distances) if left_distances else None
    # right_distance = min(right_distances) if right_distances else None

    # # Update history if new valid distances were calculated
    # if left_distance is not None:
    #     previous_left_distance.append(left_distance)
    #     if len(previous_left_distance) > 10:
    #         previous_left_distance.pop(0)
    # else:
    #     # Handle case when no left line is detected
    #     left_distance = statistics.mean(previous_left_distance) if previous_left_distance else 0

    # if right_distance is not None:
    #     previous_right_distance.append(right_distance)
    #     if len(previous_right_distance) > 10:
    #         previous_right_distance.pop(0)
    # else:
    #     # Handle case when no right line is detected
    #     right_distance = statistics.mean(previous_right_distance) if previous_right_distance else 0

    # return left_distance, right_distance
    
    if left_distances:
        left_distance = min(left_distances)
        previous_left_distance.append(left_distance)

    elif previous_left_distance:
        left_distance = max(previous_left_distance)
    else:
        left_distance = 140
        
    if right_distances:
        right_distance = statistics.mean(right_distances)
        previous_right_distance.append(right_distance)
        
    elif previous_right_distance:
        right_distance = max(previous_right_distance)
    else:
        right_distance = 140
    
    previous_left_distance = previous_left_distance[-5:]
    previous_right_distance = previous_right_distance[-5:]
    
    # left_distance = min(left_distances) if left_distances else max(previous_left_distance)
    # right_distance = min(right_distances) if right_distances else max(previous_right_distance)
    # previous_left_distance.append(left_distance)
    # previous_right_distance.append(right_distance)
    # if left_distance is not None:
    #     previous_left_distance.append(left_distance)
    #     if len(previous_left_distance) > 10:
    #         previous_left_distance.pop(0)

    # if right_distance is not None:
    #     previous_right_distance.append(right_distance)
    #     if len(previous_right_distance) > 10:
    #         previous_right_distance.pop(0)
    return left_distance, right_distance


def classify_and_measure_distances_new(lines, image_width):
    left_distances = []
    right_distances = []

    global previous_left_distance
    global previous_right_distance

    center_x = image_width // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            midpoint_x = (x1 + x2) / 2
            distance = abs(midpoint_x - center_x)

            if midpoint_x < center_x and slope < 0:
                left_distances.append(distance)
            elif midpoint_x > center_x and slope > 0:
                right_distances.append(distance)

    # --- Determine left_distance ---
    if not left_distances and right_distances:
        left_distance = 200  # No left line, default fallback
    elif left_distances:
        left_distance = min(left_distances)
        previous_left_distance.append(left_distance)
    elif previous_left_distance:
        left_distance = max(previous_left_distance)
    else:
        left_distance = 140

    # --- Determine right_distance ---
    if not right_distances and left_distances:
        right_distance = 200  # No right line, default fallback
    elif right_distances:
        right_distance = min(right_distances)
        previous_right_distance.append(right_distance)
    elif previous_right_distance:
        right_distance = max(previous_right_distance)
    else:
        right_distance = 140

    # Keep only last 5 distances in memory
    previous_left_distance = previous_left_distance[-5:]
    previous_right_distance = previous_right_distance[-5:]

    return left_distance, right_distance