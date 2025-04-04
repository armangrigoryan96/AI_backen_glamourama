import cv2
import math
import json
import base64
import numpy as np
from PIL import Image
import mediapipe as mp
from io import BytesIO
from werkzeug.datastructures import FileStorage

mp_body_segmentation = mp.solutions.selfie_segmentation

def file_storage_to_cv2(file: FileStorage) -> np.ndarray:
    if file.filename == '':
        raise ValueError("No file uploaded")

    img_bytes = file.read()
    if not img_bytes:
        raise ValueError("The file is empty")

    img_array = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_cv2 is None:
        raise ValueError("Failed to decode image")

    return img_cv2

def file_storage_to_cv2(file: FileStorage) -> np.ndarray:
    # Convert the FileStorage object to a byte stream
    img_bytes = file.read()
    
    # Convert the byte stream to a numpy array
    img_array = np.frombuffer(img_bytes, np.uint8)
    
    # Decode the image array into an OpenCV format
    img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    return img_cv2

def equalize_image_heights(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """
    Equalizes the heights of two images by resizing the second image to match the height of the first image.
    
    Args:
    - image1: First image (numpy ndarray).
    - image2: Second image (numpy ndarray).
    
    Returns:
    - tuple: A tuple of two images with the same height (image1, resized_image2).
    """
    # Get the height and width of the first image
    height1, width1 = image1.shape[:2]
    
    # Resize the second image to match the height of the first image, preserving the aspect ratio
    height2, width2 = image2.shape[:2]
    new_width = int((width2 / height2) * height1)  # Calculate the new width based on aspect ratio
    resized_image2 = cv2.resize(image2, (new_width, height1), interpolation=cv2.INTER_LINEAR)
    
    return image1, resized_image2

def segment_body(image: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_body_segmentation.SelfieSegmentation() as segmenter:
        results = segmenter.process(image_rgb)
        segmented_mask = (results.segmentation_mask * 255).astype(np.uint8)
        return cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2BGR)

def mes_height(segmented_image):
    # Check if the segmented image is empty or None
    if segmented_image is None or segmented_image.size == 0:
        print("Error: Segmented image is empty or None!")
        return 0

    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"Error converting image to grayscale: {e}")
        return 0

    try:
        # Find contours of the segmented image
        contours, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error as e:
        print(f"Error finding contours: {e}")
        return 0

    if not contours:
        return 0

    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return h

def measure_length(mask_image, y_value):
    # mask_image = segment_body(image_path)
    w = mask_image.shape[1]
    y_coordinate = y_value
    x_start = None
    x_end = None
    x_length = 0

    x1 = []
    for x in range(w):
        pixel_value = mask_image[y_coordinate][x]

        if np.array_equal(pixel_value, [255, 255, 255]):
            x_length += 1
            x1.append(x)

    z = []
    for x in range(x_length-1):
        if ((x1[x+1]-x1[x]) != 1):
            z.append(x+1)

    if len(z) == 0:
        return 0
    if len(z) >= 2:
        return (z[1]-z[0])
    if len(z) <= 2:
        return (x_length-z[0])

def measurelength(mask_image, y_value):
    # mask_image = segment_body(image_path)
    w = mask_image.shape[1]
    y_coordinate = y_value
    x_start = None
    x_end = None
    x_length = 0

    x1 = []
    for x in range(w):
        pixel_value = mask_image[y_coordinate][x]

        if np.array_equal(pixel_value, [255, 255, 255]):
            x_length += 1
    return x_length

# front_path = "processed_images/front.png"
# side_path = "processed_images/side.png"

def image_to_base64(image: np.ndarray) -> str:
    # Convert the image from ndarray to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Save the image to a BytesIO object
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    
    # Convert the BytesIO object to base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def calculate(front_image, side_image, Height) -> tuple:
        mask_image_front = segment_body(front_image)
        mask_image_side = segment_body(side_image)
        
        measure_height = mes_height(mask_image_front)

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        
        mp_drawing = mp.solutions.drawing_utils
        
        image = front_image
        image1 = side_image 


        image_rgb = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
        image_rgb1 = cv2.cvtColor(side_image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose model
        results = pose.process(image_rgb)
        results1 = pose.process(image_rgb1)
        # Check if pose landmarks are detected
        if results.pose_landmarks:
            height, width, _ = image.shape
            landmark_px = []
            for landmark in results.pose_landmarks.landmark:
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                landmark_px.append((x_px, y_px))
            # Calculate Euclidean distance between landmarks 11 (Left shoulder) and 12 (Right shoulder) for the first detected person

            x1, y1 = landmark_px[11]
            x2, y2 = landmark_px[12]
            x3, y3 = landmark_px[14]
            x4, y4 = landmark_px[16]
            x5, y5 = landmark_px[18]
            x6, y6 = landmark_px[24]
            x7, y7 = landmark_px[26]
            x8, y8 = landmark_px[28]
            x9, y9 = landmark_px[30]
            x10, y10 = landmark_px[23]
            x11, y11 = landmark_px[10]
            # Calculate Euclidean distance
            
            distance1 = measurelength(mask_image_front, y2)
            dis_shoulder = Height * distance1 / measure_height
            text_position = ((x1 + x2) // 2-30, (y1 + y2) // 2 - 10)
            METRICS_COLOR = (0, 0, 0)
            METRICS_SCALE = 0.5
            cv2.putText(img=image, text=f"shoulder", org=text_position,  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=METRICS_SCALE, color=METRICS_COLOR, thickness=2)
            
            distance2 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)+math.sqrt((x4 - x3)
                                                                        ** 2 + (y4 - y3)**2)+math.sqrt((x5 - x4)**2 + (y5 - y4)**2)
            text_position = ((x2 + x5) // 2-80, (y2 + y5) // 2 - 10)
            dis_arm = Height*distance2/measure_height
            cv2.putText(img=image, text=f"arm",
                        org=text_position,  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=METRICS_SCALE, color=METRICS_COLOR, thickness=2)

            tmp = int(y2*2/3+y6/3)
            distance3 = math.sqrt((x7 - x6)**2 + (y7 - tmp)**2) + \
                math.sqrt((x8 - x7)**2 + (y8 - y7)**2)
            dis_leg = Height*distance3/measure_height
            text_position = ((x6 + x8) // 2-40, (tmp + y8) // 2 - 10)
            cv2.putText(img=image, text=f"leg",
                        org=text_position,  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=METRICS_SCALE, color=METRICS_COLOR, thickness=2)

            distance4 = measure_length(mask_image_front, y6)
            distanceq4 = measurelength(mask_image_side, y6)
            dis4 = Height*(distance4+distanceq4)*2/measure_height

            text_position = ((x6 + x10) // 2-30, (y6 + y10) // 2 - 10)
            cv2.putText(img=image, text=f"hip",
                        org=text_position,  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=METRICS_SCALE, color=METRICS_COLOR, thickness=2)

            yy = int(y2*2/3+y6/3)
            distance5 = measurelength(mask_image_front, yy)
            distanceq5 = measurelength(mask_image_side, yy)
            dis_bust = Height*(distance5+distanceq5)*2*0.7/measure_height

            text_position = ((x6 + x10) // 2-30, (y6+y10)//2-200)
            cv2.putText(img=image, text=f"backborn",
                        org=text_position,  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=METRICS_SCALE, color=METRICS_COLOR, thickness=2)
            zz = int(y2*1/3+y6*2/3)
            distance6 = measurelength(mask_image_front, zz)
            distanceq6 = measurelength(mask_image_side, zz)
            dis_waist = Height*(distance6+distanceq6)*2*0.7/measure_height

            text_position = ((x6 + x10) // 2-30, ((y6+y10)//2 - 400))
            cv2.putText(img=image, text=f"chest",
                        org=text_position,  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=METRICS_SCALE, color=METRICS_COLOR, thickness=2)

            dis_neck = (dis_shoulder+dis_bust/3)/2
            text_position = ((x2 + x6) // 2-30, ((y2+y6)//2 - 400))
            cv2.putText(img=image, text=f"neck",
                        org=text_position,  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=METRICS_SCALE, color=METRICS_COLOR, thickness=2)

            # dis8 = (y6 - (y2 + y11) / 2.2)*Height/measure_height

            dis_upperbody = (y6-y11/2-y2/2+30)*Height/measure_height
            dis_inseam = (y9-y6)*Height/measure_height-10
            dis_thigh = measurelength(mask_image_front,
                                    y6+int(10*measure_height/Height))/2*Height/measure_height*3
            dis_biceps = dis_thigh/2+5
            dis_cuffs = dis_thigh/3+3

            # Render the pose landmarks on the image
            annotated_image = image.copy()
            sidebar_image = image1.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                sidebar_image, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # # Save the annotated image
            # output_path_front = 'processed_images/output_pose_front.jpg'
            # output_path_side = 'processed_images/output_pose_side.jpg'
            # cv2.imwrite(output_path_front, annotated_image)
            # cv2.imwrite(output_path_side, sidebar_image)

        im = image
        width = im.shape[1]
        height = im.shape[0]
        background_color = (255, 255, 255)
        image = np.ones((height, width, 3), dtype=np.uint8) * background_color

        # Define text color
        text_color = (0, 0, 0)  # Black

        # Choose font type and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # Adjusted font scale for better readability

        # Starting text position
        text_x = int(width/4)
        text_y = int(height/3)

        # List of body parts and their corresponding values
        body_parts = [
            ("Neck", int(dis_neck)),
            ("Shoulder", int(dis_shoulder)),
            ("Arm", int(dis_arm)),
            ("Leg", int(dis_leg)),
            ("Upper body", int(dis_upperbody)),
            ("Hip", int(dis4)),
            ("Bust", int(dis_bust)),
            ("Waist", int(dis_waist)),
            ("Inseam", int(dis_inseam)),
            ("Thigh", int(dis_thigh)),
            ("Biceps", int(dis_biceps)),
            ("Cuffs", int(dis_cuffs)),
        ]
        
        # Create a dictionary from the list
        body_parts_dict = dict(body_parts)

        # Specify the path where you want to save the JSON file
        json_file_path = 'body_parts.json'

        # Write the dictionary to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(body_parts_dict, json_file, indent=4)
        # return jsonify(body_parts_dict)
        result_text = ""
        for part, value in body_parts:
            result_text += f"{part}:"+" "+ f"{value}cm" + "\n"
        
        # Draw each body part and value on a new line
        img_copy = image.astype(np.uint8)
        for part, value in body_parts:
            text = f"{part}: {value}cm"
            height, width, dim = img_copy.shape
            
            cv2.putText(img=img_copy, text=text, org=(text_x, text_y),  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=text_color, thickness=2, lineType=cv2.LINE_AA)
            
            text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
            text_y += text_size[1] + 50  # Move to the next line with a 10-pixel gap

        if img_copy.dtype != np.uint8:
            img_copy = img_copy.astype(np.uint8)

        img_copy = np.clip(img_copy, 0, 255)
        annotated_image, sidebar_image = equalize_image_heights(annotated_image, sidebar_image)
        
        final_res = cv2.hconcat([annotated_image, sidebar_image, img_copy])
        body_parts_result = [{"name": name, "measurement": measurement} for name, measurement in body_parts]
        return image_to_base64(final_res), body_parts_result