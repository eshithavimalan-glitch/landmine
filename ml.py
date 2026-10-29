import cv2, time
from inference_sdk import InferenceHTTPClient,InferenceConfiguration

import cv2
import numpy as np

def img_to_thermal_with_hot_circles(image_path, output_path='thermal_output.jpg'):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 11)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=200
    )

# heat
    heat_mask = np.zeros_like(gray)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(heat_mask, center, radius, 255, -1)

            cv2.circle(heat_mask, center, int(radius*1.2), 100, -1)

    enhanced_gray = cv2.max(gray, heat_mask)

    thermal_img = cv2.applyColorMap(enhanced_gray, cv2.COLORMAP_HOT)

    cv2.imwrite(output_path, thermal_img)
    new_width = 500
    new_height = 300
    resized_dim = (new_width, new_height)
    resized_image = cv2.resize(img, resized_dim, interpolation=cv2.INTER_AREA)
    resized_th_image = cv2.resize(thermal_img, resized_dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow(resized_image)
    cv2.imshow("thermal",resized_th_image)


def ml(image_path):
    client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="wFTqKNe6L4Ms2PdmOQYn").configure(InferenceConfiguration(api_key_transport="header"))

    img = cv2.imread("arkaimage.jpg")

    result = client.run_workflow(
        workspace_name="arkasaha",
        workflow_id="custom-workflow",
        images={
            "image": image_path #pohoto
        },
        use_cache=True
    )

    prediction = result[0]['predictions']['predictions']
    # cv2.imshow('image',img)
    # print(prediction, end="\n")
    if not prediction:
        print("Not a landmine")
        return 0
    else:
        print(f"Confidence: {prediction[0]['confidence']}\nClass: {prediction[0]['class']}")
        return 1
    # print(prediction)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error1")
    exit()

print("Camera opened")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error2")
        break

    cv2.imshow('Camera Feed', frame)

    # time.sleep(2)
    # cv2.imwrite("arkaimage.jpg", frame)
    # ml("arkaimage.jpg")
    # break
    if cv2.waitKey(1) & 0xFF == ord('x'):
        cv2.imwrite("arkaimage.jpg", frame)
        v = ml("arkaimage.jpg")
        if v==1:
            img_to_thermal_with_hot_circles("arkaimage.jpg")
            img = cv2.imread("thermal_output.jpg")
            cv2.imshow('thermal', img)
            cv2.waitKey(0)

# 4. Clear the window from your screen and memory
            cv2.destroyAllWindows()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()




