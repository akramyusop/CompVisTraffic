import cv2
from ultralytics import YOLO

# Load YOLO11n model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press 'q' to quit.")

while True:
    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # -----------------------------
    # ROI coordinates
    # -----------------------------
    roi_x1 = int(width * 0.40)
    roi_y1 = int(height * 0.45)

    roi_x2 = int(width * 0.95)
    roi_y2 = int(height * 0.95)

    # Draw ROI rectangle
    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (0, 255, 255),
        2
    )

    # Detect vehicles
    results = model(
        frame,
        classes=[2, 3, 5],
        conf=0.75,
        verbose=False
    )

    result = results[0]

    # Vehicle counter inside ROI
    vehicle_count = 0

    # Check each detected vehicle
    for box in result.boxes:

        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Convert coordinates to integer
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Calculate center of vehicle
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Check if vehicle center is inside ROI
        if (
            roi_x1 <= center_x <= roi_x2
            and roi_y1 <= center_y <= roi_y2
        ):
            vehicle_count += 1

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )

    # Draw YOLO detection boxes
    annotated_frame = result.plot()

    # Draw ROI again so it appears on top
    cv2.rectangle(
        annotated_frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (0, 255, 255),
        2
    )

    # Display vehicle count
    cv2.putText(
        annotated_frame,
        f"Vehicles in ROI: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display webcam
    cv2.imshow(
        "Real-Time Vehicle Detection",
        annotated_frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()