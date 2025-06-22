import os
import re
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Constants
width, height = 1920, 1080
folderPath = "Capstone_Review final/Capstone_Review final"
gestureThreshold = 300
hs, ws = int(120 * 1), int(180 * 1)
buttonPressed = False
buttoncounter = 0
buttondelay = 15
annotations = [[]]
annotationNo = 0
annotationstart = False

# Rename PNG files sequentially
def rename_png_files(folder_path):
    png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    png_files.sort()
    for index, old_name in enumerate(png_files, start=1):
        new_name = f"{index}.png"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")

# Run rename logic
if not os.path.exists(folderPath):
    print(f"❌ Folder '{folderPath}' not found. Check the path.")
    exit()

rename_png_files(folderPath)

# Sort files by number
pattern = r"(\d+)"
def safe_sort_key(x):
    match = re.findall(pattern, x)
    return int(match[0]) if match else float('inf')

# Load image files
pathImages = sorted(
    [file for file in os.listdir(folderPath) if file.endswith('.png')],
    key=safe_sort_key
)

if not pathImages:
    print(f"❌ No PNG files found in the folder: {folderPath}")
    exit()

imgno = 0

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Initialize hand detector
det = HandDetector(detectionCon=0.8, maxHands=1)

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = det.findHands(img)

    # Load current slide
    pathFullimage = os.path.join(folderPath, pathImages[imgno])
    imgcurr = cv2.imread(pathFullimage)

    # Draw gesture threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPressed:
        hand = hands[0]
        fingers = det.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [100, height], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:
            annotationstart = False

            # Gesture 1 - Left
            if fingers == [1, 0, 0, 0, 0]:
                annotationstart = False
                print("⬅️ Slide Left")
                if imgno > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNo = 0
                    imgno -= 1

            # Gesture 2 - Right
            if fingers == [0, 0, 0, 0, 0]:
                annotationstart = False
                print("➡️ Slide Right")
                if imgno < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNo = 0
                    imgno += 1

            # Gesture 6 - Delete all slides and exit
            if fingers == [1, 1, 0, 0, 1]:
                for file in os.listdir(folderPath):
                    file_path = os.path.join(folderPath, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print("🗑️ All files deleted.")
                break

        # Gesture 3 - Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgcurr, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationstart = False

        # Gesture 4 - Draw
        if fingers == [0, 1, 0, 0, 0]:
            if not annotationstart:
                annotationstart = True
                annotationNo += 1
                annotations.append([])
            cv2.circle(imgcurr, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNo].append(indexFinger)
        else:
            annotationstart = False

        # Gesture 5 - Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotationNo >= 0 and annotations:
                annotations.pop(-1)
                annotationNo -= 1
                buttonPressed = True
    else:
        annotationstart = False

    # Reset button press delay
    if buttonPressed:
        buttoncounter += 1
        if buttoncounter > buttondelay:
            buttoncounter = 0
            buttonPressed = False

    # Draw annotations
    for i in range(len(annotations)):
        for j in range(1, len(annotations[i])):
            cv2.line(imgcurr, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 10)

    # Webcam overlay
    imgsmall = cv2.resize(img, (ws, hs))
    imgsmall_resized = cv2.resize(imgsmall, (ws, hs))
    canvas = np.zeros((imgcurr.shape[0], ws, 3), dtype=np.uint8)
    canvas[:imgsmall_resized.shape[0], :imgsmall_resized.shape[1]] = imgsmall_resized
    img_combined = cv2.hconcat([imgcurr, canvas])

    # Display result
    cv2.imshow("slides", img_combined)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
