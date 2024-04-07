import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def motionDetection():
    cap = cv.VideoCapture("in.avi")
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Initialize the pause flag to False
    paused = False

    # Set the initial zoom level to 1
    zoom_level = 1

    # Initialize video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('out.mp4', fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))
    
    while cap.isOpened():
        if not paused:
            diff = cv.absdiff(frame1, frame2)
            diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
            _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
            dilated = cv.dilate(thresh, None, iterations=3)
            contours, _ = cv.findContours(
                dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x, y, w, h) = cv.boundingRect(contour)
                if cv.contourArea(contour) < 900:
                    continue
                cv.rectangle(frame1, (x,y), (x+w, y+h), (0, 255, 0), 2)
                cv.putText(frame1, "Pedestrian {}".format('Detection'), (10,20), cv.FONT_HERSHEY_SIMPLEX,
                           1, (255, 0, 0), 3)
                
            # Zoom the frame based on the current zoom level
            zoomed = cv.resize(frame1, (0,0), fx=zoom_level, fy=zoom_level, interpolation=cv.INTER_LINEAR)

            #Write the frame to the video file
            out.write(zoomed)

            # Show the zoomed frame
            cv.imshow("Video", zoomed)
            frame1 = frame2
            ret, frame2 = cap.read()

        # Wait for the user to press a key
        key = cv.waitKey(50)
        if key == 27:
            break
        elif key == ord(' '):
            # Toggle the pause flag
            paused = not paused
        elif key == ord('+'):
            # Increase the zoom level by 0.1
            zoom_level += 0.1
        elif key == ord('-'):
            # Decrease the zoom level by 0.1, but keep it above 0.1
            zoom_level = max(0.1, zoom_level - 0.1)

    # Release the video writer and destroy all windows
    out.release()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    motionDetection()
