# Read a Video Stream from Camera(Frame by Frame)
import cv2

cap = cv2.VideoCapture(0)

while True:
	ret,frame = cap.read()

	if ret == False:
		continue

	cv2.imshow("Video Frame",frame)

	#Wait for user input - q, then you will stop the loop
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyALlWindows()








