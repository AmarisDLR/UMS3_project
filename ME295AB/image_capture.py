import cv2
import time
import calendar

def capture_img(camera):
	cam_fps = camera.get(cv2.CAP_PROP_FPS)
	print('Capture Image at %.2f FPS.' %cam_fps)
	ts = calendar.timegm(time.gmtime())
	imfile = str(ts)+'img.jpg'
	print(imfile)	
	cv2.imwrite(filename=imfile, img=frame)
	print("Image saved!")

print("\n\n")
print("Program to save images as .jpg\n")
print("\n\n")

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)#640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)#480)
webcam.set(cv2.CAP_PROP_FPS, 25)#30)

print("\n\n")
print("Click terminal window: use CTRL+c to close camera and quit program.\n")
print("Click on camera stream window: use 's' to save image.")
print("\n\n")

while True:
	try:
		check, frame = webcam.read()
		#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		cv2.imshow("Capturing", frame)
		key = cv2.waitKey(1)
		if key == ord('s'): 
			capture_img(webcam)
			print("\n\n")
			print("Click terminal window: use CTRL+C to close camera and quit program.\n")
			print("Click on camera stream window: use 's' to save image.")
			print("\n\n")


	except(KeyboardInterrupt):
		print("Turning off camera.")
		webcam.release()
		print("Camera off.")
		print("Program ended.")
		cv2.destroyAllWindows()
		break

