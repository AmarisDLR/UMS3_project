import cv2
import time
import calendar
from stl_visualization import *

def capture_img(path, camera, frame):
	cam_fps = camera.get(cv2.CAP_PROP_FPS)
	print('Capture Image at %.2f FPS.' %cam_fps)
	ts = calendar.timegm(time.gmtime())
	imfile = "database/compare/"+str(ts)+'1_5fps_3264_2448.jpg'
	imfile = str(ts)+'1_5fps_1600_1200.jpg'
	print(imfile)	
	cv2.imwrite(filename=imfile, img=frame)
	print("Image saved!")

print("\n\n")
print("Program to save images as .jpg\n")
print("\n\n")

path = "Desktop/UMS3_project/ME295AB/defects"

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)		#1280    #1024   #640    #800   #3264
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)		#720	 #768    #480    #600   #2448
#webcam.set(cv2.CAP_PROP_FPS, 15)			#10	 #15     #30     #25    #1.5

print("\n\n")
print("Click terminal window: use CTRL+c to close camera and quit program.\n")
print("Click on camera stream window: use 's' to save image.")
print("\n\n")

count = 1
while True:
	try:
		key = cv2.waitKey(1)
		count += 1
		#print("\n\n"+str(count)+"\n\n")
		check, frame = webcam.read()
		#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		cv2.imshow("Capturing", frame)
		
		if key == ord('s'): 
			capture_img(path, webcam, frame)
			stl_file = 'gcodeUM/UMS3_random39_block.stl'
			LAYER_NUMBER = 57
			img_size_x = 1600
			img_size_y = 1200
			zoom = 0.06875
			im1='database/3DPrinterWorkspace_UMS3_random39_7infill_lines_1_202111141751.jpg'
			im2='database/topview_UMS3_random39_7infill_lines_1_202111141751.jpg'
			stl_3Dworkspace(img_size_x, img_size_y, stl_file, LAYER_NUMBER,im1,im2)

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

