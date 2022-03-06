import cv2
import time
import calendar
from pygrabber.dshow_graph import FilterGraph


def getCameraIndex(camName):
    graph = FilterGraph()
    if camName == "unsure":
        print(graph.get_input_devices())# list of camera device 
        device = input("\nEnter index of desired camera: ")
    else:
        device = graph.get_input_devices().index(camName)
    return device

def capture_img(path, camera, frame):
	cam_fps = camera.get(cv2.CAP_PROP_FPS)
	print('Capture Image at %.2f FPS.' %cam_fps)
	ts = calendar.timegm(time.gmtime())
	imfile = path+str(ts)+'1_5fps_1600_1200.jpg'

	print(imfile)	
	cv2.imwrite(filename=imfile, img=frame)

	print("Image saved!")

print("\n")
print("Program to save images as .jpg\n")
print("\n")

path = "E:/AM_Papers/Ender3/directory/" 

camIdx = getCameraIndex('IPEVO V4K')
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(camIdx,cv2.CAP_DSHOW)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)		#1280    #1024   #640    #800   #3264
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)		#720	 #768    #480    #600   #2448
#webcam.set(cv2.CAP_PROP_FPS, 15)			#10	 #15     #30     #25    #1.5

webcam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

print("\n")
print("Click terminal window: use CTRL+c to close camera and quit program.\n")
print("Click on camera stream window: use 's' to save image.")
print("\n")

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
			capture_img(path,webcam,frame)
			print("\n")
			print("Click terminal window: use CTRL+c to close camera and quit program.\n")
			print("Click on camera stream window: use 's' to save image.")
			print("\n")

	except(KeyboardInterrupt):
		print("Turning off camera.")
		webcam.release()
		print("Camera off.")
		print("Program ended.")
		cv2.destroyAllWindows()
		break