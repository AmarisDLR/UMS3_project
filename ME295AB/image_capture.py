import cv2
import time
import calendar
from pygrabber.dshow_graph import FilterGraph

def getCameraIndex(camName): ## For Windows systems
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
	imfile = "database/compare/"+str(ts)+'1_5fps_3264_2448.jpg'
	imfile = str(ts)+'1_5fps_1600_1200.jpg'
	print(imfile)	
	cv2.imwrite(filename=imfile, img=frame)
	print("Image saved!")

print("\n\n")
print("Program to save images as .jpg\n")
print("\n\n")

path = "C:/Users/amari/UMS3_project/ME295AB/"


################  Start Camera - Windows System ##################
camIdx = getCameraIndex('IPEVO V4K') ## Enter camera name or 'unsure' to get
                                     ## a list of available cameras and select
                                     ## desired index number
key = cv2.waitKey(camIdx)
webcam = cv2.VideoCapture(camIdx,cv2.CAP_DSHOW)
img_size_x = 2400
img_size_y = 2400
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, img_size_x)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size_y)

#webcam.set(cv2.CAP_PROP_FPS, 15)			#10	 #15     #30     #25    #1.5

print("\n\n")
print("Click terminal window: use CTRL+c to close camera and quit program.\n")
print("Click on camera stream window: use 's' to save image.")
print("\n\n")
cam_fps = webcam.get(cv2.CAP_PROP_FPS)
print('Capture Image at %f FPS.' %cam_fps)

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

