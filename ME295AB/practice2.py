import cv2

def rescale_frame(frame, percent=75):
	width = int(frame.shape[1] * percent/ 100)
	height = int(frame.shape[0] * percent/ 100)
	dim = (width, height)
	return cv2.resize(frame, dim)

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
	try:
		check, frame = webcam.read()
		framers = rescale_frame(frame,20)#rescale_frame(frame, percent=75)
		#print(check) #prints true as long as the webcam is running
		#print(frame) #prints matrix values of each framecd 
		cv2.cvtColor(framers, cv2.COLOR_BGR2RGB)
		cv2.imshow("Capturing", framers)
		key = cv2.waitKey(1)
        
	except(KeyboardInterrupt):
		print("Turning off camera.")
		webcam.release()
		print("Camera off.")
		print("Program ended.")
		cv2.destroyAllWindows()
		break

