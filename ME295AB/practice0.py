#!/usr/bin/python

from tkinter import *
import jetson.inference
import jetson.utils

def capture_image():
    print("cheese")

# Set-up the window
window = Tk()
window.title("Temperature Converter")
window.rowconfigure(0,minsize=500,weight=1)
window.columnconfigure(1,minsize=450,weight=1)
window.resizable(width=False, height=False)

# Create the Fahrenheit entry frame with an Entry
# widget and label in it
frm_entry = Frame(master=window)
ent_camera = Entry(master=frm_entry, width=10)
lbl_camera = Label(master=frm_entry)

# Layout the temperature Entry and Label in frm_entry
# using the .grid() geometry manager
ent_camera.grid(row=0, column=0, sticky="e")
lbl_camera.grid(row=0, column=1, sticky="w")

# Create the conversion Button and result display Label
btn_convert = tk.Button(
    master=window,
    text="Capture",
    command=capture_image
)


# Set-up the layout using the .grid() geometry manager
frm_entry.grid(row=0, column=0, padx=10)
btn_convert.grid(row=0, column=1, pady=10)

# Run the application
window.mainloop()


#net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#camera = jetson.utils.videoSource("/dev/video0")      # '/dev/video0' for V4L2
#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

#while display.IsStreaming():
	#img = camera.Capture()
	#detections = net.Detect(img)
	#display.Render(img)
	#display.SetStatus("Object Detection | Network {:.0f} FmakePS".format(net.GetNetworkFPS()))
