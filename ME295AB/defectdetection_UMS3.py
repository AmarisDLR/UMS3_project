### This program will take images at evenly spaced breakpoints. Using modulus (%) to
### specify the spacing of the breakpoints.
### This program is geared to the UMS3, which uses SSH communication protocols. 
### SSH communication is used to pass on g-code and get/set printer parameters.
###  UMS3 uses g-code flavor Griffin and does not place the origin at center.
### Using Ultimaker Cura, the origin is placed in the lower left corner of the printer
### bed (default origin).

### Activate UMS3 Wi-Fi connection and developer mode. 
### This code is for use on a Windows system, for connecting 
### with the USB camera. 

import os, re, cv2, time 
import paramiko, pyodbc
import numpy as np
from datetime import datetime
from pygrabber.dshow_graph import FilterGraph

import stl_visualization_ISO_UMS3 as stl_viz

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

def getCameraIndex(camName): ## For Windows systems
    graph = FilterGraph()
    if camName == "unsure":
        print(graph.get_input_devices())# list of camera device 
        device = input("\nEnter index of desired camera: ")
    else:
        device = graph.get_input_devices().index(camName)
    return device

def sendgcode(printer, gcode):
    printer.send("sendgcode "+gcode+"\n")
    

def find_init_temperature(gfile):
### M109 does not work in in the UMS3 SSH system
    extruder_temp = bed_temp = 0
    gcode = open(gfile,'r')
    find_temp = True
    while find_temp:
        line = gcode.readline()
        extemp = re.findall(";EXTRUDER_TRAIN.0.INITIAL_TEMPERATURE:", line)
        btemp = re.findall(";BUILD_PLATE.INITIAL_TEMPERATURE:", line)
        if extemp:
            extruder_temp = line.split(":",1)[1]
        if btemp:
            bed_temp = line.split(":",1)[1]
        if extruder_temp and bed_temp:
            find_temp = False
            break
    return float(extruder_temp), float(bed_temp)

def set_temperature(extruder_temp, bed_temp, printer, start):
    if start == 1:
        out = str(printer.recv(9999))
        print("\nTarget Extruder Temperature: " + str(extruder_temp) + " F\n")
        print("Target Bed Temperature: " + str(bed_temp) + " F\n")
        print("\n\nStart heating\n\n")	
        printer.send("select printer printer/bed\n")
        printer.send("set pre_tune_target_temperature "+ str(bed_temp)+" \n")
        time.sleep(3)
        printer.send("select printer printer/head/0/slot/0 \n")
        printer.send("set pre_tune_target_temperature "+ str(extruder_temp-2)+" \n")	
        time.sleep(2)
        check_temperature(extruder_temp-2, bed_temp,printer)
        print("\n\nFinished heating\n\n")
    else:	
        printer.send("select printer printer/head/0/slot/0 \n")
        printer.send("set pre_tune_target_temperature "+ str(extruder_temp)+" \n")	
        time.sleep(2)
        printer.send("select printer printer/bed \n")
        time.sleep(2)
        printer.send("set pre_tune_target_temperature "+ str(bed_temp)+" \n")


def check_temperature(extruder_temp, bed_temp,printer):
    t = 1
    count = 0
    printer.recv(9999)
    while t:
        count += 1
        printer.send("select printer printer/bed\n")
        printer.recv(9999)
        printer.send("get current_temperature \n")
        time.sleep(1.5)
        out = str(printer.recv(9999))
        current_bed_temp = re.findall(r'[\d.\d]+', out) ### current bed temperature
        if current_bed_temp:
            current_bed_temp = float(current_bed_temp[0])

        printer.send("select printer printer/head/0/slot/0 \n")
        printer.recv(9999)
        printer.send("get current_temperature \n")
        time.sleep(1.5)
        out = str(printer.recv(9999))
        current_ex_temp = re.findall(r'[\d.\d]+', out) ### current extruder temperature
        if current_ex_temp:
            current_ex_temp = float(current_ex_temp[0])
            if count%2 == 0:		
                print("*",end="",flush=True)
            print("current_bed_temp "+str(current_bed_temp)+"\t"+"bed_temp "+str(bed_temp))
            print("current_ex_temp "+str(current_ex_temp)+"\t"+"extruder_temp "+str(extruder_temp))
            if  current_bed_temp>=bed_temp-0.5 and current_ex_temp>=extruder_temp-0.5:		
                t = 0
                break

def zero_bed(offset, printer): 
    sendgcode(printer,"G0 F450 Y180 Z"+offset)
    sendgcode(printer,"G92 Z0")
    time.sleep(1)
    printer.recv(9999)

def set_time_elapsed(gfile, times_file):
    gfile_read = open(gfile, "r")	
    times = open(times_file,"w+")
    time_elapsed = time1 = Z = 0
    count = 1
    t = True
    while t:
        line = gfile_read.readline()
        if not line:
            global final_line
            final_line = count-1
            t = False
            times.write(str(time_elapsed)+","+str(final_line)+","+str(Z+25)+"\n")
            break
        mesh_line = re.findall('NONMESH', line)
        if mesh_line:
            j = True
            while j:
                line = gfile_read.readline()
                count += 1
                Z = get_Z(line)
                if Z > 0:
                    j = False
                    break
        timeline = re.findall(';TIME_ELAPSED:', line)
        if timeline:
            time2 = re.findall(r'[\d.\d]+', line)
            time2 = float(time2[0])
            time_elapsed = time2 - time1
            times.write(str(time_elapsed)+","+str(count)+","+str(Z)+"\n")
            time1 = time2
        count += 1

def get_time_elapsed(times_file):
    line = times_file.readline()
    if not line: ### if 'End of File' ###
        return -1,-1,-1
		
    items = line.split(",")
    elapsed_time = re.findall(r'[\d.\d]+', items[0])
    time_line = re.findall(r'[\d.\d]+', items[1])
    Z = re.findall(r'[\d.\d]+', items[2])
    elapsed_time = float(elapsed_time[0])
    time_line = int(time_line[0])
    Z = float(Z[0])
    return elapsed_time, time_line, Z

def get_Z(line):
    Z = 0
    items = re.findall('Z', line)
    if items:
        items = line.split("X",1)[1]
        items = items.split()	
        if len(items) > 2: 
            Z = re.findall(r'[\d.\d]+', items[2])
            Z = float(Z[0])
    return Z

def get_XY(line):
    X = Y = 0
    if re.findall("X",line):
        items = line.split("X",1)[1]
        items = items.split()
        if len(items) > 1:	
            X = re.findall(r'[\d.\d]+', items[0])
            Y = re.findall(r'[\d.\d]+', items[1])
            X = float(X[0])
            Y = float(Y[0])
    return X,Y

def check_position(X,Y,Z,printer,layerbreak,only_z):
    X = round(X,2)
    Y = round(Y,2)
    Z = round(Z,2)
    time.sleep(0.001)
    sendgcode(printer,"M114")
    out = printer.recv(9999)
    out = str(out)
    xyz_str = re.search(':(.+?)E',out)
    moveon = 0
    if xyz_str:
        positions = xyz_str.group(1)
        xyz_items = positions.split(':')
        if len(xyz_items) == 3:
            xc = re.findall(r'[\d.\d]+', xyz_items[0])
            yc = re.findall(r'[\d.\d]+', xyz_items[1])
            zc = re.findall(r'[\d.\d]+', xyz_items[2])
            xc = float(xc[0])
            yc = float(yc[0])
            zc = float(zc[0])
            x_compare = xc-0.01 <= X <= xc+0.01
            y_compare = yc-0.01 <= Y <= yc+0.01
            z_compare = zc-0.01 <= Z <= zc+0.01
            if z_compare and only_z:
                moveon = 1
            elif x_compare and y_compare and z_compare:
                moveon = 1
    else:
        moveon = 0
    return moveon

def video_capture(image_name, webcam):
    count = 1
    t = True
    while t:
        try:
            time.sleep(0.001)
            count += 1
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            if count == 325: ### Give time for camera to autofocus
                            ### & allow time for user to view camera feed
                capture_image(image_name, webcam, frame)
                t = False
                time.sleep(2)
                break
            elif count % 4 == 0:
                print("*",end="",flush=True)
        except(KeyboardInterrupt):
            cv2.destroyAllWindows()
            break

def capture_image(imfile, webcam, frame):
    cam_fps = webcam.get(cv2.CAP_PROP_FPS)
    print("\nCapture Image at %.2f FPS." %cam_fps)
    cv2.imwrite(filename=imfile, img=frame)
    print("Image saved!")
			
def print_initial_lines(printer): ### Purge line ###
    sendgcode(printer,"G92 E0") ## Reset extruder
    sendgcode(printer,"G0 F600 X0 Y190 Z0.24") ## Move to start position
    sendgcode(printer,"G0 F1500 X0 Y50 E15") ## Draw first line
    sendgcode(printer,"G0 F5000 X0.5 Y50") ## Move to side a little
    time.sleep(3)
    sendgcode(printer,"G0 F1500 X0.5 Y190 E25") ## Draw second line
    sendgcode(printer,"G91")	### Set Relative Positioning
    sendgcode(printer,"G0 Z1.25")### Move Z Axis up a little 
    sendgcode(printer,"G90")	### Set Absolute Positioning
    sendgcode(printer,"G92 E0\n") ## Reset extruder
    adjust_extruder(printer, -5,1)


def adjust_extruder(printer, amount, retract):
    if retract == 1:
        sendgcode(printer,"G91")	### Set Relative Positioning
        sendgcode(printer,"G0 F8000 E"+str(amount))	 ### Retract extruder quickly
        time.sleep(2)
        sendgcode(printer,"G90")	### Set Absolute Positioning
    else:
        sendgcode(printer,"G91")	### Set Relative Positioning
        sendgcode(printer,"G0 F750 E"+str(amount))  ### Extruder slowly
        time.sleep(2)
        sendgcode(printer,"G90")	### Set Absolute Positioning
        sendgcode(printer,"G0 F2000")

def adjust_extrusion_amount(line,alt_amount):
    Ennn = []
    Ennn_alt = 0
    if line[0] == 'G' and re.search('E',line):
        Ennn = line.split('E')
        if Ennn:
            Ennn = Ennn[-1]
            Ennn = re.findall(r'[-?\d.\d]+',Ennn)
            Ennn_alt = float(Ennn[0]) + alt_amount
            line = line.replace("E"+Ennn[0],"E"+str(Ennn_alt))
    return line

def adjust_feedrate_amount(line,alt_factor):
    Fnnn = []
    Fnnn_alt = 0
    if line[0] == 'G':
        Fnnn_split = line.split('F')
        if Fnnn_split:
            Fnnn_split = Fnnn_split[-1]
            Fnnn = re.findall(r'[\d.\d]+', Fnnn_split)
            Fnnn_alt = float(Fnnn[0]) * alt_factor
            line = line.replace("F"+Fnnn[0],"F"+str(Fnnn_alt))
    return line

def adjust_coolingfan_speed(printer,PWM):
    command = "sendgcode M106 S"+str(255*PWM)
    sendgcode(printer,command)
    
def check(list_, val): # Check if val is GREATER than items in list
    for x in list_: # Traverse the list
        # compare with values in list with val
        # If val is less 
        if val <= x: 
            return False
    return True

def class_predictions(directory, trained_model, shape, weights, feedrate_factor,\
    extrusion_factor, ex_temp_factor, PWM_factor):
    predictions = []
    number_files = 0
    for file_name in os.listdir(directory):
        if re.findall('split', file_name):
            image_name = os.path.join(directory, file_name)
            img = load_img(image_name, target_size=(shape, shape, 3))
            img_t = img_to_array(img)
            img_t = np.expand_dims(img_t,axis=0)
            number_of_black_pix = np.sum(img_t == 0)
            zeros_check = number_of_black_pix/img_t.size
            if zeros_check <= 0.85:
                number_files += 1
                img = preprocess_input(img_t)
                class_defects = trained_model.predict(img)
                check_values = np.greater(class_defects,0.5).astype(int)
                predictions.append(check_values)
            else:
                os.remove(image_name)
    
    if not predictions:
        return '---', 0, feedrate_factor, extrusion_factor, ex_temp_factor, PWM_factor, 0
    
    wpredictions = np.multiply(predictions, weights)
    wpredictions = np.sum(wpredictions, 0)[0]
    print("Weighted predictions: ", wpredictions)
    av_nodefects = wpredictions[0]/number_files
    av_overextrusion = (wpredictions[1]+wpredictions[2])/(1.5*number_files)
    av_stringing = (wpredictions[3]+wpredictions[4])/(1.5*number_files)
    av_underextrusion = (wpredictions[5]+wpredictions[6]+wpredictions[7])/(2.3*number_files)
    print("Averaged predictions: ",av_nodefects,av_overextrusion, av_stringing, av_underextrusion)
    
    if check([av_nodefects, av_stringing, av_underextrusion],av_overextrusion):
        defect = 'overextrusion'
        defect_ratio = av_overextrusion
        if av_overextrusion > 0.6:
            feedrate_factor += 0.05
            extrusion_factor -= 0.05
            ex_temp_factor -= 5
            PWM_factor += 0.05
        elif av_overextrusion <= 0.6 and av_overextrusion > 0.35:
            feedrate_factor += 0.025
            extrusion_factor -= 0.025
            ex_temp_factor -= 2.5
            PWM_factor += 0.025
        elif av_overextrusion <= 0.35 and av_overextrusion > 0.10:
            feedrate_factor += 0.01
            extrusion_factor -= 0.01
            ex_temp_factor -= 1
            PWM_factor += 0.01
        else:
            pass
    elif check([av_nodefects, av_overextrusion, av_underextrusion], av_stringing):
        defect = 'stringing'
        defect_ratio = av_stringing
        if av_stringing > 0.6:
            extrusion_factor -= 0.05
            ex_temp_factor -= 5
            PWM_factor += 0.05
        elif av_stringing <= 0.6 and av_stringing > 0.35:
            extrusion_factor -= 0.025
            ex_temp_factor -= 2.5
            PWM_factor += 0.025
        elif av_stringing <= 0.35 and av_stringing > 0.10:
            extrusion_factor -= 0.01
            ex_temp_factor -= 1
            PWM_factor += 0.01
        else:
            pass
    elif check([av_nodefects, av_overextrusion, av_stringing], av_underextrusion):
        defect = 'underextrusion'
        defect_ratio = av_underextrusion
        if av_underextrusion > 0.6:
            feedrate_factor -= 0.05
            extrusion_factor += 0.05
            ex_temp_factor += 5
            PWM_factor -= 0.05
        elif av_underextrusion <= 0.6 and av_underextrusion > 0.35:
            feedrate_factor -= 0.025
            extrusion_factor += 0.025
            ex_temp_factor += 2.5
            PWM_factor -= 0.025
        elif av_underextrusion <= 0.35 and av_underextrusion > 0.10:
            feedrate_factor -= 0.01
            extrusion_factor += 0.01
            ex_temp_factor += .1
            PWM_factor -= 0.01
        else:
            pass
    else: # no defects
        defect = 'No defects'
        defect_ratio = av_nodefects

    if PWM_factor > 100:
        PWM_factor = 100
    elif PWM_factor < 0:
        PWM_factor = 0    
            
    return defect, number_files, feedrate_factor, extrusion_factor, ex_temp_factor, PWM_factor, defect_ratio

################  Log in to SSH  ##################
ip_address = "192.168.0.226"
username = "ultimaker"
password = "ultimaker"
ssh = paramiko.SSHClient()
### Load SSH host keys.
ssh.load_system_host_keys()
### Add SSH host key when missing.
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip_address,username=username,password=password,look_for_keys=False,port=22)
remote_connection = ssh.invoke_shell()
### send out command to begin shell program
remote_connection.send("\n")
out = remote_connection.recv(9999)
print(out)
sendgcode(remote_connection,"G28")
sendgcode(remote_connection,"G0 F2500")
out = remote_connection.recv(9999)
print(out)


################  Start Camera - Windows System ##################
camIdx = getCameraIndex('IPEVO V4K') ## Enter camera name or 'unsure' to get
                                     ## a list of available cameras and select
                                     ## desired index number
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(camIdx,
                          cv2.CAP_DSHOW)
img_size_x = 2400 # Ideal pixel size x
img_size_y = 2400 # Ideal pixel size y
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, img_size_x)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size_y)
widthx = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)) # Actual pixel capture size x
heighty = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Actual pixel capture size y
imxy = '_'+str(widthx)+'W_'+str(heighty)+'H'

################  Connect to SQL database  ##################
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=localhost\SQLEXPRESS;'
                      'Database=master;'
                      'Trusted_Connection=yes;')
defectsql = conn.cursor()


################  Get Times Btwn Layers  ##################
print("\n\n\n")
gcode_PATH = "D:/gcode_ender3/CE3_ISO_527_2_2012_FLAT_100INFILL.gcode"
gfile_name = "UMS3_ISO_527_2_2012_FLAT_100INFILL"
gfile = "gcodeUM/"+gfile_name+".gcode" #input("Gcode file: <file.gcode> \n")

times_file = "times.txt"
set_time_elapsed(gfile, times_file)
n_layers = 1 ## Capture images at every n layers

################  Get & Send Temperature  ##################
extruder_temp, bed_temp = find_init_temperature(gfile)
set_temperature(extruder_temp, bed_temp, remote_connection,0)


################  Load ML model  ##################
print("Loading model")
model_name = 'ML_training/models/ethereal-glitter-293'
tic = time.time()
vgg16_model = load_model(model_name)
toc = time.time()
load_time = toc-tic
print(f'Model load time: {load_time}')
model_shape=vgg16_model.layers[0].input.get_shape().as_list()
shapehw = model_shape[1]
class_weights = [1, # No defects
           1.0, 0.5, # overextrusion_high, overextrusion_high
           1.0, 0.5, # stringing_high, stringing_low,
           0.75, 1.0, 0.5] # underextrusion, underextrusion_high,

image_directory = "defectdatabase/"
if not os.path.exists(image_directory):
    os.mkdir(image_directory)

# Initial model prediction, ~ 20-30s, while printer heating
test_im = "test.jpg"
tic = time.time()
video_capture(test_im, webcam)
img = load_img(test_im, target_size=(shapehw, shapehw, 3))
img_t = img_to_array(img)
img_t = np.expand_dims(img_t,axis=0)
d = vgg16_model.predict(img_t)
toc = time.time()
os.remove(test_im)
print(d)
print("Initial train time: ", toc-tic)
set_temperature(extruder_temp, bed_temp, remote_connection,1) # Complete heating
################  Start Printing  ##################
print("\n\nStart printing from file.\n\n")

z_offset = str(4.60) ## z_offset = input("\nEnter height to zero bed: ")

gfile_print = open(gfile, "r")
times = open(times_file,"r")
linecount = 1

X_line = Y_line = Z_line = layercount = 0

elapsed_time, layerbreak, Z = get_time_elapsed(times)
timepause = elapsed_time
time.sleep(2)

goal_X = 10

fr_factor = 1.00 ### Feedrate adjustment factor, percent
alt_amount = 1.00 ### Extrusion adjustment factor, percent
alt_temp = 0.00 ### Temperature adjustment factor, deg F
fanPWM = 1.00 ### Fan Speed % (8-bit)

################  Print Loop  ##################
while True:
    try:
        line = gfile_print.readline()
        if not line or linecount==final_line:
			
            print("\nFinished printing. \n")  ### End of g-code file ### 
            set_temperature(0, 0, remote_connection,0)
            sendgcode(remote_connection,"G0 F5500 X0 Y150")
            sendgcode(remote_connection,"G92 E0")
            time.sleep(1)
            sendgcode(remote_connection,"G28 Z")			
            sendgcode(remote_connection,"G28 Z")
            webcam.release()
            cv2.destroyAllWindows()
            command = input("\nConfirm piece is removed from print bed by hitting 'enter'. \n")
            remote_connection.close()
            os.remove(times_file)
            break

        else: #### Print g-code ####
            time.sleep(0.01)
            print("Layer: "+str(layercount)+", Line "+str(linecount)+" of "+str(layerbreak))

            ### Send gcode to UMS3 printer
            sendgcode(remote_connection,line)

            if linecount == 33: #### Zero Bed Offset ####
                zero_bed(z_offset,remote_connection)
                print_initial_lines(remote_connection)
                for t in range(int(15)):
                    print("*",end="",flush=True)			
                    time.sleep(1)
                print("\n\n")
                starttime = time.time()

            if linecount == layerbreak-1: ### Get final X and Y positions of layer	
                X_line,Y_line = get_XY(line)
                if X_line > 0 and Y_line > 0:
                    X = X_line
                    Y = Y_line

            if linecount == layerbreak:
                print("\n\nLine: "+str(linecount)+" , Break Layer: "+str(layercount)+"\n\n")

                if layercount % n_layers == 0:
                    ### Retract extruder
                    set_temperature(extruder_temp-12, bed_temp, remote_connection,2)
                    adjust_extruder(remote_connection, -9,1) ## amount to retract in mm
                    
                    ### Position for camera capture
                    goal_Z = Z+0.1+(linecount%2)/10
                    goal_Y = 128
                    out = remote_connection.recv(9999)
                    sendgcode(remote_connection,"G0 Z"+str(goal_Z))
                    sendgcode(remote_connection,"G0 F7000 X0 Y"+str(goal_Y))
								
                    ### Sleep for estimated time for layer, check position
                    endtime = time.time()
                    timediff = (endtime-starttime)
                    time.sleep(int(abs(timepause-timediff)))
                    i = 0
                    for t in range(10000000):
                        if t % 2 == 1:
                            print("*",end="",flush=True)
                        i = check_position(0,goal_Y,goal_Z,remote_connection,layerbreak,0)
                        if i == 1:
                            print("|",end="",flush=True)
                            break
                        time.sleep(.85)
                    sendgcode(remote_connection,"G0 F5000 X"+str(goal_X)+\
                          " Y"+str(goal_Y)+" Z"+str(goal_Z))
                        
                    ### Capture overhead image
                    ts = str(time.strftime("%Y%m%d%H%M"))
                    predict_dir = os.path.join('ML_training\predict_imgs', str(ts)+'_Layer'+str(layercount)+'_'+gfile_name)
                    os.mkdir(predict_dir)
                    imfile = os.path.join(predict_dir,gfile_name+"_Layer"+str(layercount)+str(imxy)+".jpg")
                    i = 0		
                    for t in range(10000000):
                        if t % 2 == 1:
                            print("-",end="",flush=True)
                        i = check_position(goal_X,goal_Y,goal_Z,remote_connection,layerbreak,0)
                        if i == 1:
                            video_capture(imfile,webcam)
                            break
                        time.sleep(2)
						
        				  ### Position to resume printing
        				  #adjust_extruder(remote_connection, 12,0) ## amount to extrude in mmls
                    set_temperature(extruder_temp, bed_temp, remote_connection,2)
                    gpositionXY = "X"+str(X)+" Y"+str(Y)
                    gpositionZ = "Z"+str(Z)
                    ### Return to last XYZ position
                    sendgcode(remote_connection,"G0 "+gpositionZ)	
                    sendgcode(remote_connection,"G0 "+gpositionXY)	
                    time.sleep(2)
                    
                    ### Evaluate images
                    stl_viz.gcode_overlay(widthx, heighty, gcode_PATH, predict_dir,imfile, 5)
                    tic = time.time()
                    defect, number_files, fr_factor, alt_amount, alt_temp, fanPWM , def_amt = class_predictions(predict_dir,\
                            vgg16_model, shapehw, class_weights, fr_factor, alt_amount, alt_temp, fanPWM)
                    toc = time.time()
                    predict_time = toc-tic
                    print(f'Model prediction time for {number_files} images: {predict_time}')
                    
                    ###### Change printer settings ######
                    line = adjust_feedrate_amount(line, fr_factor)
                    line = adjust_extrusion_amount(line, alt_amount)
                    set_temperature(extruder_temp+alt_temp, bed_temp, remote_connection,2)
                    adjust_coolingfan_speed(remote_connection,fanPWM)
                    ###### Change printer settings ######
                    
                    ### Store data into SQL database
                    timestamp = datetime.utcnow()
                    defectsql.execute('''
                                      INSERT INTO defectdetection_UMS3 (linecount, layercount, defect, alt_temp, temp, fr_factor, alt_amount, fanPWM, capture, defect_ratio, predict_time, num_files)
                                      VALUES
                                      (?,?,?,?,?,?,?,?,?,?,?,?)
                                      ''',(linecount, layercount, defect, alt_temp, extruder_temp, fr_factor, alt_amount, fanPWM, timestamp, def_amt,predict_time,number_files))
                                      
                    conn.commit()
                    
                    
                    timepause = 0
                    starttime = time.time()
                    
                    

                ### Update layerbreak and time_break
                elapsed_time, layerbreak, Z = get_time_elapsed(times)
                timepause += elapsed_time	
                layercount += 1
				
            linecount += 1


    ### Escape/Close program manually using ctrl+c ####
    except(KeyboardInterrupt):
        print("\nExit program.\n")
        sendgcode(remote_connection,"G0 Z"+str(goal_Z+5))
        i = 0		
        for t in range(10000000):
            i = check_position(0,0,goal_Z+5,remote_connection,layerbreak,1)
            if i == 1:
                break
        sendgcode(remote_connection,"G28 Z")
        set_temperature(0, 0, remote_connection,0)
        time.sleep(1)
        sendgcode(remote_connection,"G0 F5500 X0 Y150")
        time.sleep(1)		
        sendgcode(remote_connection,"G28 Z")
        webcam.release()
        cv2.destroyAllWindows()
        remote_connection.close()
        break


