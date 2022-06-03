import os, time
import re, cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph
from datetime import datetime

import pyodbc

import serial, serial.tools.list_ports
import stl_visualization_ISO as stl_viz

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

def getCameraIndex(camName):
    graph = FilterGraph()
    if camName == "unsure":
        print(graph.get_input_devices())# list of camera device 
        device = input("\nEnter index of desired camera: ")
    else:
        device = graph.get_input_devices().index(camName)
        print(f'Connected to camera: {camName}\n')
    return device

def command(ser, line):
    matchM105 = re.findall("M105",line)
    if line != "" and line[0] != ";" and not matchM105:
        print(line)
        ser.write(str.encode(line+"\r\n")) 
        while True:
            out_line = ser.readline()
            print(out_line)
            if out_line == b'ok\n':
                break
    else:
        print(line)


def set_time_elapsed(gfile, times_file):
    gfile_read = open(gfile,'r')
    times = open(times_file,'w+')
    time_elapsed = layer_count =  time1 = Z = 0
    line_count = 1
    
    t = True
    while t:
        line = gfile_read.readline()
        if not line:
                global final_line
                final_line = line_count - 1
                t = False
                times.write(str(time_elapsed)+','+str(final_line)+','+str(Z+15)+'\n')
                print("\nTotal number of layers: "+str(layer_count))
                time.sleep(2)
                break
        
        mesh_line = re.findall('NONMESH',line)

        if mesh_line:
            j = True
            while j:
                line = gfile_read.readline()
                line_count += 1
                Z = get_Z(line)
                if Z > 0:
                    j = False
                    break

        timeline = re.findall(';TIME_ELAPSED:',line)

        if timeline:
            time2 = re.findall(r'[\d.\d]+',line)
            time2 = float(time2[0])
            time_elapsed = time2 - time1
            times.write(str(time_elapsed)+','+str(line_count)+','+str(Z)+'\n')
            layer_count += 1
            time1 = time2
        line_count += 1

def get_Z(line):
    Z = 0
    items = re.findall('Z',line)
    if items:
        items = line.split('X',1)[1]
        items = items.split()
        if len(items) > 2:
            Z = re.findall(r'[\d.\d]+',items[2])
            Z = float(Z[0])
    return Z

def get_XY(line):
    X = Y = 0
    if re.findall("X", line):
        items = line.split("X",1)[1]
        items = items.split()
        if len(items) > 1:
            X = re.findall(r'[\d.\d]+',items[0])
            Y = re.findall(r'[\d.\d]+',items[1])
            X = float(X[0])
            Y = float(Y[0])
    return X,Y

def get_time_elapsed(times_file):
    line = times_file.readline()
    if not line: ### if 'EOF'
        return -1, -1, -1
    
    items = line.split(',')
    elapsed_time = re.findall(r'[\d.\d]+', items[0])
    time_line = re.findall(r'[\d.\d]+', items[1])
    Z = re.findall(r'[\d.\d]+',items[2])
    elapsed_time = float(elapsed_time[0])
    time_line = int(time_line[0])
    Z = float(Z[0])
    return elapsed_time, time_line, Z

def find_init_temperature(gfile):
    extruder_temp = bed_temp = 0
    gcode = open(gfile,'r')
    find_temp = True
    while find_temp:
        line = gcode.readline()
        extemp = re.findall("M104 S",line)
        btemp = re.findall("M140 S",line)
        if extemp:
            extruder_temp = line.split("S",1)[1]
        if btemp:
            bed_temp = line.split("S",1)[1]
        if extruder_temp and bed_temp:
            find_temp = False
            break
    return float(extruder_temp), float(bed_temp)

def set_temperature(ser, extruder_temp, bed_temp):
    command(ser,"M104 S"+str(extruder_temp))
    command(ser,"M140 S"+str(bed_temp))

def adjust_extruder(ser, amount, retract):
    if retract == 1:
        command(ser,"M83") ### Set Relative Extruder Positioning
        command(ser,"G0 F4000 E"+str(amount))
        command(ser, "M82") ### Return to Absolute Extruder Positioning
    else: # Return slowly
        command(ser,"M83") ### Set Relative Extruder Positioning
        command(ser,"G0 F750 E"+str(amount))
        command(ser, "M82") ### Return to Absolute Extruder Positioning
        command(ser,"G0 F2000")

def adjust_feedrate_amount(line, alt_amount):
    if line[0] == 'G':
        Fnnn_split = line.split('F')
        if Fnnn_split:
            Fnnn_split = Fnnn_split[-1]
            Fnnn = re.findall(r'[\d.\d]+',Fnnn_split)
            Fnnn_alt = float(Fnnn[0]) * alt_amount
            line = line.replace("F"+Fnnn[0],"F"+str(Fnnn_alt))
            print(f'Ennn:{Fnnn[0]}\tEnnn_alt:{Fnnn_alt}')
    return line

def adjust_extrusion_amount(line, alt_amount):
    if re.findall("G0", line) or re.findall("G1", line):
        g0g1 = 1
    else:
        g0g1 = 0
    if g0g1 and re.search('E',line):
        Ennn = line.split('E')
        if Ennn:
            Ennn = Ennn[-1]
            Ennn = re.findall(r'[-?\d.\d]+',Ennn)
            Ennn_alt = float(Ennn[0]) * alt_amount
            line = line.replace("E"+Ennn[0],"E"+str(Ennn_alt))
            print(f'Ennn:{Ennn[0]}\tEnnn_alt:{Ennn_alt}')
    return line

def adjust_coolingfan_speed(ser,PWM):
    command(ser,"M106 S"+str(255*PWM))

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

def check_position(ser, goalx, goaly, goalz):
    count = 0
    X = round(goalx,2)
    Y = round(goaly,2)
    Z = round(goalz,2)
    moving = True
    while moving:
        #print("*",end="",flush=True)
        ser.write(str.encode("M114\r\n")) 
        line = ser.readline()
        time.sleep(0.001)
        if line != b'ok\n' and re.findall("Count",str(line)):
            line = str(line)
            print(line)
            split_line = line.split()
            xc = float(re.findall(r'[\d.\d]+', split_line[0])[0])
            yc = float(re.findall(r'[\d.\d]+', split_line[1])[0])
            zc = float(re.findall(r'[\d.\d]+', split_line[2])[0])
            x_compare = xc-0.02 <= X <= xc+0.2
            y_compare = yc-0.02 <= Y <= yc+0.2
            z_compare = zc-0.02 <= Z <= zc+0.2
            #print("Goal: "+str(goalx)+" "+str(goaly)+" "+str(goalz))
            #print("Compare: "+str(xc)+" "+str(yc)+" "+str(zc))
            # Check if current XYZ positions are in goal range
            if x_compare and y_compare and z_compare:
                moving = False
                break
        elif count > 10000: # Escape if check is running too long
            moving = False
        elif count%200 == 0: # Output goal position
            command(ser,"G0 X"+str(goalx)+" Y"+str(goaly)+" Z"+str(goalz))

        count +=1
        
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
            #np.count_nonzero(check_values)
            if zeros_check <= 0.85:
                number_files += 1
                img = preprocess_input(img_t)
                class_defects = trained_model.predict(img)
                print(class_defects)
                check_values = np.greater(class_defects,0.75).astype(int)
                print(check_values)
                predictions.append(check_values)
                
            else:
                os.remove(image_name)
    
    wpredictions = np.multiply(predictions, weights)
    wpredictions = np.sum(wpredictions, 0)[0]
    av_nodefects = wpredictions[0]/number_files
    av_overextrusion = (wpredictions[1]+wpredictions[2])/(1.5*number_files)
    av_stringing = (wpredictions[3]+wpredictions[4])/(1.5*number_files)
    av_underextrusion = (wpredictions[5]+wpredictions[6]+wpredictions[7])/(2.3*number_files)
    
    if check([av_nodefects, av_stringing, av_underextrusion],av_overextrusion):
        defect = 'overextrusion'
        print(av_overextrusion)
        def_ratio = av_overextrusion
        if av_overextrusion > 0.6:
            feedrate_factor += 0.05
            extrusion_factor -= 0.05
            ex_temp_factor -= 5
            PWM_factor += 0.05
        elif av_overextrusion <= 0.6 and av_overextrusion > 0.3:
            feedrate_factor += 0.025
            extrusion_factor -= 0.025
            ex_temp_factor -= 2.5
            PWM_factor += 0.025
        else:
            feedrate_factor += 0.01
            extrusion_factor -= 0.01
            ex_temp_factor -= 1
            PWM_factor += 0.01
    elif check([av_nodefects, av_overextrusion, av_underextrusion], av_stringing):
        defect = 'stringing'
        print(av_stringing)
        def_ratio = av_stringing
        if av_stringing > 0.6:
            extrusion_factor -= 0.05
            ex_temp_factor -= 5
            PWM_factor += 0.05
        elif av_stringing <= 0.6 and av_stringing > 0.3:
            extrusion_factor -= 0.025
            ex_temp_factor -= 2.5
            PWM_factor += 0.025
        else:
            extrusion_factor -= 0.01
            ex_temp_factor -= 1
            PWM_factor += 0.01
    elif check([av_nodefects, av_overextrusion, av_stringing], av_underextrusion):
        defect = 'underextrusion'
        print(av_underextrusion)
        def_ratio = av_underextrusion
        if av_underextrusion > 0.6:
            feedrate_factor -= 0.05
            extrusion_factor += 0.05
            ex_temp_factor += 5
            PWM_factor -= 0.05
        elif av_underextrusion <= 0.6 and av_underextrusion > 0.3:
            feedrate_factor -= 0.025
            extrusion_factor += 0.025
            ex_temp_factor += 2.5
            PWM_factor -= 0.025
        else:
            feedrate_factor -= 0.01
            extrusion_factor += 0.01
            ex_temp_factor += .1
            PWM_factor -= 0.01
    else: # no defects
        defect = 'No defects'
        def_ratio = av_nodefects
        
    if PWM_factor > 100:
        PWM_factor = 100
    elif PWM_factor < 0:
        PWM_factor = 0    
        
    return defect, number_files, feedrate_factor, extrusion_factor, ex_temp_factor, PWM_factor, def_ratio


### Connect to Ender3 (Serial Printer)
port_printer = [comport.device for comport in serial.tools.list_ports.comports()][0] 
print("Printer available on COM port: "+port_printer)
ender3 = serial.Serial(port_printer,baudrate=115200)
ender3.close()
ender3.open() 

time.sleep(1)


### Start Camera
camIdx = getCameraIndex('IPEVO V4K') ## Enter camera name or 'unsure' to get
                                     ## a list of available cameras and select
                                     ## desired index number before continuing
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(camIdx,cv2.CAP_DSHOW)
img_size_x = 2400 # Ideal pixel size x
img_size_y = 2400 # Ideal pixel size y
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, img_size_x)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size_y)
widthx = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)) # Actual pixel capture size x
heighty = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Actual pixel capture size y
imxy = '_'+str(widthx)+'W_'+str(heighty)+'H'


### Connect to SQL database
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=localhost\SQLEXPRESS;'
                      'Database=master;'
                      'Trusted_Connection=yes;')
defectsql = conn.cursor()
defectsql.execute('''
                                      INSERT INTO defectdetection_ender3 (linecount, layercount, defect, alt_temp, temp, fr_factor, alt_amount, fanPWM, capture, defect_ratio, predict_time, num_files)
                                      VALUES
                                      (?,?,?,?,?,?,?,?,?,?,?,?)
                                      ''',(0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0))

### Get times between layers
print('\n\n\n')
gfile_name = 'CE3_UMS3_random38_block_100fill'
gfile = "D:/gcode_ender3/"+gfile_name+".gcode"
#gfile_name = 'CE3_circle20mm_2mm'
#gfile = 'D:/gcode_ender3/'+gfile_name+'.gcode'
times_file = 'C:/Users/amari/UMS3_project/ME295AB/times.txt'
set_time_elapsed(gfile, times_file)

n_layers = 1 ## Capture images at every n layers

time_tempa = time.time()
### Get initial temperatures
extruder_temp, bed_temp = find_init_temperature(gfile)
set_temperature(ender3,extruder_temp,bed_temp)

### Load ML model
print("Loading model")
model_name = "C:/Users/amari/UMS3_project/ME295AB/ML_training/models/ethereal-glitter-293"
tic = time.time()
vgg16_model = load_model(model_name)
toc = time.time()
load_time = toc-tic
print(f'Model load time: {load_time}')
model_shape=vgg16_model.layers[0].input.get_shape().as_list()
shapehw = model_shape[1]
class_weights = [1, # No defects
           1.0, 0.5, # overextrusion_high, overextrusion_low
           1.0, 0.5, # stringing_high, stringing_low,
           0.75, 1.0, 0.25] # underextrusion, underextrusion_high, underextrusion_low



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


####### Start Printing #######
print('\n\nStart printing from file.\n\n')

gfile_print = open(gfile,'r')
times = open(times_file,'r')
linecount = 1
layercount = 0

X_line = Y_line = Z_line = 0
elapsed_time, layerbreak, Z = get_time_elapsed(times)
timepause = elapsed_time
time.sleep(2)

goal_X = 25

### Default values
fr_factor = 1.00 ### Feedrate adjustment factor, percent
alt_amount = 0.00 ### Extrusion adjustment factor, percent
alt_temp = 0.00 ### Temperature adjustment factor, deg F
fanPWM = 1.00 ### Fan Speed % (8-bit)


###### Print Loop ######
while True:
    try:
        line = gfile_print.readline()
        
        ### End of g-code file, end program
        endprint = re.findall(";Wipe out",line)
        if not line or endprint or linecount == final_line:
            print("\nFinished printing.\n") 
            adjust_extruder(ender3,-3,1 ) ## amount to retract in mm
            command(ender3,"G0 Y200 Z"+str(Z+10))
            cv2.destroyAllWindows()
            set_temperature(ender3, extruder_temp/2, bed_temp/2)
            wait_finish = input("\nConfirmpiece is removed from print bed by hitting 'enter'.\n")
            time.sleep(5)
            command(ender3,"G0 Y200 Z"+str(Z+10))
            ender3.close()
            break

        else:
            print("Layer "+str(layercount)+", Line "+str(linecount)+" of "+str(layerbreak))
            
            ###### Send g-code to Ender3
            command(ender3, line)

            if linecount == layerbreak-1: ### Get layer final X and Y positions
                X_line, Y_line = get_XY(line)
                if X_line > 0 and Y_line > 0:
                    X = X_line
                    Y = Y_line

            if linecount == layerbreak:
                print("\n\nLine: "+str(linecount)+", Break Layer: "+str(layercount)+"\n\n")

                if layercount % n_layers == 0:
                    ### Lower temperature, retract extruder, 
                    set_temperature(ender3,extruder_temp-15,bed_temp)
                    adjust_extruder(ender3,-8,1 ) ## amount to retract in mm

                    ### Position for Camera Capture
                    goal_Z = Z + 65
                    goal_Y = 135+(linecount%2)/10
                    command(ender3,"G0 Z"+str(goal_Z))
                    command(ender3,"G0 F5000 X"+str(goal_X)+" Y"+str(goal_Y)+" Z"+str(goal_Z))
                    ### Check position
                    time.sleep(1)
                    check_position(ender3, goal_X,goal_Y,goal_Z)

                    ### Capture overhead image
                    ts = str(time.strftime("%Y%m%d%H%M"))
                    predict_dir = os.path.join('ML_training\predict_imgs', str(ts)+'_Layer'+str(layercount)+'_'+gfile_name)
                    os.mkdir(predict_dir)
                    imfile = os.path.join(predict_dir,gfile_name+"_Layer"+str(layercount)+str(imxy)+".jpg")
                    video_capture(imfile,webcam)

                    ### Return to last XYZ position
                    command(ender3,"M104 S"+str(extruder_temp-5))
                    gpositionXY = "X"+str(X)+" Y"+str(Y)
                    gpositionZ = "Z"+str(Z)
                    command(ender3,"G0 F1100 "+gpositionZ+" "+gpositionXY)
                    #command(ender3,"G0 "+gpositionXY)
                    set_temperature(ender3,extruder_temp, bed_temp)

                    ### Evaluate images
                    stl_viz.gcode_overlay(widthx, heighty, gfile, predict_dir,imfile, layercount)
                    tic = time.time()
                    defect, number_files, fr_factor, alt_amount, alt_temp, fanPWM , def_amt = class_predictions(predict_dir,\
                        vgg16_model, shapehw, class_weights, fr_factor, alt_amount, alt_temp, fanPWM)
                    toc = time.time()
                    predict_time = toc-tic
                    print(f'Model prediction time for {number_files} images: {predict_time}')
                    
                    ###### Change printer settings ######
                    line = adjust_feedrate_amount(line, fr_factor)
                    line = adjust_extrusion_amount(line, alt_amount)
                    set_temperature(ender3, extruder_temp+alt_temp, bed_temp)
                    adjust_coolingfan_speed(ender3,fanPWM)
                    ###### Change printer settings ######
                    
                    ### Store data into SQL database
                    timestamp = datetime.utcnow()
                    defectsql.execute('''
                                      INSERT INTO defectdetection_ender3 (linecount, layercount, defect, alt_temp, temp, fr_factor, alt_amount, fanPWM, capture, defect_ratio, predict_time, num_files)
                                      VALUES
                                      (?,?,?,?,?,?,?,?,?,?,?,?)
                                      ''',(linecount, layercount, defect, alt_temp, extruder_temp, fr_factor, alt_amount, fanPWM, timestamp, def_amt,predict_time,number_files))
                                      
                    conn.commit()

                    timepause = 0

                elapsed_time, layerbreak, Z = get_time_elapsed(times)
                timepause += elapsed_time
                layercount += 1

            linecount += 1  

    ### Escape/Close program manually using ctrl+c
    except(KeyboardInterrupt):
        print("\nExit program.\n")
        adjust_extruder(ender3,-3,1 ) ## amount to retract in mm
        time.sleep(0.01)
        command(ender3,"G0 Y200 Z"+str(Z+10))
        wait_finish = input("\nConfirm piece is removed from print bed by hitting 'enter'.\n")
        time.sleep(2)
        cv2.destroyAllWindows()
        ender3.close()
        break
