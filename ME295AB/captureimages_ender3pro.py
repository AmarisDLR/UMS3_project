import os
import time
import re
import cv2
from pygrabber.dshow_graph import FilterGraph

import serial

def getCameraIndex(camName):
    graph = FilterGraph()
    if camName == "unsure":
        print(graph.get_input_devices())# list of camera device 
        device = input("\nEnter index of desired camera: ")
    else:
        device = graph.get_input_devices().index(camName)
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
    time1 = 0
    count = 1
    Z = 0
    t = True
    while t:
        line = gfile_read.readline()
        if not line:
                global final_line
                final_line = count - 1
                t = False
                times.write(str(time_elapsed)+','+str(final_line)+','+str(Z+15)+'\n')
                break
        
        mesh_line = re.findall('NONMESH',line)

        if mesh_line:
            j = True
            while j:
                line = gfile_read.readline()
                count += 1
                Z = get_Z(line)
                if Z > 0:
                    j = False
                    break

        timeline = re.findall(';TIME_ELAPSED:',line)

        if timeline:
            time2 = re.findall(r'[\d.\d]+',line)
            time2 = float(time2[0])
            time_elapsed = time2 - time1
            times.write(str(time_elapsed)+','+str(count)+','+str(Z)+'\n')
            time1 = time2
        count += 1

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
            Ennn_alt = float(Ennn[0]) + alt_amount
            line = line.replace("E"+Ennn[0],"E"+str(Ennn_alt))
    return line

def adjust_coolingfan_speed(ser,PWM):
    command(ser,"M106 S"+str(PWM))

def video_capture(gfile_name, webcam, layerbreak):
    count = 1
    t = True
    while t:
        try:
            time.sleep(0.001)
            count += 1
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if count == 355: ### Give time for camera to autofocus
                            ### & allow time for user to view camera feed
                capture_image(gfile_name, webcam, frame, layerbreak)
                t = False
                time.sleep(2)
                break
        except(KeyboardInterrupt):
            cv2.destroyAllWindows()
            break

def capture_image(gfile_name, webcam, frame, layerbreak):
    cam_fps = webcam.get(cv2.CAP_PROP_FPS)
    print("Capture Image at %.2f FPS." %cam_fps)
    ts = time.strftime("%Y%m%d%H%M")
    imfile = "E:/AM_Papers/Ender3/directory/"+str(ts)+"_"+gfile_name+"_"+str(layerbreak)+".jpg"
    print(imfile)
    cv2.imwrite(filename=imfile, img=frame)
    print("Image saved!")

def check_position(ser, goalx, goaly, goalz):
    count = 0
    X = round(goalx,2)
    Y = round(goaly,2)
    Z = round(goalz,2)
    moving = True
    while moving:
        print("*",end="",flush=True)
        ser.write(str.encode("M114\r\n")) 
        line = ser.readline()
        if line != b'ok\n' and re.findall("Count",str(line)):
            line = str(line)
            print(line)
            split_line = line.split()
            xc = float(re.findall(r'[\d.\d]+', split_line[0])[0])
            yc = float(re.findall(r'[\d.\d]+', split_line[1])[0])
            zc = float(re.findall(r'[\d.\d]+', split_line[2])[0])
            x_compare = xc-0.01 <= X <= xc+0.1
            y_compare = yc-0.01 <= Y <= yc+0.1
            z_compare = zc-0.01 <= Z <= zc+0.1
            if x_compare and y_compare and z_compare:
                moving = False
        elif count > 1000:
            moving = False

        count +=1

### Connect to Ender3 (Serial Printer)
ender3 = serial.Serial('COM4',baudrate=115200)
time.sleep(2)

### Get times between layers
print('\n\n\n')
gfile_name = 'CE3_random1'
gfile = "C:/Users/amari/Downloads/gcode_ender3/"+gfile_name+".gcode"
times_file = "E:/AM_Papers/Ender3/times.txt"
set_time_elapsed(gfile, times_file)

n_layers = 1 ## Capture images at every n layers

### Get initial temperatures
extruder_temp, bed_temp = find_init_temperature(gfile)
set_temperature(ender3,extruder_temp,bed_temp)

### Start Camera
camIdx = getCameraIndex('IPEVO V4K') ## Enter camera name or 'unsure' to get
                                     ## a list of available cameras and select
                                     ## desired index number
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(camIdx,cv2.CAP_DSHOW)
img_size_x = 1600
img_size_y = 1200
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, img_size_x)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size_y)

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

fr_factor = 0.725 ### Feedrate adjustment factor
alt_amount = 5 ### Extrusion adjustment factor
alt_temp = 5 ### Temperature adjustment factor
fanPWM = 100 ### PWM <0-255>

###### Print Loop ######
while True:
    try:
        line = gfile_print.readline()

        if not line or linecount == final_line or layercount == 1:
            print("\nFinished printing.\n") ### End of g-code file
            adjust_extruder(ender3,-3,1 ) ## amount to retract in mm
            command(ender3,"G0 Y200")
            cv2.destroyAllWindows()
            ender3.close()
            wait_finish = input("\nConfirmpiece is removed from print bed by hitting 'enter'.\n")
            break

        elif key == ord('r'): ### Reset
            print("\n\nRESET\n")
            time.sleep(3)
            command(ender3,"G28")
            gfile_print.seek(0)
            times.seek(0)
            linecount = 1
            layercount = 0

            X_line = Y_line = Z_line = 0
            elapsed_time, layerbreak, Z = get_time_elapsed(times)
            timepause = elapsed_time
            time.sleep(2)

        else:
            print("Layer "+str(layercount)+", Line "+str(linecount)+" of "+str(layerbreak))
            
            ### Change feedrate by factor of fr_factor
            if linecount > 30:
                if layercount >= 0:
                    line = adjust_feedrate_amount(line, fr_factor)
                else:
                    line = adjust_feedrate_amount(line, 0.85)
                ### Change extrusion amount by alt_amount
                if layercount >= 0:
                    line = adjust_extrusion_amount(line, alt_amount)
                ### Change temperature by alt_temp
                if linecount == 350:
                    extruder_temp += alt_temp
                    print("Extruder Temp: "+str(extruder_temp))
                    set_temperature(ender3, extruder_temp, bed_temp)
                else:
                    print(extruder_temp)
                ### Change fan speed (PWM)
                if layercount >= 0:
                    adjust_coolingfan_speed(ender3,fanPWM)

            ### Send g-code to Ender3
            command(ender3, line)
            endprint = re.findall(";Wipe out",line)
            if endprint:
                command(ender3, "G90 ;Absolute positioning")
                command(ender3,"G0 Z"+str(goal_Z))
                linecount = final_line

            if linecount == layerbreak-1: ### Get final X and Y positions of layer
                X_line, Y_line = get_XY(line)
                if X_line > 0 and Y_line > 0:
                    X = X_line
                    Y = Y_line

            if linecount == layerbreak:
                print("\n\nLine: "+str(linecount)+", End Layer: "+str(layercount)+"\n\n")

                if layercount % n_layers == 0:
                    ### Retract extruder
                    set_temperature(ender3,extruder_temp-25,bed_temp)
                    adjust_extruder(ender3,-8,1 ) ## amount to retract in mm

                    ### Position for Camera Capture
                    goal_Z = Z + 55
                    goal_Y = 135+(linecount%2)/10
                    command(ender3,"G0 Z"+str(goal_Z))
                    command(ender3,"G0 F5000 X"+str(goal_X)+" Y"+str(goal_Y)+" Z"+str(goal_Z))
                    ### Check position
                    time.sleep(2)
                    check_position(ender3, goal_X,goal_Y,goal_Z)

                    ### Capture overhead image
                    video_capture(gfile_name,webcam,layercount)

                    ### Return to last XYZ position
                    command(ender3,"M109 S"+str(extruder_temp-5))
                    gpositionXY = " X"+str(X)+" Y"+str(Y)
                    gpositionZ = "Z"+str(Z)
                    command(ender3,"G0 "+gpositionZ+gpositionXY)
                    #command(ender3,"G0 "+gpositionXY)
                    set_temperature(ender3,extruder_temp, bed_temp)
                    time.sleep(1)
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
        command(ender3,"G0 Y200 Z45")
        wait_finish = input("\nConfirm piece is removed from print bed by hitting 'enter'.\n")
        cv2.destroyAllWindows()
        ender3.close()
        break
