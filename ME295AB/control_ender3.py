import os
import time
from datetime import datetime
import re
import cv2
import serial
import serial.tools.list_ports

def command(ser, line):
    matchM105 = re.findall("M105",line)
    if line != "" and line[0] != ";" and not matchM105:
        print(line)
        ser.write(str.encode(line+"\r\n")) 
        time.sleep(0.1)
        while True:
            line = ser.readline()
            print(line)
            if line == b'ok\n':
                break
    else:
        print(line)


### Connect to Ender3 (Serial Printer)
port_printer = [comport.device for comport in serial.tools.list_ports.comports()][0] 
ender3 = serial.Serial(port_printer,baudrate=115200)
print("Printer available on COM port: "+port_printer)
time.sleep(1)

####### Start Printing #######

time.sleep(2)

###### Print Loop ######
while True:
    try:
        line = input("Enter g-code: ")
        command(ender3, line)


    ### Escape/Close program manually using ctrl+c
    except(KeyboardInterrupt):
        print("\nExit program.\n")
        ender3.close()
        break