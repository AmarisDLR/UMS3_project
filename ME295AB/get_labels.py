import os

fileset = "trainval"

files = os.listdir("../../../Downloads/UMS3_FFF_defects5/"+fileset)

filedir = open("../../../Downloads/UMS3_FFF_defects5/"+fileset+"/"+fileset+".txt","w+")

print(fileset)
for i in files:
	j = i[:-4]
	filedir.write(str(j)+"\n")
	
