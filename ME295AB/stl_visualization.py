
from math import ceil
import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits import mplot3d

import meshcut
from stl import mesh
import numpy.linalg as la

#--- for gcode parser
from pygcode import *
from pygcode import Line
from pygcode import Machine, GCodeRapidMove

def points3d(verts, point_size=3, **kwargs):
	if 'mode' not in kwargs:
		kwargs['mode'] = 'point'
	p = mlab.points3d(verts[:, 0], verts[:, 1], verts[:, 2], **kwargs)
	p.actor.property.point_size = point_size

def trimesh3d(verts, faces, **kwargs):
	mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, **kwargs)
	
def orthogonal_vector(v):
	### Return an arbitrary vector that is orthogonal to v ###
	if v[1] != 0 or v[2] != 0:
		c = (1, 0, 0)
	else:
		c = (0, 1, 0)
	return np.cross(v, c)

def show_plane(orig, n, scale=1.0, **kwargs):
	### Show the plane with the given origin and normal. scale give its size ###
	b1 = orthogonal_vector(n)
	b1 /= la.norm(b1)
	b2 = np.cross(b1, n)
	b2 /= la.norm(b2)
	verts = [orig + scale*(-b1 - b2),
		orig + scale*(b1 - b2),
		orig + scale*(b1 + b2),
		orig + scale*(-b1 + b2)]
	faces = [(0, 1, 2), (0, 2, 3)]
	trimesh3d(np.array(verts), faces, **kwargs)
	
def load_stl(stl_fname):
	import stl
	m = stl.mesh.Mesh.from_file(stl_fname)

	### Flatten our vert array to Nx3 and generate corresponding faces array ###
	verts = m.vectors.reshape(-1, 3)
	faces = np.arange(len(verts)).reshape(-1, 3)

	verts, faces = meshcut.merge_close_vertices(verts, faces)
	return verts, faces

def show(plane):
	P = meshcut.cross_section_mesh(mesh, plane)
	colors = [(0, 1, 1), (1, 0, 1),	(0, 0, 1)]
	print("num contours : ", len(P))

	if True:
		utils.trimesh3d(mesh.verts, mesh.tris, color=(1, 1, 1),
			opacity=0.5, representation='wireframe')
		utils.show_plane(plane.orig, plane.n, scale=1, color=(1, 0, 0),
			opacity=0.5)

	for p, color in zip(P, itertools.cycle(colors)):
		p = np.array(p)
		mlab.plot3d(p[:, 0], p[:, 1], p[:, 2], tube_radius=None,
			line_width=3.0, color=color)
	return P

#################

def stl_3Dworkspace(img_size_x, img_size_y, stl_file, LAYER_NUMBER, im_3DWorkspace, im_Top): 

	layer_height = 0.1
	z_height = layer_height*LAYER_NUMBER
	print("\nz-height for layer {}: {}mm\n".format(LAYER_NUMBER,z_height))

	#################
	zoom = 0.06875 # picture scale

	#################
		
	verts1, faces1 = load_stl(stl_file)
	mesh1 = mesh.Mesh.from_file(stl_file)
	
	volume1, cog1, inertia1 = mesh1.get_mass_properties()
	print("Volume                                  = {0}".format(volume1))
	print("Position of the center of gravity (COG) = {0}".format(cog1))
	print("Inertia matrix at expressed at the COG  = {0}".format(inertia1[0,:]))
	print("                                          {0}".format(inertia1[1,:]))
	print("                                          {0}".format(inertia1[2,:]))

	mesh_plane = meshcut.TriangleMesh(verts1, faces1)

	def show(mesh,plane):
		P = meshcut.cross_section_mesh(mesh, plane)
		if True:
			utils.trimesh3d(mesh.verts, mesh.tris, color=(1, 1, 1),
				opacity=0.5, representation='wireframe')
			utils.show_plane(plane.orig, plane.n, scale=1, color=(1, 0, 0),
				opacity=0.5)
		for p, color in zip(P, itertools.cycle(colors)):
			p = np.array(p)
			mlab.plot3d(p[:, 0], p[:, 1], p[:, 2], tube_radius=None,
				line_width=3.0, color=color)
		return P

	plane_orig_1 = (0, 0, z_height)
	plane_norm_1 = (0, 0, 1)
	plane_norm_1 /= la.norm(plane_norm_1)

	stl_plane_1 = meshcut.Plane(plane_orig_1, plane_norm_1)

	P1 = meshcut.cross_section_mesh(mesh_plane,stl_plane_1)
	print('Slice: no of shapes = {}'.format(np.shape(P1)))

	#################

	part_height = np.max(mesh1.vectors[:,:,2])-np.min(mesh1.vectors[:,:,2])
	print('Total height of the part = {} mm'.format(part_height))

	#################
	workspace_definition = [(-zoom*img_size_x/2,-zoom*img_size_y/2,0),\
				 (-zoom*img_size_x/2,zoom*img_size_y/2,0),\
				 (zoom*img_size_x/2,-zoom*img_size_y/2,0),\
				 (-zoom*img_size_x/2,-zoom*img_size_y/2,part_height)]
	workspace_definition_array = [np.array(list(item)) for item in workspace_definition]

	### Origin
	x = y = z = 0

	xx_base, yy_base = np.meshgrid(range(-100,100), range(-100,100))
	zz_base = np.zeros((200,200), dtype=int)

	#################
	### Object Transformation H
	otheta_x = 0.0 # degrees
	otheta_y = 0.0 # degrees
	otheta_z = 0.0 # degrees

	ot_x = 0.0
	ot_y = 0.0
	ot_z = 0.0

	oRx = np.array([[1,0,0],[0,np.cos(otheta_x*np.pi/180),-np.sin(otheta_x*np.pi/180)],\
			[0,np.sin(otheta_x*np.pi/180),np.cos(otheta_x*np.pi/180)]]) # rotation around x
	oRy = np.array([[np.cos(otheta_y*np.pi/180),0,np.sin(otheta_y*np.pi/180)],[0,1,0],\
			[-np.sin(otheta_y*np.pi/180),0,np.cos(otheta_y*np.pi/180)]]) # rotation around y
	oRz = np.array([[np.cos(otheta_z*np.pi/180),-np.sin(otheta_z*np.pi/180),0],\
			[np.sin(otheta_z*np.pi/180),np.cos(otheta_z*np.pi/180),0],[0,0,1]]) # rotation around z

	oR = np.dot(np.dot(oRx,oRy),oRz)
	ot = np.array([ot_x,ot_y,ot_z])

	H = np.zeros((4,4), dtype=float)
	H[0:3,0:3] = oR
	H[0:3,3] = ot.T
	H[3,3] = 1

	#################

	### Camera Transformation C

	theta_x = -180 # degrees
	theta_y = 0 # degrees
	theta_z = 0 # degrees

	t_x = 0.0
	t_y = 0.0
	t_z = 40.0+z_height 

	Rx = np.array([[1,0,0],[0,np.cos(theta_x*np.pi/180),-np.sin(theta_x*np.pi/180)],\
			[0,np.sin(theta_x*np.pi/180),np.cos(theta_x*np.pi/180)]]) # rotation around x
	Ry = np.array([[np.cos(theta_y*np.pi/180),0,np.sin(theta_y*np.pi/180)],[0,1,0],\
			[-np.sin(theta_y*np.pi/180),0,np.cos(theta_y*np.pi/180)]]) # rotation around y
	Rz = np.array([[np.cos(theta_z*np.pi/180),-np.sin(theta_z*np.pi/180),0],\
			[np.sin(theta_z*np.pi/180),np.cos(theta_z*np.pi/180),0],[0,0,1]]) # rotation around z

	R = np.dot(np.dot(Rx,Ry),Rz)
	t = np.array([t_x,t_y,t_z])

	C = np.zeros((4,4), dtype=float)
	C[0:3,0:3] = R
	C[0:3,3] = t.T
	C[3,3] = 1
	
	#################
	### Camera Location
	cam_main = np.array([0,0,0,1]) # Main Camera Location
	cam_main_x = np.array([25,cam_main[1],cam_main[2],1])
	cam_main_y = np.array([cam_main[0],25,cam_main[2],1])
	cam_main_z = np.array([cam_main[0],cam_main[1],25,1])

	cam_main = np.array([cam_main,cam_main_x,cam_main_y,cam_main_z])

	#################
	### Watching Distance
	watch = np.array([0,0,25,1])

	### Apply Transformation C to camera
	cam_main_tr = np.zeros((4,4), dtype=float)
	watch_tr = np.zeros((1,4), dtype=float)
	#################

	#------------------------------------------------- Transformed Cam Position
	for i in range(np.shape(cam_main_tr)[0]):
		cam_main_tr[i] = np.dot(C, cam_main[i])
	#------------------------------------------------- Transformed Watching Distance
	watch_tr = np.dot(C, watch)
	#------------------------------------------------- Watching Distance (vectors and plane)

	# create x,y
	xx_tr, yy_tr = np.meshgrid(range(int(-zoom*img_size_x/2),int(zoom*img_size_x/2)),\
			range(int(-zoom*img_size_y/2),int(zoom*img_size_y/2)))
	# corresponding z
	mesh_dim = np.shape(xx_tr)
	zz_tr = z_height*np.ones([mesh_dim[0],mesh_dim[1]])
	#################

	### Cubic printing zone

	points = []
	points += workspace_definition_array
	vectors = [workspace_definition_array[1] - workspace_definition_array[0],
		workspace_definition_array[2] - workspace_definition_array[0],
		workspace_definition_array[3] - workspace_definition_array[0]]

	points += [workspace_definition_array[0] + vectors[0] + vectors[1]]
	points += [workspace_definition_array[0] + vectors[0] + vectors[2]]
	points += [workspace_definition_array[0] + vectors[1] + vectors[2]]
	points += [workspace_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

	points = np.array(points)

	edges = [
		[points[0], points[3], points[5], points[1]],
		[points[1], points[5], points[7], points[4]],
		[points[4], points[2], points[6], points[7]],
		[points[2], points[6], points[3], points[0]],
		[points[0], points[2], points[4], points[1]],
		[points[3], points[6], points[7], points[5]]]

	#################
	
	faces = Poly3DCollection(edges, linewidths=1, edgecolors='gray')
	faces.set_facecolor((0,0,1,0.03))
	
	fig = plt.figure(figsize=(18,18), dpi=80)
	ax = Axes3D(fig)
	
	# -------------------------------- Printing Bed
	ax.add_collection3d(faces)
	ax.plot_surface(xx_base,yy_base,zz_base, color='slategray', alpha=0.3)
	# -------------------------------- End Printing Bed
	
	# -------------------------------- Origin (EXTENSION LINES)
	ax.scatter3D(x,y,z,color='black',s=50)
	ax.plot([x,5],[y,y],[z,z],color = 'r')
	ax.plot([x,x],[y,-5],[0,0],color = 'g')
	ax.plot([x,x],[y,y],[0,5],color = 'b')
	# --------------------------------End Origin

	#------------------------------------------------- Transformed Camera
	ax.scatter(cam_main_tr[0][0],cam_main_tr[0][1],cam_main_tr[0][2],c='r',s=50) # origin
	ax.scatter(cam_main_tr[1][0],cam_main_tr[1][1],cam_main_tr[1][2],c='r') # x2
	ax.plot([cam_main_tr[0][0],cam_main_tr[1][0]],[cam_main_tr[0][1],cam_main_tr[1][1]],\
		[cam_main_tr[0][2],cam_main_tr[1][2]],c='r')

	ax.scatter(cam_main_tr[2][0],cam_main_tr[2][1],cam_main_tr[2][2],c='g') # y2
	ax.plot([cam_main_tr[0][0],cam_main_tr[2][0]],[cam_main_tr[0][1],cam_main_tr[2][1]],\
		[cam_main_tr[0][2],cam_main_tr[2][2]],c='g')

	ax.scatter(cam_main_tr[3][0],cam_main_tr[3][1],cam_main_tr[0][2]-10,c='b') # z2
	ax.plot([cam_main_tr[0][0],cam_main_tr[3][0]],[cam_main_tr[0][1],cam_main_tr[3][1]],\
		[cam_main_tr[0][2],cam_main_tr[0][2]-10],c='b')

	ax.plot([cam_main_tr[0][0],watch_tr[0]],[cam_main_tr[0][1],watch_tr[1]],\
	[cam_main_tr[0][2],z_height],':',c='cyan')
	#------------------------------------------------- End Transformed Camera

	#------------------------------------------------- Transformed Watching distance (PLANE)
	ax.plot_surface(xx_tr,yy_tr,zz_tr,color='cyan', alpha=0.05) ## plane
	ax.scatter(z,y,z_height,c='cyan')
	#------------------------------------------------- End Transformed Watching distance (PLANE)

	#------------------------------------------------- Transformed Frame (OUTLINE)
	ax.scatter(xx_tr[0][0],yy_tr[0][0],zz_tr[0][0],color='mediumturquoise',s=50)
	ax.scatter(xx_tr[-1][0],yy_tr[-1][0],zz_tr[-1][0],color='mediumturquoise',s=50)
	ax.scatter(xx_tr[-1][-1],yy_tr[-1][-1],zz_tr[-1][-1],color='mediumturquoise',s=50)
	ax.scatter(xx_tr[0][-1],yy_tr[0][-1],zz_tr[0][-1],color='mediumturquoise',s=50)

	ax.plot([xx_tr[0][0],xx_tr[-1][0]],[yy_tr[0][0],yy_tr[-1][0]],[zz_tr[0][0],zz_tr[-1][0]],c='mediumturquoise')
	ax.plot([xx_tr[-1][0],xx_tr[-1][-1]],[yy_tr[-1][0],yy_tr[-1][-1]],[zz_tr[-1][0],zz_tr[-1][-1]],c='mediumturquoise')
	ax.plot([xx_tr[0][-1],xx_tr[-1][-1]],[yy_tr[0][-1],yy_tr[-1][-1]],[zz_tr[0][-1],zz_tr[-1][-1]],c='mediumturquoise')
	ax.plot([xx_tr[0][-1],xx_tr[0][0]],[yy_tr[0][-1],yy_tr[0][0]],[zz_tr[0][-1],zz_tr[0][0]],c='mediumturquoise')

	ax.plot([cam_main_tr[0][0],xx_tr[-1][0]],[cam_main_tr[0][1],\
		yy_tr[-1][0]],[cam_main_tr[0][2],zz_tr[-1][0]],':',c='mediumturquoise')
	ax.plot([cam_main_tr[0][0],xx_tr[0][-1]],[cam_main_tr[0][1],\
		yy_tr[0][-1]],[cam_main_tr[0][2],zz_tr[0][-1]],':',c='mediumturquoise')
	ax.plot([cam_main_tr[0][0],xx_tr[-1][-1]],[cam_main_tr[0][1],\
		yy_tr[-1][-1]],[cam_main_tr[0][2],zz_tr[-1][-1]],':',c='mediumturquoise')
	ax.plot([cam_main_tr[0][0],xx_tr[0][0]],[cam_main_tr[0][1],\
		yy_tr[0][0]],[cam_main_tr[0][2],zz_tr[0][0]],':',c='mediumturquoise')
	#------------------------------------------------- End Transformed Frame (OUTLINE)
	
	# ------------------------------------------------------ Add stl
	for i in range(np.shape(mesh1.normals)[0]):
		if((np.dot(mesh1.normals[i],mesh1.vectors[i])[0])>0.0 and (np.dot(mesh1.normals[i],mesh1.vectors[i])[2])>0.0):
			ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh1.vectors[i:i+1],alpha=0.4,facecolor='lightcoral'))
		else:
			ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh1.vectors[i:i+1],alpha=0.4,facecolor='orangered'))

	ax.add_collection3d(mplot3d.art3d.Line3DCollection(mesh1.vectors,alpha=1,linewidths=1,\
		color=[0.1,0.1,0.1],linestyle=':'))

	# COG
	ax.scatter(cog1[0],cog1[1],cog1[2],color='navy',marker='+',s=350)
	ax.scatter(0,0,0,color='navy',marker='x',s=350)

	for i in range(np.shape(P1[0])[0]):
		ax.scatter(P1[0][i][0],P1[0][i][1],P1[0][i][2],color='lightseagreen',s=20) ## OUTLINE
		ax.plot([P1[0][i][0],P1[0][i-1][0]],[P1[0][i][1],P1[0][i-1][1]],[P1[0][i][2],P1[0][i-1][2]],\
			color='lightseagreen',linewidth=7)	## OUTLINE OF PART

	# ------------------------------------------------------ End Add stl
	
	ax.set_xlabel('X, mm')
	ax.set_ylabel('Y, mm')
	ax.set_zlabel('Z, mm')
	ax.grid(False)
	ax.set_xlim(-80,80)
	ax.set_ylim(-80,80)
	ax.set_zlim(0,50+ceil(z_height/10)*10)

	ax.set_title('3D Printer Workspace')
	plt.savefig(im_3DWorkspace)

	# ------------------------------------------------------

	# STL slice 

	fig = plt.figure(figsize=(18,18*img_size_y/img_size_x), dpi=80)

	for i in range(np.shape(P1[0])[0]):
		plt.scatter(P1[0][i][0],P1[0][i][1],color='blue',s=20)
		plt.plot([P1[0][i][0],P1[0][i-1][0]],[P1[0][i][1],P1[0][i-1][1]],color='blue',linewidth=2)

	plt.xlabel('X, mm')
	plt.ylabel('Y, mm')
	plt.title('Top view from STL')
	plt.savefig(im_Top)
	print('\nImages saved.\n')


def gcode_overlay(img_size_x, img_size_y, gcode_file, image_and_layer, LAYER_NUMBER, im_projection):
	### Plot overlay transparency
	im_alpha = 0.6

	### Picture scale
	zoom = 0.06875 
	### Intrinsic Camera Parameters

	fx = 1000.0
	fy = 1000.0
	cx = 0
	cy = 0

	cam_mtx = np.array([[fx,        0,     cx], [0,         fy,    cy], [0,         0,     1]])

	#################

	### Intrinsic camera parameters (obtained on the calibration stage)
	### Source images have already been undistorted

	camera_intrinsic_K = np.array(
				[[1552.3, 0,      650.1],
				[0,       1564.8, 486.2],
				[0,       0,      1]], dtype = "float")

	#################

	# Project G-Code / STL on image

	#2D image points, [pixels]
	image_points = np.array([
				(0, 0),
				(img_size_x, 0),
				(img_size_x, img_size_x),
				(0, img_size_x),
				], dtype="double")
	 
	# 3D model points
	model_points = np.array([
				(-45.0, -45.0, 0.0),
				(-45.0, 45.0, 0.0),
				(45.0, 45.0, 0.0),
				(45.0, -45.0, 0.0)
				])

	src = cv2.imread(image_and_layer)
	img = src.copy()

	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,\
					camera_intrinsic_K, dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)
	 
	# Object rotation
	otheta_x = 0.0 # degrees
	otheta_y = 0.0 # degrees
	otheta_z = -90.0 # degrees

	ot_x = -16.0
	ot_y = 4.5
	ot_z = -28.0

	oRx = np.array([[1,0,0],[0,np.cos(otheta_x*np.pi/180),-np.sin(otheta_x*np.pi/180)],\
			[0,np.sin(otheta_x*np.pi/180),np.cos(otheta_x*np.pi/180)]]) # rotation around x axis
	oRy = np.array([[np.cos(otheta_y*np.pi/180),0,np.sin(otheta_y*np.pi/180)],[0,1,0],\
			[-np.sin(otheta_y*np.pi/180),0,np.cos(otheta_y*np.pi/180)]]) # rotation around y axis
	oRz = np.array([[np.cos(otheta_z*np.pi/180),-np.sin(otheta_z*np.pi/180),0],\
			[np.sin(otheta_z*np.pi/180),np.cos(otheta_z*np.pi/180),0],[0,0,1]]) # rotation around z axis

	oR = np.dot(np.dot(oRx,oRy),oRz)
	ot = np.array([ot_x,ot_y,ot_z])

	H = np.zeros((4,4), dtype=float)
	H[0:3,0:3] = oR
	H[0:3,3] = ot.T
	H[3,3] = 1

	# layer number

	# TYPE:WALL-OUTER          -- 1
	# TYPE:WALL-INNER          -- 2
	# TYPE:FILL                -- 3
	# TYPE:SUPPORT             -- 4
	# TYPE:SUPPORT-INTERFACE   -- 5

	word_bank  = []
	layer_bank = [] # number of layer
	type_bank = []
	line_bank = []
	parsed_Num_of_layers = 0
	gcode_type = 0

	with open(gcode_file, 'r') as fh:
		for line_text in fh.readlines():
			line = Line(line_text) # all lines in file
			w = line.block.words # splits blocks into XYZEF, omits comments
			if(np.shape(w)[0] == 0): # if line is empty, i.e. comment line -> then skip it
				pass
			else:
				word0 = str(w[0])
				if word0[0] =='G':
					word_bank.append(w) # <Word: G01>, <Word: X15.03>, <Word: Y9.56>, <Word: Z0.269>, ...
					layer_bank.append(parsed_Num_of_layers)
					type_bank.append(gcode_type)
					line_bank.append(line_text)

			if line.comment:
				if (line.comment.text[0:6] == "LAYER:"):
					parsed_Num_of_layers = parsed_Num_of_layers + 1
					gcode_type = 0
				if line.comment:
					if (line.comment.text[0:15] == "TYPE:WALL-OUTER"):
						gcode_type = 1
					if (line.comment.text[0:15] == "TYPE:WALL-INNER"):
						gcode_type = 2
					if (line.comment.text[0:9] == "TYPE:FILL"):
						gcode_type = 3
					if (line.comment.text[0:12] == "TYPE:SUPPORT"):
						gcode_type = 4
					if (line.comment.text[0:22] == "TYPE:SUPPORT-INTERFACE"):
						gcode_type = 5

	print("parsed_Num_of_layers: {}".format(parsed_Num_of_layers))
	print("Layer capture: {}".format(LAYER_NUMBER))

	#######################
	fig = plt.figure(figsize=(18,14), dpi=80)
	plt.imshow(img)

	plt.scatter(0,0,c='springgreen',s=30)
	plt.scatter(img_size_x,0,c='springgreen',s=30)
	plt.scatter(img_size_x,img_size_y,c='springgreen',s=30)
	plt.scatter(0,img_size_y,c='springgreen',s=30)

	plt.plot([0,img_size_x],[0,0],c='springgreen')
	plt.plot([img_size_x,img_size_x],[0,img_size_y],c='springgreen')
	plt.plot([0,img_size_x],[img_size_y,img_size_y],c='springgreen')
	plt.plot([0,0],[img_size_y,0],c='springgreen')

	plt.plot([0,img_size_x],[0,img_size_y],c='springgreen',linestyle=':')
	plt.plot([img_size_x,0],[0,img_size_y],c='springgreen',linestyle=':')
	
	plt.xlabel('X, mm')
	plt.ylabel('Y, mm')
	locsx, labelsx = plt.xticks()
	labelsx = [round((float(itemx)-img_size_x/2)*zoom) for itemx in locsx]
	plt.xticks(locsx,labelsx)
	
	locsy, labelsy = plt.yticks()
	labelsy = [round((float(itemy)-img_size_y/2)*zoom) for itemy in locsy]
	plt.yticks(locsy,labelsy)
	#######################

	for k in [LAYER_NUMBER+2]: # layers

		X_active_bank = []
		Y_active_bank = []
		Z_active_bank = []
		G_active_bank = []
		E_active_bank = []
		F_active_bank = []

		idx = []
	    
		X_active_default = []; X_active_wall_outer = []; X_active_wall_inner = []
		Y_active_default = []; Y_active_wall_outer = []; Y_active_wall_inner = []
		Z_active_default = []; Z_active_wall_outer = []; Z_active_wall_inner = []
		G_active_default = []; G_active_wall_outer = []; G_active_wall_inner = []
		E_active_default = []; E_active_wall_outer = []; E_active_wall_inner = []
		F_active_default = []; F_active_wall_outer = []; F_active_wall_inner = []

		X_active_fill = []; X_active_support = []; X_active_support_interface = []
		Y_active_fill = []; Y_active_support = []; Y_active_support_interface = []
		Z_active_fill = []; Z_active_support = []; Z_active_support_interface = []
		G_active_fill = []; G_active_support = []; G_active_support_interface = []
		E_active_fill = []; E_active_support = []; E_active_support_interface = []
		F_active_fill = []; F_active_support = []; F_active_support_interface = []

		for i in range(len(layer_bank)): # for each line in file
			if (layer_bank[i] == k):
				idx.append(i)
				for j in range(len(word_bank[i])):
					if (str(word_bank[i][j])[:1] == 'G0'):
						G_active_bank.append(float(str(word_bank[i][j])[1:]))
					if (str(word_bank[i][j])[:1] == 'X'):
						X_active_bank.append(float(str(word_bank[i][j])[1:]))
					if (str(word_bank[i][j])[:1] == 'Y'):
						Y_active_bank.append(float(str(word_bank[i][j])[1:]))
					if (str(word_bank[i][j])[:1] == 'Z'):
						Z_active_bank.append(float(str(word_bank[i][j])[1:]))
					if (str(word_bank[i][j])[:1] == 'E'):
						E_active_bank.append(float(str(word_bank[i][j])[1:]))
					if (str(word_bank[i][j])[:1] == 'F'):
						F_active_bank.append(float(str(word_bank[i][j])[1:]))

		for m in range(len(X_active_bank)):
			if(type_bank[np.min(idx)+m] == 0):
				X_active_default.append(X_active_bank[m])
				Y_active_default.append(Y_active_bank[m])
				if m < len(Z_active_bank):
					Z_active_default.append(Z_active_bank[m])
			if(type_bank[np.min(idx)+m] == 1):
				X_active_wall_outer.append(X_active_bank[m])
				Y_active_wall_outer.append(Y_active_bank[m])
				if m < len(Z_active_bank):
					Z_active_wall_outer.append(Z_active_bank[m])
			if(type_bank[np.min(idx)+m] == 2):
				X_active_wall_inner.append(X_active_bank[m])
				Y_active_wall_inner.append(Y_active_bank[m])
				if m < len(Z_active_bank):
					Z_active_wall_inner.append(Z_active_bank[m])
			if(type_bank[np.min(idx)+m] == 3):
				X_active_fill.append(X_active_bank[m])
				Y_active_fill.append(Y_active_bank[m])
				if m < len(Z_active_bank):
					Z_active_fill.append(Z_active_bank[m])
			if(type_bank[np.min(idx)+m] == 4):
				X_active_support.append(X_active_bank[m])
				Y_active_support.append(Y_active_bank[m])
				if m < len(Z_active_bank):
					Z_active_support.append(Z_active_bank[m])
			if(type_bank[np.min(idx)+m] == 5):
				X_active_support_interface.append(X_active_bank[m])
				Y_active_support_interface.append(Y_active_bank[m])
				if m < len(Z_active_bank):
					Z_active_support_interface.append(Z_active_bank[m])

		if not Z_active_default:
			Z_active_default = Z_active_bank

		G_default = np.zeros((np.shape(X_active_default)[0],4),dtype=np.float32)
		G_wall_outer = np.zeros((np.shape(X_active_wall_outer)[0],4),dtype=np.float32)
		G_wall_inner = np.zeros((np.shape(X_active_wall_inner)[0],4),dtype=np.float32)
		G_fill = np.zeros((np.shape(X_active_fill)[0],4),dtype=np.float32)
		G_support = np.zeros((np.shape(X_active_support)[0],4),dtype=np.float32)

		if X_active_default:
			G_default[:,0] = X_active_default
			G_default[:,1] = Y_active_default
			G_default[:,2] = Z_active_default
			G_default[:,3] = np.ones((1,np.shape(X_active_default)[0]),dtype=np.float32)

		if G_default.any():
			tG_default = np.zeros((np.shape(G_default)[0],4), dtype=np.float32)
			for i in range(np.shape(G_default)[0]):
				tG_default[i] = np.dot(H,G_default[i])
			tGp_default = cv2.projectPoints(np.asarray(tG_default[:,0:3],dtype=float),rotation_vector,\
					translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
			for i in range(np.shape(tGp_default)[0]):
				plt.plot([tGp_default[i][0],tGp_default[i-1][0]],\
				[tGp_default[i][1],tGp_default[i-1][1]],alpha=im_alpha,color='sienna')

		if X_active_wall_outer:
			G_wall_outer[:,0] = X_active_wall_outer
			G_wall_outer[:,1] = Y_active_wall_outer
			if Z_active_wall_outer:
				G_wall_outer[:,2] = Z_active_wall_outer
			else:
				G_wall_outer[:,2] = Z_active_default
			G_wall_outer[:,3] = np.ones((1,np.shape(X_active_wall_outer)[0]),dtype=np.float32)

		if G_wall_outer.any():
			tG_wall_outer = np.zeros((np.shape(G_wall_outer)[0],4), dtype=np.float32)
			for i in range(np.shape(G_wall_outer)[0]):
				tG_wall_outer[i] = np.dot(H,G_wall_outer[i])
			tGp_wall_outer = cv2.projectPoints(np.asarray(tG_wall_outer[:,0:3],dtype=float),rotation_vector,\
				translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
			for i in range(np.shape(tGp_wall_outer)[0]-1):
				plt.plot([tGp_wall_outer[i][0],tGp_wall_outer[i-1][0]],\
				[tGp_wall_outer[i][1],tGp_wall_outer[i-1][1]],alpha=im_alpha,color='deepskyblue',linewidth=4)

		if X_active_wall_inner:
			G_wall_inner[:,0] = X_active_wall_inner
			G_wall_inner[:,1] = Y_active_wall_inner
			if Z_active_wall_inner:
				G_wall_inner[:,2] = Z_active_wall_inner
			else:
				G_wall_inner[:,2] = Z_active_default
			G_wall_inner[:,3] = np.ones((1,np.shape(X_active_wall_inner)[0]),dtype=np.float32)

		if G_wall_inner.any():
			tG_wall_inner = np.zeros((np.shape(G_wall_inner)[0],4), dtype=np.float32)
			for i in range(np.shape(G_wall_inner)[0]):
				tG_wall_inner[i] = np.dot(H,G_wall_inner[i])
			tGp_wall_inner = cv2.projectPoints(np.asarray(tG_wall_inner[:,0:3],dtype=float),rotation_vector,\
				translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
			for i in range(np.shape(tGp_wall_inner)[0]):
				plt.plot([tGp_wall_inner[i][0],tGp_wall_inner[i-1][0]],\
				[tGp_wall_inner[i][1],tGp_wall_inner[i-1][1]],alpha=im_alpha,color='tomato')

		if X_active_fill:
			G_fill[:,0] = X_active_fill
			G_fill[:,1] = Y_active_fill
			if Z_active_fill:
				G_fill[:,2] = Z_active_fill
			else:
				G_fill[:,2] = Z_active_default
			G_fill[:,3] = np.ones((1,np.shape(X_active_fill)[0]),dtype=np.float32)

		if G_fill.any():
			tG_fill = np.zeros((np.shape(G_fill)[0],4), dtype=np.float32)
			for i in range(np.shape(G_fill)[0]):
				tG_fill[i] = np.dot(H,G_fill[i])
			tGp_fill = cv2.projectPoints(np.asarray(tG_fill[:,0:3],dtype=float),rotation_vector,\
				translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
			for i in range(np.shape(tGp_fill)[0]):
				plt.plot([tGp_fill[i][0],tGp_fill[i-1][0]],\
				[tGp_fill[i][1],tGp_fill[i-1][1]],alpha=im_alpha,color='aquamarine')
		
		if X_active_support:
			G_support[:,0] = X_active_support
			G_support[:,1] = Y_active_support
			if Z_active_support:
				G_support[:,2] = Z_active_support
			else:
				G_support[:,2] = Z_active_default
			G_support[:,3] = np.ones((1,np.shape(X_active_support)[0]),dtype=np.float32)

		if G_support.any():
			tG_support = np.zeros((np.shape(G_support)[0],4), dtype=np.float32)
			for i in range(np.shape(G_support)[0]):
				tG_support[i] = np.dot(H,G_support[i])
			tGp_support = cv2.projectPoints(np.asarray(tG_support[:,0:3],dtype=float),rotation_vector,\
				translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
			for i in range(np.shape(tGp_support)[0]):
				plt.plot([tGp_support[i][0],tGp_support[i-1][0]],\
				[tGp_support[i][1],tGp_support[i-1][1]],alpha=im_alpha,color='yellow')
	
	plt.savefig(im_projection)
	print('\nProjection image saved.\n')

