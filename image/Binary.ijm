/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////  SCRIPT TO IDENTIFY SEGMENT NUCLEI IN 3D AND MEASURE THE INTENSITY IN CHANNEL 1 AND 2         											                    /////////////
///////  WRITTEN BY: SARADHA VENKATACHALAPATHY                                                                                                                   /////////////
///////  ASSUMPTIONS: The input image is a confocal zstack and the channel 1 contains the nucleus 
///////  DESCRIPTION: Two dialog box opens where the user inputs the source (where the folder containing images are strored). The program creates 2 subfolders where
///////               cropped 3D nuclei are stored. The program opens nucleus channel, smoothens the stack and the thresholds the image and identifies objects in 3D. 
///////               The program then opens the raw image and crops the image in the all channels individually. 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


dirsa = '/mnt/external_ssd/xinyi/2021-07-12-mAD-2766-genes-original-images/ADmouse_9494/'


//dirw= dirsa + "after watershed";
//dirraw= dirsa + "rawimages";
//File.makeDirectory(dirw); 
//File.makeDirectory(dirraw); 
run("3D OC Options", "volume surface nb_of_obj._voxels nb_of_surf._voxels integrated_density mean_gray_value std_dev_gray_value median_gray_value minimum_gray_value maximum_gray_value centroid mean_distance_to_surface std_dev_distance_to_surface median_distance_to_surface centre_of_mass bounding_box dots_size=5 font_size=20 redirect_to=none");
run("3D Manager Options", "volume surface compactness fit_ellipse 3d_moments integrated_density mean_grey_value std_dev_grey_value mode_grey_value feret minimum_grey_value maximum_grey_value centroid_(pix) centroid_(unit) distance_to_surface centre_of_mass_(pix) centre_of_mass_(unit) bounding_box radial_distance surface_contact closest use distance_between_centers=10 distance_max_contact=1.80");

path=dirsa+'pi.tif';
run("Bio-Formats", "open=path color_mode=Grayscale specify_range view=Hyperstack stack_order=XYCZT c_begin=1 c_end=1 c_step=1");
print("opened");
	
run("Gaussian Blur...", "sigma=1 stack");
setAutoThreshold("Otsu dark stack");
//setSlice(nSlices/2);
//waitForUser("Please Check Threshold");
run("Make Binary", "method=Default background=Default");
//run("Invert LUT");
run("Fill Holes", "stack");
run("Erode", "stack");
run("Dilate", "stack");
//run("Watershed", "stack");
saveAs("Tiff", dirsa+'pi_binary.tif'); 
//run("Close All");

run("Collect Garbage");
	run("Collect Garbage");
	run("Collect Garbage");
		


