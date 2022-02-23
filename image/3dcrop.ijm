dirsa = '/mnt/external_ssd/xinyi/2021-07-12-mAD-2766-genes-original-images/ADmouse_9494/'

run("3D OC Options", "volume surface nb_of_obj._voxels nb_of_surf._voxels integrated_density mean_gray_value std_dev_gray_value median_gray_value minimum_gray_value maximum_gray_value centroid mean_distance_to_surface std_dev_distance_to_surface median_distance_to_surface centre_of_mass bounding_box dots_size=5 font_size=20 redirect_to=none");
run("3D Manager Options", "volume surface compactness fit_ellipse 3d_moments integrated_density mean_grey_value std_dev_grey_value mode_grey_value feret minimum_grey_value maximum_grey_value centroid_(pix) centroid_(unit) distance_to_surface centre_of_mass_(pix) centre_of_mass_(unit) bounding_box radial_distance surface_contact closest use distance_between_centers=10 distance_max_contact=1.80");
open(dirsa+'pi_binary.tif')

//width=0.0946
//height=0.0946
//depth=0.3463
//a=200/(width*height*depth);// maximum volume is 1500 cu.microns
//a_1=3000/(width*height*depth);// minimum volume is 200 cu.microns
//run("3D Objects Counter", "threshold=128 slice=12 min.=a_1 max.=a objects"); //Size filter

//run 3d watershed voronoi
run("3D Manager");
Ext.Manager3D_AddImage();
Ext.Manager3D_Count(nb_obj); 
saveAs("Tiff", dirsa+'pi_segmented.tif'); 