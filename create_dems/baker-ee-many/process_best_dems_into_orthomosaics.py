import os
from hsfm.pipeline import TimesiftPipeline

# Required inputs:
main_directory = '/data2/elilouis/timesift/baker-ee-many/'
combined_metadata_file_path = '/data2/elilouis/timesift/baker-ee-many/combined_metashape_metadata.csv'
combined_image_metadata_file_path = '/data2/elilouis/timesift/baker-ee-many/combined_image_metadata.csv'
preprocessed_images_dir = '/data2/elilouis/timesift/baker-ee-many/preprocessed_images'
timesift_output_path = '/data2/elilouis/timesift/baker-ee-many/mixed_timesift'
reference_dem_lowres = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015_10m.tif'
reference_dem_hires = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif'
license_path = "/home/elilouis/hsfm/uw_agisoft.lic"

# Create pipeline input
timesift_pipeline = TimesiftPipeline(
    metashape_metadata_file = combined_metadata_file_path,
    image_metadata_file = combined_image_metadata_file_path,
    raw_images_directory = preprocessed_images_dir,
    output_directory = timesift_output_path,
    reference_dem_lowres = reference_dem_lowres,
    reference_dem_hires = reference_dem_hires,
    image_matching_accuracy = 1,
    densecloud_quality = 2,
    output_DEM_resolution = 5,
    license_path=license_path,
    parallelization=1
)

best_results_dict = {
	#maybe delete this first one:
	"/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/47_9.0_14.0/cluster0" : "spoint2point_bareground",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/67_9.0_21.0/cluster2" : "spoint2point_bareground",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/70_9.0_29.0/cluster0" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/70_9.0_9.0/cluster0"  : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/70_9.0_9.0/cluster1"  : "spoint2point_bareground",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/74_8.0_10.0/cluster3" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/74_8.0_10.0/cluster4" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/74_8.0_10.0/cluster5" : "spoint2point_bareground",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/74_8.0_10.0/cluster6" : "spoint2point_bareground",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/77_9.0_27.0/cluster0" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/77_9.0_27.0/cluster2" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/77_9.0_27.0/cluster3" : "spoint2point_bareground",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/77_9.0_27.0/cluster4" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/77_9.0_27.0/cluster5" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/77_9.0_27.0/cluster6" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/77_9.0_27.0/cluster7" : "point2plane",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/79_10.0_6.0/cluster0" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/87_8.0_21.0/cluster0" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/87_8.0_21.0/cluster2" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/87_8.0_21.0/cluster3" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/90_9.0_5.0/cluster1"  : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/90_9.0_5.0/cluster2"  : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/91_9.0_9.0/cluster0"  : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/91_9.0_9.0/cluster1"  : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/91_9.0_9.0/cluster3"  : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/92_9.0_15.0/cluster2" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/92_9.0_15.0/cluster3" : "spoint2point",
	# "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/92_9.0_18.0/cluster1" : "spoint2point"
}
timesift_pipeline.process_final_orthomosaics(best_results_dict)
timesift_pipeline.process_selected_dems_into_mosaics(best_results_dict)