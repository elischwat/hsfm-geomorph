#Copy source DEMs
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/67_9/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-66.12_y+10.97_z-12.84_align.tif \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/input_dems/1967-09.tif
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/70_09/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.18_y+1.20_z+0.81_align.tif \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/input_dems/1970-09.tif
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/77_09/cluster4/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-59.20_y+7.22_z-13.20_align.tif \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/input_dems/1977-09.tif
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/79_10/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.82_y+1.43_z+1.33_align.tif \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/input_dems/1979-10.tif
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/87_08/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-8.80_y+3.23_z+0.00_align.tif \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/input_dems/1987-08.tif
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/90_09/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-8.41_y+3.44_z+0.72_align.tif \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/input_dems/1990-09.tif
cp \
    /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/input_dems/2015-00.tif


#Copy associated *align.png files 

#
# %%
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/67_9/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/*align.png \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/alignment_qc/1967-09-align.png
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/70_09/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/*align.png \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/alignment_qc/1970-09-align.png
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/77_09/cluster4/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/*align.png \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/alignment_qc/1977-09-align.png
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/79_10/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/*align.png \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/alignment_qc/1979-10-align.png
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/87_08/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/*align.png \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/alignment_qc/1987-08-align.png
cp \
    /data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/90_09/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/*align.png \
    /data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/alignment_qc/1990-09-align.png

final_dems = [
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/67_9/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-66.12_y+10.97_z-12.84_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/70_09/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.18_y+1.20_z+0.81_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/77_09/cluster4/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-59.20_y+7.22_z-13.20_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/79_10/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.82_y+1.43_z+1.33_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/87_08/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-8.80_y+3.23_z+0.00_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/90_09/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-8.41_y+3.44_z+0.72_align.tif"
]
# %%
import hsfm
import Metashape

# %%
hsfm.metashape.authentication("/home/elilouis/hsfm/uw_agisoft.lic")
print(Metashape.app.activated)

# %%
def export_updated_orthomosaic(project_path, metadata_csv_path, dem_path, ortho_output_path):
    """
    Takes a Metashape project file, a CSV containing camera metadata, and a DEM tif file, and outputs an orthomosaic.
    This is useful for creating an orthomosaic that is in alignment with a DEM that has been aligned using tools
    outside of Metashape (such as pc_align or knuth and kaab).
    
    Params:
    project_path (str): path to the Metashape project 
    metadata_csv_path (str): path to the CSV containing camera information, camera extrinsics will be used to create orthomosaic
    dem_path (str): path to the DEM tif file that will be used to create orthomosaic
    ortho_output_path (str): path to create the orthomosaic
    """
    doc = Metashape.Document()
    doc.open(project_path)
    doc.read_only=False
    chunk = doc.chunk
    
    chunk.importReference(metadata_csv_path,
                          columns="nxyzXYZabcABC", # from metashape py api docs
                          delimiter=',',
                          format=Metashape.ReferenceFormatCSV)
    T = chunk.transform.matrix
    for camera in chunk.cameras:
        image = camera.label
        lon, lat, alt = chunk.crs.project(T.mulp(camera.center))
        m = chunk.crs.localframe(T.mulp(camera.center)) #transformation matrix to the LSE coordinates in the given point
        R = m * T * camera.transform * Metashape.Matrix().Diag([1, -1, -1, 1])
        row = list()
        for j in range (0, 3): #creating normalized rotation matrix 3x3
            row.append(R.row(j))
            row[j].size = 3
            row[j].normalize()
        R = Metashape.Matrix([row[0], row[1], row[2]])
        yaw, pitch, roll = Metashape.utils.mat2ypr(R) #estimated orientation angles
    chunk.updateTransform()
    chunk.dense_cloud.crs = chunk.crs
    chunk.dense_cloud.transform = chunk.transform.matrix
    chunk.importRaster(dem_path)
    chunk.calibrateColors(source_data=Metashape.ElevationData)
    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
    chunk.exportRaster(ortho_output_path,
                       source_data= Metashape.OrthomosaicData)
    return ortho_output_path

# %%
project_paths = [
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/67_9/cluster0/1/project.psx",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/70_09/cluster0/1/project.psx",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/77_09/cluster4/1/project.psx",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/79_10/cluster0/1/project.psx",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/87_08/cluster1/1/project.psx",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/90_09/cluster1/1/project.psx"
]
aligned_metadata_csv_paths = [
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/67_9/cluster0/1/nuth_aligned_bundle_adj_metadata.csv",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/70_09/cluster0/1/nuth_aligned_bundle_adj_metadata.csv",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/77_09/cluster4/1/nuth_aligned_bundle_adj_metadata.csv",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/79_10/cluster0/1/nuth_aligned_bundle_adj_metadata.csv",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/87_08/cluster1/1/nuth_aligned_bundle_adj_metadata.csv",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/90_09/cluster1/1/nuth_aligned_bundle_adj_metadata.csv"
]
final_dems = [
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/67_9/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-66.12_y+10.97_z-12.84_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/70_09/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.18_y+1.20_z+0.81_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/77_09/cluster4/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-59.20_y+7.22_z-13.20_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/79_10/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.82_y+1.43_z+1.33_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/87_08/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-8.80_y+3.23_z+0.00_align.tif",
    "/data2/elilouis/mt_baker_timesift_cam_calib/individual_clouds/90_09/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-8.41_y+3.44_z+0.72_align.tif"
]
ortho_output_paths = [
    "/data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/orthos/1967-09.tif",
    "/data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/orthos/1970-09.tif",
    "/data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/orthos/1977-09.tif",
    "/data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/orthos/1979-10.tif",
    "/data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/orthos/1987-08.tif",
    "/data2/elilouis/hsfm-geomorph/data/mt_baker_coleman_glacier/orthos/1990-09.tif"
]

# %%
for project, aligned_metadata_csv, final_dem, ortho_output_path in zip(
    project_paths, aligned_metadata_csv_paths, final_dems, ortho_output_paths
):
    export_updated_orthomosaic(
    project, 
    aligned_metadata_csv, 
    final_dem,
    ortho_output_path
)


