# %% [markdown]
# Copy DEMs

# %%
# cp \
#     /data2/elilouis/mt_adams_timesift/individual_clouds/67_9/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-0.40_y-0.02_z+0.01_align.tif \
#     /data2/elilouis/hsfm-geomorph/data/mt_adams_rusk_glacier/input_dems/1967-09.tif

# cp \
#     /data2/elilouis/mt_adams_timesift/individual_clouds/87_08/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.81_y-0.53_z-0.21_align.tif \
#     /data2/elilouis/hsfm-geomorph/data/mt_adams_rusk_glacier/input_dems/1987-08.tif

# cp \
#     /data2/elilouis/mt_adams_timesift/individual_clouds/67_9/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-0.40_y-0.02_z+0.01_align.png \
#     /data2/elilouis/hsfm-geomorph/data/mt_adams_rusk_glacier/alignment_qc/1967-09.png

# cp \
#     /data2/elilouis/mt_adams_timesift/individual_clouds/87_08/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.81_y-0.53_z-0.21_align.png \
#     /data2/elilouis/hsfm-geomorph/data/mt_adams_rusk_glacier/alignment_qc/1987-08.png

# %% [markdown]
# Generate and export mosaics with final camera positions

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
project_path_87 = "/data2/elilouis/mt_adams_timesift/individual_clouds/87_08/cluster0/1/project.psx"
aligned_metadata_csv_87 = "/data2/elilouis/mt_adams_timesift/individual_clouds/87_08/cluster0/1/nuth_aligned_bundle_adj_metadata.csv"
final_dem_87 = "/data2/elilouis/mt_adams_timesift/individual_clouds/87_08/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.81_y-0.53_z-0.21_align.tif"
ortho_output_path_87 = "/data2/elilouis/hsfm-geomorph/data/mt_adams_rusk_glacier/orthos/1987-08.tif"

export_updated_orthomosaic(
    project_path_87, 
    aligned_metadata_csv_87, 
    final_dem_87,
    ortho_output_path_87
)

# %%
project_path_67 = "/data2/elilouis/mt_adams_timesift/individual_clouds/67_9/cluster0/1/project.psx"
aligned_metadata_csv_67 = "/data2/elilouis/mt_adams_timesift/individual_clouds/67_9/cluster0/1/nuth_aligned_bundle_adj_metadata.csv"
final_dem_67 = "/data2/elilouis/mt_adams_timesift/individual_clouds/67_9/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-0.40_y-0.02_z+0.01_align.tif"
ortho_output_path_67 = "/data2/elilouis/hsfm-geomorph/data/mt_adams_rusk_glacier/orthos/1967-09.tif"

export_updated_orthomosaic(
    project_path_67, 
    aligned_metadata_csv_67, 
    final_dem_67,
    ortho_output_path_67
)

# %%
