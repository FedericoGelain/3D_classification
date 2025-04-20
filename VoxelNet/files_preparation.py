import os
import open3d as o3d
import numpy as np
import argparse
from pathlib import Path

def command_line_arguments():
    """
    This functions defines all the command line arguments that the user has or can provide
    All the arguments and the respective descriptions and requirements can be read typing on the terminal
    python dataset_cleaning.py -h
    """

    parser = argparse.ArgumentParser(description="Fix the header of the .off files, creating a new folder with the correct ones, if not done already")

    parser.add_argument("input", help="Path of the folder containing the .off files to fix (or the ones already fixed)")
    
    # store true means that if you add -v the corresponding arg will be set to true
    parser.add_argument("-f", "--fix", help="Fix the headers of the .off files", action="store_true")
    parser.add_argument("-x", "--xvoxel", help="Voxelize the triangle meshes", action="store_true")

    parser.add_argument("-o", "--output", help="Path of the folder where the fixed .off will be stored", default="Model10NetFixed")
    parser.add_argument("-v", "--voxels", help="Path of the folder where voxel grids will be stored", default="VoxelGrids")

    return parser.parse_args()



def fix_off_header(input_file, output_file):
    """
    Function that fixes the header of some of the .off files in ModelNet10 or ModelNet40
    Some headers are written as OFFnum_vertices num_faces 0, while the expected format should be
    OFF
    num_vertices num_faces 0

    :param file: string path of the .off file
    
    """
    with open(input_file, 'r') as f:
        # read all the file lines
        lines = f.readlines()

        # if the header is written as, for example OFF1672 1096 0\n
        # (the expected header is only OFF\n, so 4 characters long)
        if lines[0].startswith('OFF') and len(lines[0]) > 4:
            # split header and the model data
            header, model_data = lines[0][:3], lines[0][3:]

            # rewrite the first line correctly
            lines[0] = header+"\n"+model_data

            # just for check
            #print(lines[0])

    # rewrite the file with the correct header
    with open(output_file, 'w') as w:
            for line in lines:
                w.write(line)


if __name__ == '__main__':
    args = command_line_arguments()

    input_folder = args.input

    if (args.fix):
        output_folder = args.output

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Collect all folders names and files
        folders = [dir for dir in sorted(os.listdir(input_folder))]
        classes = {folders[i]: i for i in range(len(folders))}

        for dir in folders:
            Path(os.path.join(output_folder, dir)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(output_folder, dir, 'train')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(output_folder, dir, 'test')).mkdir(parents=True, exist_ok=True)        

            for filename in os.listdir(os.path.join(input_folder, dir, 'train')):
                if filename.endswith('.off'):
                    fix_off_header(os.path.join(input_folder, dir, 'train', filename), os.path.join(output_folder, dir, 'train', filename))

            for filename in os.listdir(os.path.join(input_folder, dir, 'test')):
                if filename.endswith('.off'):
                    fix_off_header(os.path.join(input_folder, dir, 'test', filename), os.path.join(output_folder, dir, 'test', filename))

    if (args.xvoxel):
        input_files_paths = []

        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".off"):
                    input_files_paths.append(os.path.join(root, file))

        voxels_path = args.voxels

        # Voxelize the triangle meshes (following open3D tutorial)
        for file in input_files_paths:
            mesh = o3d.io.read_triangle_mesh(file)
            
            # fit to unit cube
            mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
            #o3d.visualization.draw_geometries([mesh], width=600, height=400)
            
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.05)
            #o3d.visualization.draw_geometries([voxel_grid], width=600, height=400)

            # save the output
            # Normalize the path to handle any slashes and OS differences
            normalized_path = os.path.normpath(file)

            # Remove the extension
            path_no_ext = os.path.splitext(normalized_path)[0]

            # Split into parts
            parts = path_no_ext.split(os.sep)

            # Drop the first folder (like 'Model10NetFixed')
            result_parts = parts[1:]  # Skip the first one

            voxel_grid_folder = os.path.join("Voxels", result_parts[0], result_parts[1])
            voxel_grid_name = os.path.join("Voxels", result_parts[0], result_parts[1], result_parts[2]+".ply")
            
            Path(voxel_grid_folder).mkdir(parents=True, exist_ok=True)  

            o3d.io.write_voxel_grid(voxel_grid_name, voxel_grid, write_ascii=False, compressed=False)


