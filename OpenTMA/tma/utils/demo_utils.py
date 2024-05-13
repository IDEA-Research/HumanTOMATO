import os
from pathlib import Path


# load example data
def load_example_input(txt_path):
    """
    Parameters:
    txt_path (str): The path to the text file.

    Returns:
    texts (list): The list of text strings.
    lens (list): The list of lengths of the text strings.
    """

    file = open(txt_path, "r")
    Lines = file.readlines()
    count = 0
    texts, lens = [], []

    # Strips the newline character
    for line in Lines:
        count += 1

        # Strip the newline character from the line and split it into length and text
        s = line.strip()
        s_l = s.split(" ")[0]
        s_t = s[(len(s_l) + 1):]

        # Append the length and text to the respective lists
        lens.append(int(s_l))
        texts.append(s_t)
        print("Length-{}: {}".format(s_l, s_t))
    return texts, lens


# render batch
def render_batch(npy_dir, execute_python="./scripts/visualize_motion.sh", mode="sequence"):
    """
    Parameters:
    npy_dir (str): The directory containing the npy files.
    execute_python (str): The path to the Python script to execute. Default is "./scripts/visualize_motion.sh".
    mode (str): The mode for rendering. Default is "sequence".
    """
    # Execute the Python script with the directory and mode as arguments
    os.system(f"{execute_python} {npy_dir} {mode}")


# render
def render(execute_python, npy_path, jointtype, cfg_path):
    """
    Parameters:
    execute_python (str): The path to the Python script to execute.
    npy_path (str): The path to the npy file.
    jointtype (str): The type of joints for the skeleton.
    cfg_path (str): The path to the configuration file.

    Returns:
    fig_path (Path): The path to the rendered figure.
    """

    export_scripts = "render.py"

    os.system(
        f"{execute_python} --background --python {export_scripts} -- --cfg={cfg_path} --npy={npy_path} --joint_type={jointtype}"
    )

    # Define the path to the rendered figure and return it
    fig_path = Path(str(npy_path).replace(".npy", ".png"))
    return fig_path


# origin render
def export_fbx_hand(pkl_path):
    """
    Parameters:
    pkl_path (str): The path to the .pkl file.

    Returns:
    None
    """
    _input = pkl_path
    output = pkl_path.replace(".pkl", ".fbx")

    execute_python = "/apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender"
    export_scripts = "./scripts/fbx_output_smplx.py"
    os.system(
        f"{execute_python} -noaudio --background --python {export_scripts}\
                --input {_input} \
                --output {output}"
    )


# export fbx without hand params from pkl files
def export_fbx(pkl_path):
    """
    Parameters:
    pkl_path (str): The path to the .pkl file.

    Returns:
    None
    """
    _input = pkl_path
    output = pkl_path.replace(".pkl", ".fbx")

    execute_python = "/apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender"
    export_scripts = "./scripts/fbx_output.py"
    os.system(
        f"{execute_python} -noaudio --background --python {export_scripts}\
                --input {_input} \
                --output {output}"
    )
