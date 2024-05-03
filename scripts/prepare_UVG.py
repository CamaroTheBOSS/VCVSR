import os
import subprocess
import zipfile
import py7zr


def relevant(filename, unzip=False):
    keywords = ("Beauty_", "Bosphorus_", "HoneyBee_", "Jockey_", "ReadySetGo_" if unzip else "ReadySteadyGo_",
                "ShakeNDry_", "YachtRide_")
    return filename.startswith(keywords) and "1920x1080" in filename


def unzip_uvg(src_path: str):
    """
    This function search for UVG .7z files and unzip them to YUV files. YUV files are saved in the same location where
    they were foundNote that YUV files are 2x more heavy in memory that .7z.
    :param src_path: Path where to find UVG files
    :return: None
    """
    files = list(filter(lambda x: relevant(x, unzip=True) and x.endswith(".7z"), os.listdir(src_path)))
    for file in files:
        print(f"Processing {file}...")
        filepath = os.path.join(src_path, file)
        archive = py7zr.SevenZipFile(filepath, mode='r')
        archive.extractall(src_path)
        archive.close()


def prepare_uvg(src_path: str, dst_path: str):
    """
    Function, which search for UVG YUV files and transform them into iframes. Note that if you just downloaded videos
    from UVG website, you need to unzip them first to use this function
    :param src_path: root path where to search for UVG YUV videos
    :param dst_path: root path where to save UVG videos
    :return: None
    """
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    files = list(filter(lambda x: relevant(x) and x.endswith(".yuv"), os.listdir(src_path)))
    for file in files:
        name = file.split("_")[0]
        savepath = os.path.join(dst_path, name)
        filepath = os.path.join(src_path, file)
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        print(f"Processing {file}...")
        command = ["ffmpeg",
                   "-y",
                   "-pix_fmt", "yuv420p",
                   "-s", "1920x1080",
                   "-i", filepath,
                   f"{savepath}/%03d.png"]
        subprocess.run(command)


if __name__ == "__main__":
    # unzip_uvg(r"C:\Users\CamaroTheBOSS\Downloads")
    prepare_uvg(r"C:\Users\CamaroTheBOSS\Downloads", r"..\..\Datasets\UVG")
