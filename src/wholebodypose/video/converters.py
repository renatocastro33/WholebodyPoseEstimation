"""
Video format conversion utilities.

REQUIRES: ffmpeg-python
Install with: pip install wholebodypose[video]
"""

import subprocess
from pathlib import Path
from typing import Tuple
import sys

# Add src/ to path
current_file = Path(__file__).resolve()
src_path = current_file.parents[2]
sys.path.insert(0, str(src_path))



def convert_mp4_to_mov(
    input_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = True
) -> Tuple[bool, str]:
    """
    Convert MP4 video to MOV format using ffmpeg.
    
    Args:
        input_path: Path to input MP4 file
        output_dir: Directory for output MOV file
        overwrite: If True, overwrites existing file
        
    Returns:
        Tuple of (success: bool, output_path: str)
    """
    from wholebodypose.utils.files import create_directory
    
    input_file = Path(input_path)
    output_folder = Path(output_dir)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_file.suffix.lower() == '.mp4':
        raise ValueError(f"Input must be MP4, got: {input_file.suffix}")
    
    create_directory(output_folder)
    output_file = output_folder / f"{input_file.stem}.mov"
    
    cmd = ["ffmpeg", "-i", str(input_file), "-f", "mov"]
    
    if overwrite:
        cmd.append("-y")
    
    cmd.append(str(output_file))
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=False
        )
        
        return result.returncode == 0, str(output_file)
        
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: pip install wholebodypose[video]"
        )


def reencode_for_web(
    input_path: str | Path,
    output_folder: str | Path,
    crf: int = 23
) -> Tuple[bool, str]:
    """
    Re-encode video with web-friendly H.264 codec.
    
    Args:
        input_path: Path to input video
        output_folder: Folder for output video
        crf: Quality parameter (lower = better, 23 is default)
        
    Returns:
        Tuple of (success: bool, output_path: str)
    """
    try:
        import ffmpeg
    except ImportError:
        raise ImportError(
            "ffmpeg-python not installed. "
            "Install with: pip install wholebodypose[video]"
        )
    
    from wholebodypose.utils.files import create_directory
    
    input_file = Path(input_path)
    output_dir = Path(output_folder)
    
    create_directory(output_dir)
    output_file = output_dir / f"{input_file.stem}_web.mp4"
    
    try:
        (
            ffmpeg
            .input(str(input_file))
            .output(
                str(output_file),
                vcodec='libx264',
                acodec='aac',
                movflags='faststart',
                preset='fast',
                crf=crf
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return True, str(output_file)
        
    except ffmpeg.Error as e:
        print(f"[ERROR] Re-encoding failed: {e}")
        return False, str(output_file)