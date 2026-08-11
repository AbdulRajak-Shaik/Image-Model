import os
from PIL import Image
from PIL.ExifTags import TAGS

class MetadataExtractor:
    def __init__(self, exiftool_path=None):
        self.exiftool_path = exiftool_path
        self._exiftool_available = self._check_exiftool()

    def _check_exiftool(self):
        """Check if exiftool is available on the system path."""
        import shutil
        return shutil.which("exiftool") is not None

    def extract_metadata(self, file_path):
        """
        Extracts metadata from the file.
        Routes video files to cv2-based extraction; images to ExifTool/Pillow.
        """
        if self._is_video(file_path):
            return self._extract_video_metadata(file_path)
        if self._exiftool_available:
            return self._extract_with_exiftool(file_path)
        return self._extract_with_pillow(file_path)

    def _is_video(self, file_path):
        """Check if file is a video by extension."""
        return os.path.splitext(file_path)[1].lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def _extract_video_metadata(self, file_path):
        """Extract video container metadata using OpenCV."""
        result = {}
        try:
            import cv2
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return {'error': f'Cannot open video file: {os.path.basename(file_path)}'}

            fps        = cap.get(cv2.CAP_PROP_FPS)
            width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration   = (frame_count / fps) if fps > 0 else 0
            codec_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec      = ''.join([chr((codec_int >> (8 * i)) & 0xFF) for i in range(4)])
            cap.release()

            file_size  = os.path.getsize(file_path)
            result = {
                'File:FileName':     os.path.basename(file_path),
                'File:FileSize':     f'{file_size / (1024*1024):.2f} MB',
                'File:FileType':     os.path.splitext(file_path)[1].upper().lstrip('.'),
                'Video:ImageWidth':  width,
                'Video:ImageHeight': height,
                'Video:FrameRate':   f'{fps:.2f} fps',
                'Video:FrameCount':  frame_count,
                'Video:Duration':    f'{duration:.2f} seconds',
                'Video:Codec':       codec.strip(),
            }
        except Exception as e:
            result = {'error': f'Video metadata extraction failed: {e}'}
        return result

    def _extract_with_exiftool(self, file_path):
        """ExifTool-based extraction (rich metadata)."""
        try:
            import exiftool
            with exiftool.ExifToolHelper() as et:
                metadata = et.get_metadata(file_path)
                if isinstance(metadata, list) and len(metadata) > 0:
                    return metadata[0]
                return metadata
        except Exception as e:
            return {'error': str(e)}

    def _extract_with_pillow(self, file_path):
        """
        Pure-Python fallback using Pillow for image files.
        """
        result = {}
        try:
            img = Image.open(file_path)
            result['File:FileName']  = os.path.basename(file_path)
            result['File:FileSize']  = f"{os.path.getsize(file_path)} bytes"
            result['Image:Format']   = img.format or "Unknown"
            result['Image:Mode']     = img.mode
            result['Image:Width']    = img.width
            result['Image:Height']   = img.height

            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        continue
                    result[f'EXIF:{tag_name}'] = str(value)
            else:
                result['Info'] = 'No EXIF data embedded in this image.'
        except Exception as e:
            result['error'] = str(e)
        return result

    def analyze_metadata(self, metadata):
        """
        Analyzes metadata to flag suspicious modifications.
        Returns a dict with integrity score and flags.
        """
        flags = []
        score = 100

        if not metadata or 'error' in metadata:
            error_msg = metadata.get('error', 'Unknown error') if metadata else 'Empty metadata'
            return {
                'score': 50,
                'flags': [f'Metadata reading failed: {error_msg}'],
                'raw': metadata
            }

        # Detect if this is video metadata
        is_video = any(k.startswith('Video:') for k in metadata)

        if is_video:
            # Video-specific analysis
            codec = metadata.get('Video:Codec', '').strip()
            fps   = metadata.get('Video:FrameRate', '')
            dur   = metadata.get('Video:Duration', '')
            flags.append(f"Video container: {metadata.get('File:FileType','MP4')} | "
                         f"Codec: {codec or 'Unknown'} | FPS: {fps} | Duration: {dur}")
            if codec and codec.upper() in ['H264', 'AVC1', 'X264']:
                flags.append("H.264 codec detected — widely used in deepfake generation pipelines.")
                score -= 10
            flags.append(f"File size: {metadata.get('File:FileSize','Unknown')} | "
                         f"Resolution: {metadata.get('Video:ImageWidth','?')}x{metadata.get('Video:ImageHeight','?')}")
            if not flags:
                flags.append("No suspicious video metadata found.")
        else:
            # Image-specific analysis
            suspicious_software = ['photoshop', 'gimp', 'lightroom', 'after effects', 'premiere', 'affinity']
            software = str(metadata.get('Software', metadata.get('EXIF:Software', ''))).lower()
            if software and software not in ('', 'none', 'unknown'):
                for sw in suspicious_software:
                    if sw in software:
                        flags.append(f"Image was edited using detected software: '{software}'")
                        score -= 40
                        break

            dt_orig = metadata.get('EXIF:DateTimeOriginal', metadata.get('DateTimeOriginal'))
            dt_mod  = metadata.get('EXIF:ModifyDate',       metadata.get('ModifyDate'))
            if dt_orig and dt_mod and str(dt_orig) != str(dt_mod):
                flags.append("Modification date differs from original creation date.")
                score -= 20

            has_camera = (
                'EXIF:Make'  in metadata or 'Make'  in metadata or
                'EXIF:Model' in metadata or 'Model' in metadata
            )
            has_exif = any(k.startswith('EXIF:') for k in metadata)
            if not has_camera and not has_exif:
                flags.append("Missing camera Make/Model. Likely AI-generated or metadata was scrubbed.")
                score -= 20
            elif not has_camera:
                flags.append("Camera Make/Model not found — possible synthetic or edited image.")
                score -= 10

            if not flags:
                flags.append("No suspicious metadata found. Image metadata appears intact.")

        return {
            'score': max(0, score),
            'flags': flags,
            'raw': metadata
        }
