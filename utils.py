import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional

class ImageUtils:
    """Utility functions for image processing and conversion."""
    
    # Supported image formats
    SUPPORTED_FORMATS = {
        'image/jpeg': ['.jpg', '.jpeg'],
        'image/png': ['.png'],
        'image/gif': ['.gif'],
        'image/bmp': ['.bmp'],
        'image/tiff': ['.tiff', '.tif'],
        'image/webp': ['.webp']
    }
    
    @staticmethod
    def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
        """Load image from Streamlit uploaded file."""
        try:
            image = Image.open(uploaded_file)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            return None
    
    @staticmethod
    def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to NumPy array."""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return np.array(pil_image)
    
    @staticmethod
    def numpy_to_pil(numpy_image: np.ndarray) -> Image.Image:
        """Convert NumPy array to PIL Image."""
        # Ensure the array is in the correct format
        if numpy_image.dtype != np.uint8:
            numpy_image = numpy_image.astype(np.uint8)
        
        # Handle grayscale images
        if len(numpy_image.shape) == 2:
            return Image.fromarray(numpy_image, mode='L')
        else:
            return Image.fromarray(numpy_image, mode='RGB')
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        max_w, max_h = max_size
        
        # Calculate scaling factor
        scale = min(max_w / w, max_h / h, 1.0)
        
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            pil_image = ImageUtils.numpy_to_pil(image)
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return ImageUtils.pil_to_numpy(pil_image)
        
        return image
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """Get basic information about the image."""
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            'width': w,
            'height': h,
            'channels': channels,
            'size': f"{w} x {h}",
            'format': 'RGB' if channels == 3 else 'Grayscale'
        }
    
    @staticmethod
    def create_download_link(image: np.ndarray, filename: str = "filtered_image.png") -> bytes:
        """Create downloadable image data."""
        pil_image = ImageUtils.numpy_to_pil(image)
        
        # Create bytes buffer
        img_buffer = io.BytesIO()
        
        # Determine format from filename extension
        if filename.lower().endswith(('.jpg', '.jpeg')):
            pil_image.save(img_buffer, format='JPEG', quality=90)
        elif filename.lower().endswith('.png'):
            pil_image.save(img_buffer, format='PNG')
        elif filename.lower().endswith('.bmp'):
            pil_image.save(img_buffer, format='BMP')
        elif filename.lower().endswith(('.tiff', '.tif')):
            pil_image.save(img_buffer, format='TIFF')
        elif filename.lower().endswith('.webp'):
            pil_image.save(img_buffer, format='WEBP')
        else:
            # Default to PNG
            pil_image.save(img_buffer, format='PNG')
        
        return img_buffer.getvalue()
    
    @staticmethod
    def validate_image_format(uploaded_file) -> bool:
        """Validate if uploaded file is a supported image format."""
        if uploaded_file is None:
            return False
        
        file_type = uploaded_file.type
        return file_type in ImageUtils.SUPPORTED_FORMATS
    
    @staticmethod
    def get_supported_extensions() -> list:
        """Get list of all supported file extensions."""
        extensions = []
        for ext_list in ImageUtils.SUPPORTED_FORMATS.values():
            extensions.extend(ext_list)
        return sorted(extensions)