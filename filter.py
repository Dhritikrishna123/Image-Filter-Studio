import numpy as np
from typing import Tuple

class ImageFilters:
    """Collection of image filters using only NumPy operations."""
    
    @staticmethod
    def grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale using luminance weights."""
        if len(image.shape) == 3:
            weights = np.array([0.299, 0.587, 0.114])
            grayscale = np.dot(image, weights)
            return np.stack([grayscale] * 3, axis=2).astype(np.uint8)
        return image
    
    @staticmethod
    def sepia(image: np.ndarray) -> np.ndarray:
        """Apply sepia tone effect."""
        if len(image.shape) == 3:
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            sepia_img = np.dot(image, sepia_matrix.T)
            sepia_img = np.clip(sepia_img, 0, 255)
            return sepia_img.astype(np.uint8)
        return image
    
    @staticmethod
    def brightness(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Adjust image brightness."""
        bright_img = image.astype(np.float32) * factor
        return np.clip(bright_img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def contrast(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Adjust image contrast."""
        img_float = image.astype(np.float32) - 128
        contrast_img = img_float * factor + 128
        return np.clip(contrast_img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def invert(image: np.ndarray) -> np.ndarray:
        """Invert image colors."""
        return 255 - image
    
    @staticmethod
    def blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply simple box blur filter."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        pad_size = kernel_size // 2
        if len(image.shape) == 3:
            h, w, c = image.shape
            padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
            blurred = np.zeros_like(image, dtype=np.float32)
            
            for i in range(h):
                for j in range(w):
                    for ch in range(c):
                        window = padded[i:i+kernel_size, j:j+kernel_size, ch]
                        blurred[i, j, ch] = np.mean(window)
        else:
            h, w = image.shape
            padded = np.pad(image, pad_size, mode='edge')
            blurred = np.zeros_like(image, dtype=np.float32)
            
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    blurred[i, j] = np.mean(window)
        
        return blurred.astype(np.uint8)
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur filter."""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        
        # Apply convolution
        return ImageFilters._apply_kernel(image, kernel)
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter using convolution."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        return ImageFilters._apply_kernel(image, kernel)
    
    @staticmethod
    def edge_detection(image: np.ndarray) -> np.ndarray:
        """Apply edge detection using Sobel operator."""
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = ImageFilters.grayscale(image)[:, :, 0]
        else:
            gray = image
        
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
        
        edges_x = ImageFilters._apply_kernel_single(gray, sobel_x)
        edges_y = ImageFilters._apply_kernel_single(gray, sobel_y)
        
        # Combine gradients
        edges = np.sqrt(edges_x**2 + edges_y**2)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        
        # Convert back to 3-channel for consistency
        return np.stack([edges] * 3, axis=2)
    
    @staticmethod
    def vintage(image: np.ndarray) -> np.ndarray:
        """Apply vintage/retro effect."""
        # Apply sepia first
        vintage_img = ImageFilters.sepia(image)
        
        # Add some noise and reduce contrast slightly
        noise = np.random.normal(0, 10, vintage_img.shape)
        vintage_img = vintage_img.astype(np.float32) + noise
        
        # Reduce contrast slightly
        vintage_img = ImageFilters.contrast(vintage_img.astype(np.uint8), 0.8)
        
        # Add vignette effect (darken edges)
        h, w = vintage_img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        vignette = 1 - (distance / max_distance) * 0.3
        vignette = np.clip(vignette, 0.4, 1)
        
        if len(vintage_img.shape) == 3:
            vignette = np.stack([vignette] * 3, axis=2)
        
        vintage_img = vintage_img.astype(np.float32) * vignette
        return np.clip(vintage_img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def emboss(image: np.ndarray) -> np.ndarray:
        """Apply emboss effect."""
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [0,   1, 2]])
        
        embossed = ImageFilters._apply_kernel(image, kernel)
        # Add 128 to make it visible and convert to grayscale-like
        embossed = np.clip(embossed.astype(np.float32) + 128, 0, 255)
        return embossed.astype(np.uint8)
    
    @staticmethod
    def outline(image: np.ndarray) -> np.ndarray:
        """Apply outline/edge enhancement filter."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        
        return ImageFilters._apply_kernel(image, kernel)
    
    @staticmethod
    def posterize(image: np.ndarray, levels: int = 4) -> np.ndarray:
        """Reduce the number of colors (posterization effect)."""
        if levels < 2:
            levels = 2
        if levels > 256:
            levels = 256
        
        factor = 255 // (levels - 1)
        posterized = (image // factor) * factor
        return posterized.astype(np.uint8)
    
    @staticmethod
    def solarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
        """Apply solarization effect."""
        solarized = np.where(image < threshold, image, 255 - image)
        return solarized.astype(np.uint8)
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Apply gamma correction."""
        gamma_corrected = 255 * ((image / 255) ** (1 / gamma))
        return np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    
    @staticmethod
    def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
        """Apply logarithmic transformation."""
        log_transformed = c * np.log(1 + image.astype(np.float32))
        log_transformed = (log_transformed / np.max(log_transformed)) * 255
        return log_transformed.astype(np.uint8)
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to enhance contrast."""
        if len(image.shape) == 3:
            # Apply to each channel separately
            equalized = np.zeros_like(image)
            for i in range(3):
                equalized[:, :, i] = ImageFilters._equalize_channel(image[:, :, i])
            return equalized
        else:
            return ImageFilters._equalize_channel(image)
    
    @staticmethod
    def _equalize_channel(channel: np.ndarray) -> np.ndarray:
        """Equalize histogram for a single channel."""
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        equalized = np.interp(channel.flatten(), bins[:-1], cdf_normalized)
        return equalized.reshape(channel.shape).astype(np.uint8)
    
    @staticmethod
    def color_balance(image: np.ndarray, red_factor: float = 1.0, green_factor: float = 1.0, blue_factor: float = 1.0) -> np.ndarray:
        """Apply color balance adjustment."""
        if len(image.shape) == 3:
            balanced = image.astype(np.float32)
            balanced[:, :, 0] *= red_factor    # Red channel
            balanced[:, :, 1] *= green_factor  # Green channel
            balanced[:, :, 2] *= blue_factor   # Blue channel
            return np.clip(balanced, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def channel_mixer(image: np.ndarray, red_mix: Tuple[float, float, float] = (1, 0, 0), 
                     green_mix: Tuple[float, float, float] = (0, 1, 0), 
                     blue_mix: Tuple[float, float, float] = (0, 0, 1)) -> np.ndarray:
        """Mix color channels with custom weights."""
        if len(image.shape) == 3:
            mixed = np.zeros_like(image, dtype=np.float32)
            
            # Mix each output channel
            mixed[:, :, 0] = (image[:, :, 0] * red_mix[0] + 
                             image[:, :, 1] * red_mix[1] + 
                             image[:, :, 2] * red_mix[2])
            
            mixed[:, :, 1] = (image[:, :, 0] * green_mix[0] + 
                             image[:, :, 1] * green_mix[1] + 
                             image[:, :, 2] * green_mix[2])
            
            mixed[:, :, 2] = (image[:, :, 0] * blue_mix[0] + 
                             image[:, :, 1] * blue_mix[1] + 
                             image[:, :, 2] * blue_mix[2])
            
            return np.clip(mixed, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def duotone(image: np.ndarray, color1: Tuple[int, int, int] = (255, 0, 0), 
               color2: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Create duotone effect with two colors."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image
        
        # Normalize grayscale to 0-1
        gray_norm = gray / 255.0
        
        # Create duotone
        duotone = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)
        
        for i in range(3):
            duotone[:, :, i] = color1[i] * (1 - gray_norm) + color2[i] * gray_norm
        
        return np.clip(duotone, 0, 255).astype(np.uint8)
    
    @staticmethod
    def vignette(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply vignette effect."""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        vignette_mask = 1 - (distance / max_distance) * intensity
        vignette_mask = np.clip(vignette_mask, 0, 1)
        
        if len(image.shape) == 3:
            vignette_mask = np.stack([vignette_mask] * 3, axis=2)
        
        vignetted = image.astype(np.float32) * vignette_mask
        return np.clip(vignetted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def noise(image: np.ndarray, intensity: float = 20.0) -> np.ndarray:
        """Add random noise to image."""
        noise_array = np.random.normal(0, intensity, image.shape)
        noisy = image.astype(np.float32) + noise_array
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def oil_painting(image: np.ndarray, radius: int = 3, intensity: int = 20) -> np.ndarray:
        """Apply oil painting effect."""
        h, w = image.shape[:2]
        oil_img = np.zeros_like(image)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                if len(image.shape) == 3:
                    for c in range(3):
                        region = image[i-radius:i+radius+1, j-radius:j+radius+1, c]
                        hist, bins = np.histogram(region, bins=intensity, range=(0, 256))
                        max_bin = np.argmax(hist)
                        oil_img[i, j, c] = bins[max_bin]
                else:
                    region = image[i-radius:i+radius+1, j-radius:j+radius+1]
                    hist, bins = np.histogram(region, bins=intensity, range=(0, 256))
                    max_bin = np.argmax(hist)
                    oil_img[i, j] = bins[max_bin]
        
        return oil_img.astype(np.uint8)
    
    @staticmethod
    def pixelate(image: np.ndarray, pixel_size: int = 10) -> np.ndarray:
        """Apply pixelation effect."""
        h, w = image.shape[:2]
        
        # Downsample
        small_h, small_w = h // pixel_size, w // pixel_size
        if len(image.shape) == 3:
            small_img = np.zeros((small_h, small_w, 3), dtype=np.uint8)
            for i in range(small_h):
                for j in range(small_w):
                    y_start, y_end = i * pixel_size, (i + 1) * pixel_size
                    x_start, x_end = j * pixel_size, (j + 1) * pixel_size
                    small_img[i, j] = np.mean(image[y_start:y_end, x_start:x_end], axis=(0, 1))
        else:
            small_img = np.zeros((small_h, small_w), dtype=np.uint8)
            for i in range(small_h):
                for j in range(small_w):
                    y_start, y_end = i * pixel_size, (i + 1) * pixel_size
                    x_start, x_end = j * pixel_size, (j + 1) * pixel_size
                    small_img[i, j] = np.mean(image[y_start:y_end, x_start:x_end])
        
        # Upsample back
        pixelated = np.repeat(np.repeat(small_img, pixel_size, axis=0), pixel_size, axis=1)
        
        # Crop to original size
        return pixelated[:h, :w]
    
    @staticmethod
    def thermal(image: np.ndarray) -> np.ndarray:
        """Apply thermal/heat vision effect."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.dot(image, [0.299, 0.587, 0.114])
        else:
            gray = image
        
        # Create thermal colormap
        thermal_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        
        # Map grayscale to thermal colors (blue -> purple -> red -> yellow -> white)
        normalized = gray / 255.0
        
        thermal_img[:, :, 0] = np.clip(255 * (normalized * 3 - 1), 0, 255)  # Red
        thermal_img[:, :, 1] = np.clip(255 * (normalized * 2 - 0.5), 0, 255)  # Green
        thermal_img[:, :, 2] = np.clip(255 * (1 - normalized * 2), 0, 255)  # Blue
        
        return thermal_img
    
    @staticmethod
    def cold(image: np.ndarray) -> np.ndarray:
        """Apply cold tone effect."""
        cold_matrix = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.3, 1.4]
        ])
        
        if len(image.shape) == 3:
            cold_img = np.dot(image, cold_matrix.T)
            return np.clip(cold_img, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def warm(image: np.ndarray) -> np.ndarray:
        """Apply warm tone effect."""
        warm_matrix = np.array([
            [1.4, 0.3, 0.1],
            [0.3, 1.0, 0.1],
            [0.1, 0.2, 0.6]
        ])
        
        if len(image.shape) == 3:
            warm_img = np.dot(image, warm_matrix.T)
            return np.clip(warm_img, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def cross_process(image: np.ndarray) -> np.ndarray:
        """Apply cross-processing effect."""
        # Enhance contrast and shift colors
        processed = image.astype(np.float32)
        
        if len(image.shape) == 3:
            # Apply curves to each channel
            processed[:, :, 0] = 255 * ((processed[:, :, 0] / 255) ** 0.8)  # Red curve
            processed[:, :, 1] = 255 * ((processed[:, :, 1] / 255) ** 1.2)  # Green curve
            processed[:, :, 2] = 255 * ((processed[:, :, 2] / 255) ** 0.9)  # Blue curve
        
        return np.clip(processed, 0, 255).astype(np.uint8)
    
    @staticmethod
    def dreamy(image: np.ndarray) -> np.ndarray:
        """Apply dreamy/soft focus effect."""
        # Apply gaussian blur and blend with original
        blurred = ImageFilters.gaussian_blur(image, sigma=2.0)
        
        # Blend original with blurred (soft light effect)
        dreamy = 0.7 * image.astype(np.float32) + 0.3 * blurred.astype(np.float32)
        
        # Slightly brighten
        dreamy *= 1.1
        
        return np.clip(dreamy, 0, 255).astype(np.uint8)
    
    # Helper methods
    @staticmethod
    def _apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply convolution kernel to image."""
        kernel_size = kernel.shape[0]
        pad_size = kernel_size // 2
        
        if len(image.shape) == 3:
            h, w, c = image.shape
            padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
            result = np.zeros_like(image, dtype=np.float32)
            
            for i in range(h):
                for j in range(w):
                    for ch in range(c):
                        window = padded[i:i+kernel_size, j:j+kernel_size, ch]
                        result[i, j, ch] = np.sum(window * kernel)
        else:
            h, w = image.shape
            padded = np.pad(image, pad_size, mode='edge')
            result = np.zeros_like(image, dtype=np.float32)
            
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+kernel_size, j:j+kernel_size]
                    result[i, j] = np.sum(window * kernel)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _apply_kernel_single(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply convolution kernel to single channel image."""
        kernel_size = kernel.shape[0]
        pad_size = kernel_size // 2
        h, w = image.shape
        
        padded = np.pad(image, pad_size, mode='edge')
        result = np.zeros_like(image, dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.sum(window * kernel)
        
        return result