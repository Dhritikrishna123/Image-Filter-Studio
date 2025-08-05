import streamlit as st
import numpy as np
from filters import ImageFilters
from utils import ImageUtils
import time

def main():
    st.set_page_config(
        page_title="Image Filter Studio",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎨 Image Filter Studio")
    st.markdown("Upload an image and apply various filters using pure NumPy operations!")
    
    # Sidebar for filter controls
    with st.sidebar:
        st.header("Filter Controls")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=ImageUtils.get_supported_extensions(),
            help=f"Supported formats: {', '.join(ImageUtils.get_supported_extensions())}"
        )
        
        if uploaded_file is not None:
            # Validate file format
            if not ImageUtils.validate_image_format(uploaded_file):
                st.error("Unsupported file format!")
                return
            
            # Load and display image info
            original_image = ImageUtils.load_image_from_upload(uploaded_file)
            
            if original_image is None:
                st.error("Error loading image. Please try another file.")
                return
            
            # Resize for better performance
            image = ImageUtils.resize_image(original_image, max_size=(1000, 800))
            
            # Display image info
            info = ImageUtils.get_image_info(image)
            st.info(f"📊 **Image Info**\n\n"
                   f"Size: {info['size']}\n\n"
                   f"Format: {info['format']}\n\n"
                   f"Channels: {info['channels']}")
            
            st.divider()
            
            # Filter selection
            st.subheader("🎛️ Available Filters")
            
            # Organize filters by category
            filter_categories = {
                "Original": ["Original"],
                "Basic Effects": ["Grayscale", "Sepia", "Invert", "Vintage"],
                "Adjustments": ["Brightness", "Contrast", "Gamma Correction", "Log Transform", "Histogram Equalization"],
                "Blur & Focus": ["Blur", "Gaussian Blur", "Sharpen", "Dreamy"],
                "Edge & Texture": ["Edge Detection", "Emboss", "Outline", "Oil Painting"],
                "Color Effects": ["Posterize", "Solarize", "Duotone", "Color Balance", "Channel Mixer"],
                "Temperature": ["Cold", "Warm", "Cross Process", "Thermal"],
                "Artistic": ["Pixelate", "Noise", "Vignette"]
            }
            
            # Category selection
            selected_category = st.selectbox("Filter Category:", list(filter_categories.keys()))
            
            # Filter selection within category
            filter_type = st.selectbox(
                "Select a filter:",
                filter_categories[selected_category]
            )
            
            # Filter-specific parameters
            filter_params = {}
            
            if filter_type == "Brightness":
                filter_params['factor'] = st.slider(
                    "Brightness Factor", 
                    min_value=0.1, 
                    max_value=3.0, 
                    value=1.2, 
                    step=0.1
                )
            
            elif filter_type == "Contrast":
                filter_params['factor'] = st.slider(
                    "Contrast Factor", 
                    min_value=0.1, 
                    max_value=3.0, 
                    value=1.5, 
                    step=0.1
                )
            
            elif filter_type == "Blur":
                filter_params['kernel_size'] = st.slider(
                    "Blur Intensity", 
                    min_value=3, 
                    max_value=15, 
                    value=5, 
                    step=2
                )
            
            elif filter_type == "Gaussian Blur":
                filter_params['sigma'] = st.slider(
                    "Blur Sigma", 
                    min_value=0.5, 
                    max_value=5.0, 
                    value=1.0, 
                    step=0.1
                )
            
            elif filter_type == "Gamma Correction":
                filter_params['gamma'] = st.slider(
                    "Gamma Value", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=2.2, 
                    step=0.1
                )
            
            elif filter_type == "Log Transform":
                filter_params['c'] = st.slider(
                    "Log Constant", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=1.0, 
                    step=0.1
                )
            
            elif filter_type == "Posterize":
                filter_params['levels'] = st.slider(
                    "Color Levels", 
                    min_value=2, 
                    max_value=16, 
                    value=4, 
                    step=1
                )
            
            elif filter_type == "Solarize":
                filter_params['threshold'] = st.slider(
                    "Solarize Threshold", 
                    min_value=0, 
                    max_value=255, 
                    value=128, 
                    step=1
                )
            
            elif filter_type == "Color Balance":
                st.write("**Color Balance Controls:**")
                filter_params['red_factor'] = st.slider("Red", 0.0, 2.0, 1.0, 0.1)
                filter_params['green_factor'] = st.slider("Green", 0.0, 2.0, 1.0, 0.1)
                filter_params['blue_factor'] = st.slider("Blue", 0.0, 2.0, 1.0, 0.1)
            
            elif filter_type == "Channel Mixer":
                st.write("**Channel Mixer Controls:**")
                with st.expander("Red Output Channel"):
                    r_r = st.slider("Red→Red", -2.0, 2.0, 1.0, 0.1, key="rr")
                    r_g = st.slider("Green→Red", -2.0, 2.0, 0.0, 0.1, key="rg")
                    r_b = st.slider("Blue→Red", -2.0, 2.0, 0.0, 0.1, key="rb")
                
                with st.expander("Green Output Channel"):
                    g_r = st.slider("Red→Green", -2.0, 2.0, 0.0, 0.1, key="gr")
                    g_g = st.slider("Green→Green", -2.0, 2.0, 1.0, 0.1, key="gg")
                    g_b = st.slider("Blue→Green", -2.0, 2.0, 0.0, 0.1, key="gb")
                
                with st.expander("Blue Output Channel"):
                    b_r = st.slider("Red→Blue", -2.0, 2.0, 0.0, 0.1, key="br")
                    b_g = st.slider("Green→Blue", -2.0, 2.0, 0.0, 0.1, key="bg")
                    b_b = st.slider("Blue→Blue", -2.0, 2.0, 1.0, 0.1, key="bb")
                
                filter_params['red_mix'] = (r_r, r_g, r_b)
                filter_params['green_mix'] = (g_r, g_g, g_b)
                filter_params['blue_mix'] = (b_r, b_g, b_b)
            
            elif filter_type == "Duotone":
                st.write("**Duotone Colors:**")
                col1, col2 = st.columns(2)
                with col1:
                    color1 = st.color_picker("Shadow Color", "#000080")
                with col2:
                    color2 = st.color_picker("Highlight Color", "#FF6B35")
                
                # Convert hex to RGB
                filter_params['color1'] = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
                filter_params['color2'] = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))
            
            elif filter_type == "Vignette":
                filter_params['intensity'] = st.slider(
                    "Vignette Intensity", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.5, 
                    step=0.1
                )
            
            elif filter_type == "Noise":
                filter_params['intensity'] = st.slider(
                    "Noise Intensity", 
                    min_value=1.0, 
                    max_value=50.0, 
                    value=20.0, 
                    step=1.0
                )
            
            elif filter_type == "Oil Painting":
                filter_params['radius'] = st.slider(
                    "Brush Size", 
                    min_value=1, 
                    max_value=10, 
                    value=3, 
                    step=1
                )
                filter_params['intensity'] = st.slider(
                    "Color Intensity", 
                    min_value=5, 
                    max_value=50, 
                    value=20, 
                    step=5
                )
            
            elif filter_type == "Pixelate":
                filter_params['pixel_size'] = st.slider(
                    "Pixel Size", 
                    min_value=2, 
                    max_value=50, 
                    value=10, 
                    step=1
                )
            
            # Apply filter button
            apply_filter = st.button("🚀 Apply Filter", type="primary")
    
    # Main content area
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, caption="Original", use_container_width=True)
        
        with col2:
            st.subheader("✨ Filtered Image")
            
            if apply_filter or 'filtered_image' not in st.session_state:
                if filter_type != "Original":
                    # Show processing indicator
                    with st.spinner(f"Applying {filter_type} filter..."):
                        filtered_image = apply_image_filter(image, filter_type, filter_params)
                    
                    st.session_state.filtered_image = filtered_image
                    st.session_state.current_filter = filter_type
                else:
                    st.session_state.filtered_image = image
                    st.session_state.current_filter = "Original"
            
            # Display filtered image
            if 'filtered_image' in st.session_state:
                st.image(
                    st.session_state.filtered_image, 
                    caption=f"Filtered ({st.session_state.current_filter})", 
                    use_container_width=True
                )
                
                # Download section
                st.divider()
                st.subheader("💾 Download Options")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    download_format = st.selectbox(
                        "Download Format:",
                        ["PNG", "JPEG", "BMP", "TIFF", "WEBP"]
                    )
                
                with col_dl2:
                    filename = st.text_input(
                        "Filename:",
                        value=f"filtered_image.{download_format.lower()}"
                    )
                
                # Create download button
                image_data = ImageUtils.create_download_link(
                    st.session_state.filtered_image, 
                    filename
                )
                
                st.download_button(
                    label="📥 Download Filtered Image",
                    data=image_data,
                    file_name=filename,
                    mime=f"image/{download_format.lower()}"
                )
    
    else:
        # Landing page content
        st.markdown("""
        ### 🌟 Features
        
        - **30+ Filters**: Comprehensive collection of image filters organized by category
        - **All Major Formats**: Support for JPG, PNG, GIF, BMP, TIFF, WebP
        - **Pure NumPy**: All filters implemented using only NumPy operations
        - **Interactive Controls**: Adjust filter parameters in real-time
        - **Easy Download**: Save your filtered images in various formats
        
        ### 🎨 Filter Categories
        
        **Basic Effects**: Grayscale, Sepia, Invert, Vintage  
        **Adjustments**: Brightness, Contrast, Gamma Correction, Log Transform, Histogram Equalization  
        **Blur & Focus**: Box Blur, Gaussian Blur, Sharpen, Dreamy  
        **Edge & Texture**: Edge Detection, Emboss, Outline, Oil Painting  
        **Color Effects**: Posterize, Solarize, Duotone, Color Balance, Channel Mixer  
        **Temperature**: Cold, Warm, Cross Process, Thermal  
        **Artistic**: Pixelate, Noise, Vignette  
        
        ### 🚀 How to Use
        
        1. **Upload**: Choose an image file from your device
        2. **Category**: Select a filter category
        3. **Filter**: Pick a specific filter
        4. **Adjust**: Fine-tune filter parameters if available
        5. **Apply**: Click the "Apply Filter" button
        6. **Download**: Save your filtered image
        
        ### 📁 Supported Formats
        
        **Input**: JPG, JPEG, PNG, GIF, BMP, TIFF, TIF, WebP
        
        **Output**: PNG, JPEG, BMP, TIFF, WebP
        
        ---
        
        **Ready to get started?** Upload an image using the sidebar! 👈
        """)

def apply_image_filter(image: np.ndarray, filter_type: str, params: dict) -> np.ndarray:
    """Apply the selected filter to the image."""
    
    try:
        # Basic Effects
        if filter_type == "Grayscale":
            return ImageFilters.grayscale(image)
        elif filter_type == "Sepia":
            return ImageFilters.sepia(image)
        elif filter_type == "Invert":
            return ImageFilters.invert(image)
        elif filter_type == "Vintage":
            return ImageFilters.vintage(image)
        
        # Adjustments
        elif filter_type == "Brightness":
            return ImageFilters.brightness(image, params.get('factor', 1.2))
        elif filter_type == "Contrast":
            return ImageFilters.contrast(image, params.get('factor', 1.5))
        elif filter_type == "Gamma Correction":
            return ImageFilters.gamma_correction(image, params.get('gamma', 2.2))
        elif filter_type == "Log Transform":
            return ImageFilters.log_transform(image, params.get('c', 1.0))
        elif filter_type == "Histogram Equalization":
            return ImageFilters.histogram_equalization(image)
        
        # Blur & Focus
        elif filter_type == "Blur":
            return ImageFilters.blur(image, params.get('kernel_size', 5))
        elif filter_type == "Gaussian Blur":
            return ImageFilters.gaussian_blur(image, params.get('sigma', 1.0))
        elif filter_type == "Sharpen":
            return ImageFilters.sharpen(image)
        elif filter_type == "Dreamy":
            return ImageFilters.dreamy(image)
        
        # Edge & Texture
        elif filter_type == "Edge Detection":
            return ImageFilters.edge_detection(image)
        elif filter_type == "Emboss":
            return ImageFilters.emboss(image)
        elif filter_type == "Outline":
            return ImageFilters.outline(image)
        elif filter_type == "Oil Painting":
            return ImageFilters.oil_painting(
                image, 
                params.get('radius', 3), 
                params.get('intensity', 20)
            )
        
        # Color Effects
        elif filter_type == "Posterize":
            return ImageFilters.posterize(image, params.get('levels', 4))
        elif filter_type == "Solarize":
            return ImageFilters.solarize(image, params.get('threshold', 128))
        elif filter_type == "Duotone":
            return ImageFilters.duotone(
                image, 
                params.get('color1', (255, 0, 0)), 
                params.get('color2', (0, 0, 255))
            )
        elif filter_type == "Color Balance":
            return ImageFilters.color_balance(
                image,
                params.get('red_factor', 1.0),
                params.get('green_factor', 1.0),
                params.get('blue_factor', 1.0)
            )
        elif filter_type == "Channel Mixer":
            return ImageFilters.channel_mixer(
                image,
                params.get('red_mix', (1, 0, 0)),
                params.get('green_mix', (0, 1, 0)),
                params.get('blue_mix', (0, 0, 1))
            )
        
        # Temperature
        elif filter_type == "Cold":
            return ImageFilters.cold(image)
        elif filter_type == "Warm":
            return ImageFilters.warm(image)
        elif filter_type == "Cross Process":
            return ImageFilters.cross_process(image)
        elif filter_type == "Thermal":
            return ImageFilters.thermal(image)
        
        # Artistic
        elif filter_type == "Pixelate":
            return ImageFilters.pixelate(image, params.get('pixel_size', 10))
        elif filter_type == "Noise":
            return ImageFilters.noise(image, params.get('intensity', 20.0))
        elif filter_type == "Vignette":
            return ImageFilters.vignette(image, params.get('intensity', 0.5))
        
        else:
            return image
            
    except Exception as e:
        st.error(f"Error applying filter: {str(e)}")
        return image

if __name__ == "__main__":
    main()