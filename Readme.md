# 🎨 Image Filter Studio [https://img-filter.streamlit.app/]

A web-based image filtering application built with Streamlit and NumPy. Apply various image filters to your photos with an intuitive interface.

## ✨ Features

- **Multiple Filters**: Grayscale, Sepia, Brightness, Contrast, Invert, Blur, Sharpen, Edge Detection, Vintage
- **Format Support**: JPG, PNG, GIF, BMP, TIFF, WebP
- **Pure NumPy**: All filters implemented using only NumPy operations for educational purposes
- **Interactive Controls**: Real-time parameter adjustment
- **Easy Download**: Save filtered images in multiple formats

## 🚀 Quick Start

### Local Installation

1. **Clone or download the files**:
   - `app.py` - Main Streamlit application
   - `filters.py` - Image filter implementations
   - `utils.py` - Utility functions
   - `requirements.txt` - Dependencies

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Cloud Deployment

#### Streamlit Cloud
1. Upload files to a GitHub repository
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io/)
3. Deploy directly from your repository

#### Other Platforms
- **Heroku**: Add a `Procfile` with `web: streamlit run app.py --server.port=$PORT`
- **Railway**: Works out of the box with the provided `requirements.txt`
- **Render**: Deploy as a web service using the Streamlit command

## 📁 File Structure

```
image-filter-studio/
├── app.py              # Main Streamlit application
├── filters.py          # Image filter implementations
├── utils.py           # Image utility functions
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🎛️ Available Filters

### Basic Filters
- **Grayscale**: Convert to black and white using luminance weights
- **Sepia**: Apply vintage sepia tone effect
- **Invert**: Invert all colors

### Adjustment Filters
- **Brightness**: Adjust image brightness (0.1x to 3.0x)
- **Contrast**: Modify image contrast (0.1x to 3.0x)

### Effect Filters
- **Blur**: Apply box blur with adjustable kernel size
- **Sharpen**: Enhance image sharpness using convolution
- **Edge Detection**: Detect edges using Sobel operator
- **Vintage**: Apply retro effect with vignetting

## 💻 Usage

1. **Upload Image**: Use the sidebar file uploader to select an image
2. **Choose Filter**: Select from the dropdown menu
3. **Adjust Parameters**: Use sliders for filters with customizable settings
4. **Apply Filter**: Click the "Apply Filter" button
5. **Download**: Choose format and download your filtered image

## 🔧 Technical Details

### Image Processing
- All filters implemented using pure NumPy operations
- Automatic image resizing for performance optimization
- Support for RGB and grayscale images
- Memory-efficient processing

### Supported Formats
**Input**: JPG, JPEG, PNG, GIF, BMP, TIFF, TIF, WebP  
**Output**: PNG, JPEG, BMP, TIFF, WebP

## 🛠️ Customization

### Adding New Filters

1. **Add filter method to `filters.py`**:
   ```python
   @staticmethod
   def your_filter(image: np.ndarray, param: float = 1.0) -> np.ndarray:
       # Your filter implementation
       return processed_image
   ```

2. **Update the filter list in `app.py`**:
   ```python
   filter_type = st.selectbox("Select a filter:", [
       # ... existing filters
       "Your Filter"
   ])
   ```

3. **Add filter application logic**:
   ```python
   elif filter_type == "Your Filter":
       return ImageFilters.your_filter(image, params.get('param', 1.0))
   ```

### Adding Parameters
Add sliders or other input widgets in the sidebar section of `app.py` for your custom parameters.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new filters
- Improve existing implementations
- Enhance the UI/UX
- Fix bugs or optimize performance

## 📞 Support

If you encounter any issues or have questions, please check the documentation or create an issue in the repository.