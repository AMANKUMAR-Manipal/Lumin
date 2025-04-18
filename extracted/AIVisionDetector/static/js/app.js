// Main JavaScript for Object Detection Application

document.addEventListener('DOMContentLoaded', function() {
    // Toggle file input when clicking on the preview area
    const imagePreview = document.getElementById('image-preview');
    const fileInput = document.getElementById('file');
    
    if (imagePreview) {
        imagePreview.addEventListener('click', function() {
            fileInput.click();
        });
    }
    
    // Form validation
    const uploadForm = document.getElementById('upload-form');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            const file = fileInput.files[0];
            
            // Validate file type
            if (file) {
                const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
                if (!validTypes.includes(file.type)) {
                    event.preventDefault();
                    alert('Please select a valid image file (JPG or PNG).');
                    return false;
                }
                
                // Validate file size (max 16MB)
                if (file.size > 16 * 1024 * 1024) {
                    event.preventDefault();
                    alert('File size is too large. Maximum size is 16MB.');
                    return false;
                }
                
                // Show loading state
                const detectBtn = document.getElementById('detect-btn');
                if (detectBtn) {
                    detectBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                    detectBtn.disabled = true;
                }
            } else {
                event.preventDefault();
                alert('Please select an image file.');
                return false;
            }
        });
    }
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Image comparison slider (if present)
    const comparisonSlider = document.getElementById('comparison-slider');
    if (comparisonSlider) {
        comparisonSlider.addEventListener('input', function() {
            const overlayImage = document.querySelector('.comparison-overlay');
            if (overlayImage) {
                overlayImage.style.width = this.value + '%';
            }
        });
    }
});

// Update image preview when a file is selected
function updatePreview(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        const previewImg = document.getElementById('preview-img');
        const placeholder = document.querySelector('.preview-placeholder');
        
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            if (placeholder) {
                placeholder.style.display = 'none';
            }
        };
        
        reader.readAsDataURL(input.files[0]);
    }
}

// Helper function to format numbers with commas
function numberWithCommas(x) {
    return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
