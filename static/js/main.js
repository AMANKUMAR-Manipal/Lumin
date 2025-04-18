document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const cameraStream = document.getElementById('camera-stream');
    const cameraLoading = document.getElementById('camera-loading');
    const imageUpload = document.getElementById('image-upload');
    const processImageBtn = document.getElementById('process-image');
    const uploadPreview = document.getElementById('upload-preview');
    const uploadLoading = document.getElementById('upload-loading');
    const detectionInfo = document.getElementById('detection-info');
    const cameraTab = document.getElementById('camera-tab');
    const uploadTab = document.getElementById('upload-tab');

    // Camera stream status
    let isCameraActive = false;

    // Event Listeners
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);
    imageUpload.addEventListener('change', handleImageUpload);
    processImageBtn.addEventListener('click', processImage);
    
    // Handle tab switching
    cameraTab.addEventListener('click', function() {
        if (isCameraActive) {
            cameraStream.src = '/video_feed?' + new Date().getTime();
        } else {
            cameraStream.src = '/static/svg/camera.svg';
        }
    });
    
    uploadTab.addEventListener('click', function() {
        if (isCameraActive) {
            stopCamera();
        }
    });

    // Functions
    function startCamera() {
        // Show loading indicator
        cameraLoading.classList.remove('d-none');
        
        // Make API call to start camera
        fetch('/start_camera')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Update camera stream with video feed URL and cache-busting query param
                    cameraStream.src = '/video_feed?' + new Date().getTime();
                    isCameraActive = true;
                    
                    // Update button states
                    startCameraBtn.disabled = true;
                    stopCameraBtn.disabled = false;
                    
                    // Update detection info
                    detectionInfo.innerHTML = '<p>Camera active. Objects will be detected in real-time.</p>';
                    
                    // Hide loading indicator after a short delay
                    setTimeout(() => {
                        cameraLoading.classList.add('d-none');
                    }, 1000);
                } else {
                    // Handle error
                    cameraLoading.classList.add('d-none');
                    detectionInfo.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
                }
            })
            .catch(error => {
                console.error('Error starting camera:', error);
                cameraLoading.classList.add('d-none');
                detectionInfo.innerHTML = `<div class="alert alert-danger">Error connecting to server</div>`;
            });
    }

    function stopCamera() {
        // Make API call to stop camera
        fetch('/stop_camera')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Reset camera stream to default image
                    cameraStream.src = '/static/svg/camera.svg';
                    isCameraActive = false;
                    
                    // Update button states
                    startCameraBtn.disabled = false;
                    stopCameraBtn.disabled = true;
                    
                    // Update detection info
                    detectionInfo.innerHTML = '<p>Camera stopped. Start the camera or upload an image to see detection results.</p>';
                }
            })
            .catch(error => {
                console.error('Error stopping camera:', error);
            });
    }

    function handleImageUpload(event) {
        const file = event.target.files[0];
        
        // Enable/disable process button based on file selection
        processImageBtn.disabled = !file;
        
        if (file) {
            // Create a preview of the selected image
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadPreview.src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            // Update detection info
            detectionInfo.innerHTML = '<p>Image selected. Click "Process Image" to detect objects.</p>';
        } else {
            // Reset preview if no file is selected
            uploadPreview.src = '/static/svg/upload.svg';
            detectionInfo.innerHTML = '<p>No image selected.</p>';
        }
    }

    function processImage() {
        const file = imageUpload.files[0];
        
        if (!file) {
            detectionInfo.innerHTML = '<div class="alert alert-warning">Please select an image first</div>';
            return;
        }
        
        // Show loading indicator
        uploadLoading.classList.remove('d-none');
        
        // Create form data for the upload
        const formData = new FormData();
        formData.append('file', file);
        
        // Make API call to process the image
        fetch('/detect_image', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                uploadLoading.classList.add('d-none');
                
                if (data.status === 'success') {
                    // Display the processed image
                    uploadPreview.src = data.image_path + '?' + new Date().getTime(); // Add timestamp to prevent caching
                    
                    // Update detection info
                    detectionInfo.innerHTML = `
                        <div class="alert alert-success">
                            <p>${data.message}</p>
                            <p>Object detection completed successfully</p>
                        </div>
                    `;
                } else {
                    // Handle error
                    detectionInfo.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
                }
            })
            .catch(error => {
                console.error('Error processing image:', error);
                uploadLoading.classList.add('d-none');
                detectionInfo.innerHTML = `<div class="alert alert-danger">Server error when processing image</div>`;
            });
    }
});
