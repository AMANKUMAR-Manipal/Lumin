// Main JavaScript for the Lumin AI Vision Detection App

document.addEventListener('DOMContentLoaded', function() {
    // Image Upload Preview
    const fileInput = document.getElementById('file');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const previewPlaceholder = document.querySelector('.preview-placeholder');
    const imageDimensions = document.getElementById('image-dimensions');
    const imageInfo = document.getElementById('image-info');
    const detectBtn = document.getElementById('detect-btn');
    
    // Only initialize if elements exist (to prevent errors on pages without these elements)
    if (fileInput && imagePreview) {
        // Handle file selection
        fileInput.addEventListener('change', updatePreview);
        
        // Handle drag and drop
        imagePreview.addEventListener('dragover', function(e) {
            e.preventDefault();
            imagePreview.classList.add('dragover');
        });
        
        imagePreview.addEventListener('dragleave', function(e) {
            e.preventDefault();
            imagePreview.classList.remove('dragover');
        });
        
        imagePreview.addEventListener('drop', function(e) {
            e.preventDefault();
            imagePreview.classList.remove('dragover');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updatePreview();
            }
        });
    }
    
    function updatePreview() {
        if (fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            
            // Create a URL for the file
            const url = URL.createObjectURL(file);
            
            // Update image preview
            previewImg.src = url;
            previewImg.style.display = 'block';
            
            // Hide placeholder
            if (previewPlaceholder) {
                previewPlaceholder.style.display = 'none';
            }
            
            // Enable detect button
            if (detectBtn) {
                detectBtn.disabled = false;
            }
            
            // Get and display image dimensions
            const img = new Image();
            img.onload = function() {
                if (imageDimensions) {
                    imageDimensions.textContent = `${this.width} x ${this.height}`;
                }
                if (imageInfo) {
                    imageInfo.style.display = 'block';
                }
            };
            img.src = url;
        }
    }
    
    // Results page: Algorithm comparison
    const compareBtn = document.getElementById('compare-btn');
    const comparisonResults = document.getElementById('comparison-results');
    
    if (compareBtn && comparisonResults) {
        compareBtn.addEventListener('click', function() {
            const imagePath = this.getAttribute('data-image-path');
            
            // Show loading state
            comparisonResults.innerHTML = '<div class="text-center py-3"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Comparing detection results...</p></div>';
            
            // Fetch comparison data
            fetch('/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image_path: imagePath }),
            })
            .then(response => response.json())
            .then(data => {
                // Create comparison table
                let comparisonHTML = `
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th scope="col">Metric</th>
                `;
                
                // Add algorithm columns
                for (const algo in data) {
                    comparisonHTML += `<th scope="col" class="text-center">
                        <i class="fas fa-${algo === 'cnn' ? 'brain' : (algo === 'knn' ? 'project-diagram' : 'box')} 
                            text-${algo === 'cnn' ? 'danger' : (algo === 'knn' ? 'info' : 'success')} me-2"></i>
                        ${algo.toUpperCase()}
                    </th>`;
                }
                
                comparisonHTML += `
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Total Objects</td>
                `;
                
                // Add total objects row
                for (const algo in data) {
                    comparisonHTML += `<td class="text-center">${data[algo].total_objects}</td>`;
                }
                
                comparisonHTML += `
                            </tr>
                            <tr>
                                <td>Average Confidence</td>
                `;
                
                // Add confidence row
                for (const algo in data) {
                    const confidence = data[algo].avg_confidence.toFixed(2);
                    let confidenceClass = 'success';
                    if (confidence < 0.7) confidenceClass = 'warning';
                    if (confidence < 0.5) confidenceClass = 'danger';
                    
                    comparisonHTML += `<td class="text-center text-${confidenceClass}">${(confidence * 100).toFixed(0)}%</td>`;
                }
                
                comparisonHTML += `
                            </tr>
                            <tr>
                                <td>Processing Time</td>
                `;
                
                // Add time row
                for (const algo in data) {
                    comparisonHTML += `<td class="text-center">${data[algo].time_taken.toFixed(3)} sec</td>`;
                }
                
                comparisonHTML += `
                            </tr>
                            <tr>
                                <td>Object Types</td>
                `;
                
                // Add object types row
                for (const algo in data) {
                    comparisonHTML += `<td class="text-center">`;
                    
                    const objects = data[algo].object_counts;
                    for (const obj in objects) {
                        comparisonHTML += `<span class="badge bg-secondary me-1 mb-1">${obj}: ${objects[obj]}</span>`;
                    }
                    
                    comparisonHTML += `</td>`;
                }
                
                comparisonHTML += `
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div class="alert alert-info mt-3">
                    <i class="fas fa-info-circle me-2"></i>
                    <strong>Analysis:</strong> Each algorithm has different strengths. CNN typically excels at recognizing complex objects,
                    KNN is fastest for processing, while YOLOv3 often has the best overall accuracy.
                </div>
                `;
                
                // Update results
                comparisonResults.innerHTML = comparisonHTML;
            })
            .catch(error => {
                comparisonResults.innerHTML = `<div class="alert alert-danger">Error comparing results: ${error.message}</div>`;
            });
        });
    }
    
    // Dashboard charts initialization
    const processingTimesChart = document.getElementById('processingTimesChart');
    const objectDistributionChart = document.getElementById('objectDistributionChart');
    
    if (processingTimesChart) {
        // Get data from data attributes
        const processingTimes = JSON.parse(processingTimesChart.getAttribute('data-processing-times'));
        const timeLabels = JSON.parse(processingTimesChart.getAttribute('data-time-labels'));
        
        // Create chart using Chart.js
        new Chart(processingTimesChart, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [{
                    label: 'Processing Time (seconds)',
                    data: processingTimes,
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true,
                    pointBackgroundColor: '#dc3545',
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(3) + 's';
                            }
                        }
                    }
                }
            }
        });
    }
    
    if (objectDistributionChart) {
        // Get data from data attributes
        const labels = JSON.parse(objectDistributionChart.getAttribute('data-labels'));
        const data = JSON.parse(objectDistributionChart.getAttribute('data-values'));
        
        // Create colorful palette
        const backgroundColors = [
            '#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6f42c1', '#fd7e14'
        ];
        
        // Create chart using Chart.js
        new Chart(objectDistributionChart, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: backgroundColors.slice(0, data.length),
                    borderWidth: 1,
                    borderColor: '#343a40'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1) + '%';
                                return `${label}: ${value} (${percentage})`;
                            }
                        }
                    }
                },
                cutout: '65%'
            }
        });
    }
    
    // Radar chart for algorithm comparison
    const algorithmComparisonChart = document.getElementById('algorithmComparisonChart');
    
    if (algorithmComparisonChart) {
        // Get data from data attributes
        const cnnCount = parseFloat(algorithmComparisonChart.getAttribute('data-cnn-count'));
        const cnnConf = parseFloat(algorithmComparisonChart.getAttribute('data-cnn-conf'));
        const cnnSpeed = parseFloat(algorithmComparisonChart.getAttribute('data-cnn-speed'));
        
        const knnCount = parseFloat(algorithmComparisonChart.getAttribute('data-knn-count'));
        const knnConf = parseFloat(algorithmComparisonChart.getAttribute('data-knn-conf'));
        const knnSpeed = parseFloat(algorithmComparisonChart.getAttribute('data-knn-speed'));
        
        const yoloCount = parseFloat(algorithmComparisonChart.getAttribute('data-yolo-count'));
        const yoloConf = parseFloat(algorithmComparisonChart.getAttribute('data-yolo-conf'));
        const yoloSpeed = parseFloat(algorithmComparisonChart.getAttribute('data-yolo-speed'));
        
        // Create chart using Chart.js
        new Chart(algorithmComparisonChart, {
            type: 'radar',
            data: {
                labels: ['Object Count', 'Confidence', 'Speed'],
                datasets: [
                    {
                        label: 'CNN',
                        data: [cnnCount, cnnConf, cnnSpeed],
                        backgroundColor: 'rgba(220, 53, 69, 0.2)',
                        borderColor: '#dc3545',
                        pointBackgroundColor: '#dc3545',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#dc3545',
                        borderWidth: 2
                    },
                    {
                        label: 'KNN',
                        data: [knnCount, knnConf, knnSpeed],
                        backgroundColor: 'rgba(23, 162, 184, 0.2)',
                        borderColor: '#17a2b8',
                        pointBackgroundColor: '#17a2b8',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#17a2b8',
                        borderWidth: 2
                    },
                    {
                        label: 'YOLOv3',
                        data: [yoloCount, yoloConf, yoloSpeed],
                        backgroundColor: 'rgba(40, 167, 69, 0.2)',
                        borderColor: '#28a745',
                        pointBackgroundColor: '#28a745',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#28a745',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                elements: {
                    line: {
                        tension: 0.1
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: true,
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        min: 0,
                        max: 1,
                        ticks: {
                            stepSize: 0.2,
                            display: false
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            font: {
                                weight: 'bold'
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Animation on scroll initialization
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true
    });
});