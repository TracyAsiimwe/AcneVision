// ========== WEBCAM FUNCTIONALITY ==========
const startWebcamBtn = document.getElementById('start-webcam');
const webcamContainer = document.getElementById('webcam-container');
const video = document.getElementById('webcam-video');
const canvas = document.getElementById('webcam-canvas');
const captureBtn = document.getElementById('capture-btn');
const retakeBtn = document.getElementById('retake-btn');
const analyzeBtn = document.getElementById('analyze-webcam');

let stream = null;
let capturedImage = null;

if (startWebcamBtn) {
    startWebcamBtn.addEventListener('click', async function() {
        try {
            // Check if running on localhost (camera works without HTTPS)
            const isLocalhost = window.location.hostname === 'localhost' || 
                               window.location.hostname === '127.0.0.1';
            
            if (!isLocalhost && window.location.protocol !== 'https:') {
                alert('Camera access requires HTTPS or localhost.\n\nPlease use: http://127.0.0.1:5000\n\nOr access via HTTPS.');
                return;
            }
            
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                } 
            });
            
            video.srcObject = stream;
            webcamContainer.classList.remove('hidden');
            startWebcamBtn.classList.add('hidden');
            
        } catch (err) {
            console.error('Webcam error:', err);
            if (err.name === 'NotAllowedError') {
                alert('Camera permission denied. Please allow camera access in your browser.');
            } else if (err.name === 'NotFoundError') {
                alert('No camera found. Please connect a webcam.');
            } else {
                alert('Could not access webcam: ' + err.message);
            }
        }
    });
}

// Capture photo
if (captureBtn) {
    captureBtn.addEventListener('click', function() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        // Stop video stream
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        video.classList.add('hidden');
        canvas.classList.remove('hidden');
        
        capturedImage = canvas.toDataURL('image/png');
        
        captureBtn.classList.add('hidden');
        retakeBtn.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
    });
}

// Retake photo
if (retakeBtn) {
    retakeBtn.addEventListener('click', async function() {
        // Restart camera
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } 
            });
            video.srcObject = stream;
        } catch (err) {
            console.error('Could not restart camera:', err);
        }
        
        video.classList.remove('hidden');
        canvas.classList.add('hidden');
        
        captureBtn.classList.remove('hidden');
        retakeBtn.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        
        capturedImage = null;
    });
}

// Analyze webcam photo
if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async function() {
        if (!capturedImage) return;
        
        if (loadingOverlay) {
            loadingOverlay.classList.remove('hidden');
        }

        try {
            const response = await fetch('/webcam_capture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: capturedImage })
            });

            const result = await response.json();

            if (result.success) {
                window.location.reload();
            } else {
                alert('Analysis failed: ' + (result.error || 'Unknown error'));
                if (loadingOverlay) {
                    loadingOverlay.classList.add('hidden');
                }
            }

        } catch (err) {
            console.error('Analysis error:', err);
            alert('Failed to analyze image. Please try again.');
            if (loadingOverlay) {
                loadingOverlay.classList.add('hidden');
            }
        }
    });
}