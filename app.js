document.addEventListener('DOMContentLoaded', () => {
    const gradingForm = document.getElementById('gradingForm');
    const submitBtn = document.getElementById('submitBtn');
    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');
    const resultsBody = document.getElementById('resultsBody');
    const overallStatus = document.getElementById('overall-status');
    
    // Modal elements
    const feedbackModal = document.getElementById('feedbackModal');
    const closeModal = document.getElementById('closeModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');

    let pollingInterval;
    const API_BASE_URL = 'http://127.0.0.1:5000';

    gradingForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const answerFiles = document.getElementById('answerPapers').files;
        const kbFiles = document.getElementById('knowledgeBase').files;
        
        // Validation checks
        if (answerFiles.length === 0) return alert("Please upload at least one Answer Paper PDF.");
        if (kbFiles.length === 0) return alert("Please upload at least one Course Content PDF.");

        // 1. Setup the UI for processing
        submitBtn.disabled = true;
        submitBtn.textContent = "Processing...";
        uploadSection.style.display = "none";
        resultsSection.style.display = "block";

        // 2. Populate the table with "Pending" rows based on answer papers
        resultsBody.innerHTML = '';
        Array.from(answerFiles).forEach((file, index) => {
            const row = document.createElement('tr');
            row.id = `file-row-${index}`;
            row.innerHTML = `
                <td>${file.name}</td>
                <td class="status-pending" id="status-${index}">Pending</td>
                <td id="score-${index}">--</td>
                <td><button disabled id="btn-${index}">View Feedback</button></td>
            `;
            resultsBody.appendChild(row);
        });

        // 3. Prepare FormData for the backend (This automatically packages all files and text inputs)
        const formData = new FormData(gradingForm);
        
        try {
            // Step 1: Send the initial batch request to the Python backend
            const response = await fetch(`${API_BASE_URL}/api/grade-batch`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Failed to start grading process.");

            const data = await response.json();
            const taskId = data.task_id; // Backend returns a unique task_id
            
            overallStatus.innerHTML = `Status: <strong>Processing ${answerFiles.length} papers...</strong>`;

            // Step 2: Start polling the backend for updates
            startPolling(taskId, answerFiles.length);

        } catch (error) {
            console.error("Error:", error);
            overallStatus.innerHTML = `Status: <span class="status-error">Error initiating grading.</span>`;
            uploadSection.style.display = "block"; // allow retry
            submitBtn.disabled = false;
            submitBtn.textContent = "Grade Papers";
        }
    });

    function startPolling(taskId, totalFiles) {
        // Poll every 5 seconds
        pollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/status/${taskId}`);
                const statusData = await response.json();

                let completedCount = 0;

                // Update the table rows based on backend status
                statusData.results.forEach((result, index) => {
                    const statusTd = document.getElementById(`status-${index}`);
                    const scoreTd = document.getElementById(`score-${index}`);
                    const btn = document.getElementById(`btn-${index}`);

                    if (result.status === 'completed') {
                        statusTd.textContent = 'Graded';
                        statusTd.className = 'status-done';
                        scoreTd.textContent = result.score; 
                        
                        btn.disabled = false;
                        btn.onclick = () => showFeedback(result.filename, result.feedback);
                        
                        completedCount++;
                    } else if (result.status === 'error') {
                        statusTd.textContent = 'Failed';
                        statusTd.className = 'status-error';
                        scoreTd.textContent = 'N/A';
                        
                        btn.disabled = false;
                        btn.textContent = "View Error";
                        btn.onclick = () => showFeedback(result.filename, result.feedback);
                        
                        completedCount++;
                    } else if (result.status === 'processing') {
                        statusTd.textContent = 'Grading...';
                    }
                });

                // Check if all files are processed
                if (completedCount === totalFiles) {
                    clearInterval(pollingInterval);
                    overallStatus.innerHTML = `Status: <span class="status-done">All papers processed successfully!</span>`;
                }

            } catch (error) {
                console.error("Polling error:", error);
            }
        }, 5000);
    }

    // Handle showing the feedback modal
    function showFeedback(filename, feedbackText) {
        modalTitle.textContent = `Feedback: ${filename}`;
        modalBody.innerHTML = `<p>${feedbackText.replace(/\n/g, '<br>')}</p>`;
        feedbackModal.style.display = 'block';
    }

    // Handle closing the modal
    closeModal.onclick = () => {
        feedbackModal.style.display = "none";
    }

    window.onclick = (event) => {
        if (event.target === feedbackModal) {
            feedbackModal.style.display = "none";
        }
    }
}); 