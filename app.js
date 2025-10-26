// Wait for the webpage to be fully loaded
document.addEventListener("DOMContentLoaded", () => {

    // Get the HTML elements
    const gradeButton = document.getElementById("grade-button");
    const rubricInput = document.getElementById("rubric-input");
    const answerFileInput = document.getElementById("answer-file");
    const resultsContainer = document.getElementById("results-container");
    const resultsOutput = document.getElementById("results");

    // Add a click event listener to the button
    gradeButton.addEventListener("click", async () => {

        const rubric = rubricInput.value;
        const studentAnswerFile = answerFileInput.files[0]; // Get the first (and only) file

        // --- 1. Validation ---
        if (!rubric) {
            alert("Please paste the rubric.");
            return;
        }
        if (!studentAnswerFile) {
            alert("Please upload the student's answer PDF.");
            return;
        }

        // --- 2. Prepare for API Call ---
        resultsOutput.textContent = "🔄 Processing PDF and grading... This may take a minute or two.";
        resultsContainer.style.display = "block";
        gradeButton.disabled = true;
        gradeButton.textContent = "Grading...";

        // Use FormData to send both text and a file
        const formData = new FormData();
        formData.append('rubric', rubric);
        formData.append('student_answer_pdf', studentAnswerFile);

        try {
            // --- 3. Call the Backend ---
            const response = await fetch('http://127.0.0.1:5000/grade', {
                method: 'POST',
                // IMPORTANT: Do NOT set 'Content-Type' header.
                // The browser automatically sets it to 'multipart/form-data'
                // and includes the file boundary.
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle errors from the backend (like "OCR Failure")
                throw new Error(data.error || 'An unknown server error occurred.');
            }
            
            // --- 4. Display Results ---
            // Success! Display the AI's evaluation
            resultsOutput.textContent = data.evaluation;

        } catch (error) {
            // Display any errors
            resultsOutput.textContent = `Error: ${error.message}`;
            console.error("Error:", error);
        } finally {
            // --- 5. Reset Button ---
            gradeButton.disabled = false;
            gradeButton.textContent = "Grade Answer";
        }
    });
});