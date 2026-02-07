document.addEventListener("DOMContentLoaded", () => {
    const gradeButton = document.getElementById("grade-button");
    const rubricInput = document.getElementById("rubric-input");
    const answerFileInput = document.getElementById("answer-file");
    const referenceFileInput = document.getElementById("reference-file");
    const resultsContainer = document.getElementById("results-container");
    const resultsOutput = document.getElementById("results");

    gradeButton.addEventListener("click", async () => {
        const rubric = rubricInput.value;
        const studentFile = answerFileInput.files[0];
        const referenceFile = referenceFileInput.files[0];

        if (!rubric) return alert("Please paste the rubric.");
        if (!studentFile) return alert("Please upload the student's answer.");
        // Note: Reference file is optional (if missing, we skip RAG)

        resultsOutput.textContent = "🚀 Initializing RAG Pipeline...\n\n1. Ingesting Reference Material...\n2. Extracting Student Claims...\n3. Retrieving Ground Truth...\n4. Running Multi-Agent Grading...";
        resultsContainer.style.display = "block";
        gradeButton.disabled = true;

        const formData = new FormData();
        formData.append('rubric', rubric);
        formData.append('student_answer_pdf', studentFile);
        if (referenceFile) {
            formData.append('reference_pdf', referenceFile);
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/grade', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error');
            
            resultsOutput.textContent = data.evaluation;

        } catch (error) {
            resultsOutput.textContent = `Error: ${error.message}`;
        } finally {
            gradeButton.disabled = false;
        }
    });
});