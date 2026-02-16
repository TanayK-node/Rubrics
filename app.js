document.addEventListener("DOMContentLoaded", () => {
    const gradeButton = document.getElementById("grade-button");
    const rubricInput = document.getElementById("rubric-input");
    const answerFileInput = document.getElementById("answer-file");
    
    // Output Elements
    const resultsArea = document.getElementById("results-area");
    const tableBody = document.getElementById("grading-table-body");
    const displayScore = document.getElementById("display-score");
    const displayFeedback = document.getElementById("display-feedback");
    
    // Metric Elements
    const metricTime = document.getElementById("metric-time");
    const metricMethod = document.getElementById("metric-method");

    gradeButton.addEventListener("click", async () => {
        // 1. Basic Validation
        const rubric = rubricInput.value.trim();
        const studentAnswerFile = answerFileInput.files[0];

        if (!rubric) return alert("Please paste the grading rubric.");
        if (!studentAnswerFile) return alert("Please upload a PDF file.");

        // 2. UI Loading State
        gradeButton.textContent = "⏳ Analyzing Paper... (Agents Running)";
        gradeButton.disabled = true;
        resultsArea.style.display = "none";
        tableBody.innerHTML = ""; // Clear previous results

        const formData = new FormData();
        formData.append('rubric', rubric);
        formData.append('student_answer_pdf', studentAnswerFile);

        try {
            // 3. Send Request
            const response = await fetch('http://127.0.0.1:5000/grade', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.error || "Server Error");

            // 4. Parse the Nested JSON
            // The AI returns a JSON string *inside* the 'result' field
            // We need to parse that string into a real Object
            let aiResult;
            if (typeof data.result === "string") {
                 aiResult = JSON.parse(data.result);
            } else {
                 aiResult = data.result;
            }

            // 5. Update Research Metrics
            metricTime.textContent = `${data.research_metadata.processing_time_seconds}s`;
            // Format the method list nicely
            const uniqueMethods = [...new Set(data.research_metadata.extraction_method_breakdown)];
            metricMethod.textContent = uniqueMethods.join(", ") || "Native";

            // 6. Update Total Score & Feedback
            displayScore.textContent = `${aiResult.total_score_awarded} / ${aiResult.total_max_score}`;
            displayFeedback.textContent = aiResult.summary_feedback;

            // 7. Build the Table (The "Reasoning Trace")
            aiResult.evaluation.forEach(item => {
                const row = document.createElement("tr");

                row.innerHTML = `
                    <td class="q-col">${item.question_id}</td>
                    <td class="score-col">${item.score_awarded} / ${item.max_score}</td>
                    <td>
                        <strong>Reasoning:</strong> ${item.reasoning_trace}<br>
                        ${item.evidence_quote ? `<span class="quote">"Doc: ${item.evidence_quote}"</span>` : ''}
                        ${item.improvement_suggestion ? `<div class="suggestion">💡 Tip: ${item.improvement_suggestion}</div>` : ''}
                    </td>
                `;
                tableBody.appendChild(row);
            });

            // Show Results
            resultsArea.style.display = "block";

        } catch (error) {
            console.error(error);
            alert(`Error: ${error.message}`);
        } finally {
            gradeButton.textContent = "🚀 Run Evaluation";
            gradeButton.disabled = false;
        }
    });
});