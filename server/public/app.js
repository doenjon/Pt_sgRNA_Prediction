const API_BASE_URL = window.location.port ? `${window.location.protocol}//${window.location.hostname}:${window.location.port}` : `${window.location.protocol}//${window.location.hostname}`;

document.getElementById('guideForm')?.addEventListener('submit', function(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitButton = form.querySelector('button[type="submit"]');
    const spinner = submitButton.querySelector('.spinner-border');
    const sequence = document.getElementById('sequence').value;

    // Clear previous error messages
    const existingError = form.querySelector('.error-message');
    if (existingError) existingError.remove();

    // Disable button and show spinner
    submitButton.disabled = true;
    spinner.classList.remove('d-none');

    fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sequence }),
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            console.error('Response not ok:', response.statusText);
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Redirecting to results page with ID:', data.resultId);
        window.location.href = `results.html?resultId=${data.resultId}`;
    })
    .catch((error) => {
        console.error('Error:', error);
        console.error('Error details:', error.message);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger error-message';
        errorDiv.textContent = 'An error occurred. Please try again later.';
        form.appendChild(errorDiv);
    })
    .finally(() => {
        // Reset button state
        submitButton.disabled = false;
        spinner.classList.add('d-none');
    });
});

// Results page handling
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, checking for results page');
    if (document.querySelector('.results-page')) {
        console.log('On results page');
        const urlParams = new URLSearchParams(window.location.search);
        const resultId = urlParams.get('resultId');
        
        if (!resultId) {
            console.log('No resultId found, redirecting to home');
            window.location.href = '/';
            return;
        }

        console.log('Fetching results for ID:', resultId);
        fetchResults(resultId);
    }
});

function fetchResults(resultId) {
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');
    const errorState = document.getElementById('errorState');
    const downloadBtn = document.getElementById('downloadBtn');

    function pollResults() {
        fetch(`${API_BASE_URL}/api/results/${resultId}`)
            .then(response => {
                if (response.status === 500) {
                    throw new Error('Server error');
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'processing') {
                    // If still processing, continue polling after a delay
                    setTimeout(pollResults, 2000);
                    
                    // Ensure loading state is visible
                    loadingState.style.display = 'block';
                    resultsContent.style.display = 'none';
                    errorState.style.display = 'none';
                    
                } else if (data.status === 'completed') {
                    // Results are ready
                    loadingState.style.display = 'none';
                    resultsContent.style.display = 'block';
                    errorState.style.display = 'none';
                    displayResults(data);
                    createSequenceMap(data.inputSequence, data.guides);
                    downloadBtn.disabled = false;
                } else {
                    // Handle error state
                    throw new Error(data.error || 'Failed to process results');
                }
            })
            .catch(error => {
                console.error('Error fetching results:', error);
                loadingState.style.display = 'none';
                errorState.style.display = 'block';
                resultsContent.style.display = 'none';
                document.getElementById('errorMessage').textContent = 
                    error.message || 'Failed to load results. Please try again.';
            });
    }

    // Show initial loading state
    loadingState.style.display = 'block';
    resultsContent.style.display = 'none';
    errorState.style.display = 'none';
    downloadBtn.disabled = true;

    // Start polling
    pollResults();
}

function displayResults(data) {
    if (!data || !data.guides || data.error) {
        const errorState = document.getElementById('errorState');
        errorState.style.display = 'block';
        document.getElementById('errorMessage').textContent = 
            data?.error || 'Failed to generate guides. Please try again.';
        return;
    }

    console.log('Displaying results:', data);
    const resultsContent = document.getElementById('resultsContent');
    let html = '';

    // Display input sequence
    html += `
        <div class="sequence-display mb-4">
            <code>${data.inputSequence}</code>
        </div>
    `;

    // Display guide sequences
    data.guides.forEach((guide, index) => {
        console.log(`Guide ${index + 1} data:`, guide);  // Log each guide's data
        html += `
            <div class="guide-result" id="guide-${index + 1}">
                <div class="d-flex justify-content-between align-items-start">
                    <h5>Guide ${index + 1}</h5>
                    <span class="score-badge">Score: ${(guide.score != null ? Number(guide.score).toFixed(2) : 'N/A')}</span>
                </div>
                <div class="sequence-display">
                    <code>${guide.sequence}</code>
                    <small class="text-muted ms-2">
                        <i class="fas fa-arrow-${guide.strand === '+' ? 'right' : 'left'} me-1"></i>
                        ${guide.strand} strand
                    </small>
                </div>
                <div class="row mt-3">
                    <div class="col-md-3">
                        <small class="text-muted">
                            <i class="fas fa-map-marker-alt me-2"></i>Position: ${guide.position}
                        </small>
                    </div>
                    <div class="col-md-3">
                        <small class="text-muted">
                            <i class="fas fa-percentage me-2"></i>GC Content: ${guide.gcContent}%
                        </small>
                    </div>
                    <div class="col-md-3">
                        <small class="text-muted">
                            <i class="fas fa-exclamation-triangle me-2"></i>Off-targets: ${guide.offTargets}
                        </small>
                    </div>
                    <div class="col-md-3">
                        <small class="text-muted">
                            <i class="fas fa-dna me-2"></i>Strand: ${guide.strand}
                        </small>
                    </div>
                </div>
            </div>
        `;
    });

    resultsContent.innerHTML = html;
}

function createSequenceMap(sequence, guides) {
    const seqLength = sequence.length;
    console.log(`Sequence length: ${seqLength}bp`);

    // Create a scale function to convert bp positions to percentages
    function bpToPercent(bp) {
        return (bp / seqLength) * 100;
    }

    // Create ruler with precise bp positions
    const rulerNumbers = document.getElementById('rulerNumbers');
    rulerNumbers.innerHTML = '';
    
    // Create tick marks every 20bp
    for (let i = 0; i <= seqLength; i += 20) {
        const tick = document.createElement('span');
        tick.style.left = `${bpToPercent(i)}%`;
        tick.textContent = i;
        rulerNumbers.appendChild(tick);
    }

    // Create guide markers
    const guideMarkers = document.getElementById('guideMarkers');
    guideMarkers.innerHTML = '';
    
    guides.forEach((guide, index) => {
        const marker = document.createElement('div');
        marker.className = 'guide-marker';
        marker.setAttribute('data-guide-id', index + 1);
        marker.setAttribute('data-strand', guide.strand);

        // Calculate exact guide position
        const GUIDE_LENGTH = 23;  // bp
        const PAM_LENGTH = 3;     // bp
        let guideStart;          // leftmost bp position

        if (guide.strand === '+') {
            // For + strand, position marks cut site which is 3bp from right
            // So: position = guideStart + 20 (because cut site is -3 from end)
            guideStart = guide.position - 20;
        } else {
            // For - strand, position marks cut site which is 3bp from left
            // So: position = guideStart + 3
            guideStart = guide.position - 3;
        }

        // Convert bp positions to percentages
        const leftPercent = bpToPercent(guideStart);
        const widthPercent = bpToPercent(GUIDE_LENGTH);

        console.log(`Guide ${index + 1}:`, {
            strand: guide.strand,
            position: guide.position,
            guideStart,
            guideEnd: guideStart + GUIDE_LENGTH,
            leftPercent: leftPercent.toFixed(2) + '%',
            widthPercent: widthPercent.toFixed(2) + '%'
        });

        marker.style.left = `${leftPercent}%`;
        marker.style.width = `${widthPercent}%`;

        // Handle vertical stacking (existing code)
        let verticalPosition = 0;
        while (isPositionOverlapping(leftPercent, widthPercent, verticalPosition, usedPositions)) {
            verticalPosition += 20;
        }
        marker.style.top = `${verticalPosition}px`;
        usedPositions.add({ left: leftPercent, width: widthPercent, top: verticalPosition });

        // Add event listeners (existing code)
        marker.addEventListener('mouseenter', () => {
            const guideElement = document.querySelector(`#guide-${index + 1}`);
            if (guideElement) guideElement.classList.add('highlight');
        });
        
        marker.addEventListener('mouseleave', () => {
            const guideElement = document.querySelector(`#guide-${index + 1}`);
            if (guideElement) guideElement.classList.remove('highlight');
        });

        guideMarkers.appendChild(marker);
    });
}

// Helper function to check if a position overlaps with existing guides
function isPositionOverlapping(left, width, top, usedPositions) {
    for (const pos of usedPositions) {
        // Check if there's any overlap in both horizontal and vertical space
        const horizontalOverlap = !(left + width < pos.left - 5 || left > pos.left + pos.width + 5);
        const verticalOverlap = Math.abs(top - pos.top) < 15;
        
        if (horizontalOverlap && verticalOverlap) {
            return true;
        }
    }
    return false;
}

