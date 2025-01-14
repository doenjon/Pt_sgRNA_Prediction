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

    fetch('/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sequence }),
    })
    .then(response => {
        if (!response.ok) {
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
    fetch(`/api/results/${resultId}`)
        .then(response => {
            console.log('API Response:', response.status);
            if (!response.ok) {
                throw new Error('Results not found');
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            displayResults(data);
            createSequenceMap(data.inputSequence, data.guides);
            document.getElementById('downloadBtn').disabled = false;
        })
        .catch(error => {
            console.error('Error fetching results:', error);
            document.getElementById('results').innerHTML = `
                <div class="alert alert-danger">
                    Failed to load results. Please try again.
                </div>
            `;
        });
}

function displayResults(data) {
    if (!data || !data.guides || data.error) {
        const resultsContainer = document.getElementById('results');
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                ${data?.error || 'Failed to generate guides. Please try again.'}
            </div>
        `;
        return;
    }

    console.log('Displaying results:', data);
    const resultsContainer = document.getElementById('results');
    let html = '';

    // Display input sequence
    html += `
        <div class="sequence-display mb-4">
            <code>${data.inputSequence}</code>
        </div>
    `;

    // Display guide sequences
    data.guides.forEach((guide, index) => {
        html += `
            <div class="guide-result" id="guide-${index + 1}">
                <div class="d-flex justify-content-between align-items-start">
                    <h5>Guide ${index + 1}</h5>
                    <span class="score-badge">Score: ${guide.score.toFixed(2)}</span>
                </div>
                <div class="sequence-display">
                    <code>${guide.sequence}</code>
                </div>
                <div class="row mt-3">
                    <div class="col-md-4">
                        <small class="text-muted">
                            <i class="fas fa-map-marker-alt me-2"></i>Position: ${guide.position}
                        </small>
                    </div>
                    <div class="col-md-4">
                        <small class="text-muted">
                            <i class="fas fa-percentage me-2"></i>GC Content: ${guide.gcContent}%
                        </small>
                    </div>
                    <div class="col-md-4">
                        <small class="text-muted">
                            <i class="fas fa-exclamation-triangle me-2"></i>Off-targets: ${guide.offTargets}
                        </small>
                    </div>
                </div>
            </div>
        `;
    });

    resultsContainer.innerHTML = html;
}

function createSequenceMap(sequence, guides) {
    console.log('Creating sequence map:', { sequence, guides });
    const seqLength = sequence.length;
    document.getElementById('seqLength').textContent = seqLength;

    // Create ruler numbers
    const rulerNumbers = document.getElementById('rulerNumbers');
    rulerNumbers.innerHTML = '';
    
    for (let i = 0; i <= seqLength; i += 20) {
        const number = document.createElement('span');
        number.style.position = 'absolute';
        number.style.left = `${(i / seqLength) * 100}%`;
        number.textContent = i;
        rulerNumbers.appendChild(number);
    }

    // Create guide markers with staggering
    const guideMarkers = document.getElementById('guideMarkers');
    guideMarkers.innerHTML = '';
    
    // Sort guides by position to check for overlaps
    const sortedGuides = [...guides].sort((a, b) => a.position - b.position);
    
    // Track used vertical positions
    const usedPositions = new Set();
    
    sortedGuides.forEach((guide, index) => {
        const marker = document.createElement('div');
        marker.className = 'guide-marker';
        marker.setAttribute('data-guide-id', index + 1);
        
        // Calculate horizontal position and width
        const leftPos = (guide.position / seqLength) * 100;
        const width = (guide.sequence.length / seqLength) * 100;
        marker.style.left = `${leftPos}%`;
        marker.style.width = `${width}%`;
        
        // Find a suitable vertical position
        let verticalPosition = 0;
        while (isPositionOverlapping(leftPos, width, verticalPosition, usedPositions)) {
            verticalPosition += 20; // Increment by 20px until we find a free spot
        }
        
        marker.style.top = `${verticalPosition}px`;
        usedPositions.add({ left: leftPos, width, top: verticalPosition });
        
        // Add event listeners
        marker.addEventListener('mouseenter', () => {
            const guideElement = document.querySelector(`#guide-${index + 1}`);
            if (guideElement) {
                guideElement.classList.add('highlight');
            }
        });
        
        marker.addEventListener('mouseleave', () => {
            const guideElement = document.querySelector(`#guide-${index + 1}`);
            if (guideElement) {
                guideElement.classList.remove('highlight');
            }
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

