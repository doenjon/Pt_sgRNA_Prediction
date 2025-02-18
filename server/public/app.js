const API_BASE_URL = window.location.port ? `${window.location.protocol}//${window.location.hostname}:${window.location.port}` : `${window.location.protocol}//${window.location.hostname}`;

document.getElementById('guideForm')?.addEventListener('submit', function(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitButton = form.querySelector('button[type="submit"]');
    const spinner = submitButton.querySelector('.spinner-border');
    let sequence = document.getElementById('sequence').value;
    const email = document.getElementById('email')?.value;

    // Validate and clean sequence
    sequence = sequence.replace(/[^ACTGactg]/g, '').toUpperCase();
    document.getElementById('sequence').value = sequence;

    if (sequence.length === 0) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger error-message';
        errorDiv.textContent = 'Please enter a valid DNA sequence using only A, C, T, G nucleotides.';
        form.appendChild(errorDiv);
        return;
    }

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
        body: JSON.stringify({ sequence, email }),
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
    console.log("Fetching results for ID:", resultId);
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');
    const errorState = document.getElementById('errorState');
    const downloadBtn = document.getElementById('downloadBtn');
    const emailNotification = document.getElementById('emailNotification');
    const waitingNotification = document.getElementById('waitingNotification');

    function pollResults() {
        console.log("Polling results for ID:", resultId);
        fetch(`${API_BASE_URL}/api/results/${resultId}`)
            .then(response => {
                if (response.status === 500) {
                    throw new Error('Server error');
                }
                return response.json();
            })
            .then(data => {
                console.log('Poll results data:', data);

                if (data.status === 'pending' || data.status === 'processing') {
                    // If still processing, continue polling after a delay
                    setTimeout(pollResults, 2000);
                    
                    // Show appropriate notification based on whether email was provided
                    if (data.email) {
                        emailNotification.style.display = 'block';
                        waitingNotification.style.display = 'none';
                    } else {
                        emailNotification.style.display = 'none';
                        waitingNotification.style.display = 'block';
                    }
                    
                    loadingState.style.display = 'block';
                    resultsContent.style.display = 'none';
                    errorState.style.display = 'none';
                    document.getElementById('sequenceMapSection').style.display = 'none';
                    
                } else if (data.status === 'completed') {
                    // Results are ready
                    loadingState.style.display = 'none';
                    resultsContent.style.display = 'block';
                    errorState.style.display = 'none';
                    document.getElementById('sequenceMapSection').style.display = 'block';
                    displayResults(data);
                    createSequenceMap(data.inputSequence, data.guides);
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

    console.log("Starting polling");
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
    const downloadBtn = document.getElementById('downloadBtn');
    let html = '';

    // Show download button when results are ready
    downloadBtn.style.display = 'inline-block';

    // Display guide sequences
    data.guides.forEach((guide, index) => {
        console.log(`Guide ${index + 1} data:`, guide);  // Log each guide's data

        // Calculate GC content
        const gcContent = guide.sequence ? 
            ((guide.sequence.match(/[GC]/gi) || []).length / guide.sequence.length) * 100 : 0;

        // Ensure off_targets is defined and display "0 off-targets" if it's zero
        const offTargets = guide.off_targets !== undefined ? guide.off_targets : 0;

        html += `
            <div class="guide-result" id="guide-${index + 1}">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h5 class="mb-0">Guide ${index + 1}</h5>
                    <span class="score-badge">Score: ${(guide.score != null ? Number(guide.score).toFixed(2) : 'N/A')}</span>
                </div>
                <div class="guide-card">
                    <div class="guide-sequence">
                        <code>${guide.sequence}</code>
                    </div>
                    <div class="guide-details">
                        <span class="detail-item">
                            position ${guide.position}
                        </span>
                        <span class="detail-item">
                            ${Math.round(gcContent)}% GC
                        </span>
                        <span class="detail-item">
                            ${offTargets} off-targets
                        </span>
                        <span class="detail-item">
                            ${guide.strand} strand
                        </span>
                    </div>
                </div>
            </div>
        `;
    });

    resultsContent.innerHTML = html;
}

function createSequenceMap(sequence, guides) {
    const seqLength = sequence.length;
    // Update the sequence length display
    const seqLengthElement = document.getElementById('seqLength');
    if (seqLengthElement) {
        seqLengthElement.textContent = seqLength;
    }

    // Create a scale function to convert bp positions to percentages
    function bpToPercent(bp) {
        return (bp / seqLength) * 100;
    }

    // Create ruler with precise bp positions
    const rulerNumbers = document.getElementById('rulerNumbers');
    const rulerMarks = document.getElementById('rulerMarks');
    rulerNumbers.innerHTML = '';
    rulerMarks.innerHTML = '';
    
    // Create numbers every 20bp
    for (let i = 0; i <= seqLength; i += 20) {
        // Create number
        const number = document.createElement('span');
        number.style.position = 'absolute';
        number.style.left = `${bpToPercent(i)}%`;
        number.style.transform = 'translateX(-50%)';
        number.textContent = i;
        rulerNumbers.appendChild(number);
    }

    // Create tick marks every 5bp
    for (let i = 0; i <= seqLength; i += 5) {
        // Create tick mark
        const tick = document.createElement('div');
        tick.className = 'ruler-tick';
        tick.style.left = `${bpToPercent(i)}%`;
        rulerMarks.appendChild(tick);
    }

    // Create guide markers
    const guideMarkers = document.getElementById('guideMarkers');
    guideMarkers.innerHTML = '';
    
    // Track used vertical positions
    const usedPositions = new Set();
    
    guides.forEach((guide, index) => {
        const marker = document.createElement('div');
        marker.className = 'guide-marker';
        marker.setAttribute('data-guide-id', index + 1);
        marker.setAttribute('data-strand', guide.strand);
        
        // Add tooltip content
        marker.setAttribute('data-tooltip', `Guide ${index + 1} (${guide.strand} strand)`);
        
        // Calculate position and width
        const GUIDE_LENGTH = guide.sequence.length;
        let leftPercent;
        
        if (guide.strand === '+') {
            // For + strand, position is at the cut site (near right end)
            // Move left by guide length minus 3bp (typical distance from PAM to cut site)
            // Subtract 1 to adjust position
            const guideStart = (guide.position - 1) - (GUIDE_LENGTH - 3);
            leftPercent = (guideStart / seqLength) * 100;
        } else {
            // For - strand, position is already at the left end
            // Subtract 1 to adjust position
            leftPercent = ((guide.position - 1) / seqLength) * 100;
        }
        
        const widthPercent = (GUIDE_LENGTH / seqLength) * 100;
        
        // Calculate color based on score
        const score = guide.score || 0;
        const grayValue = Math.round(150 - (score * 0.9));
        const color = `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
        
        marker.style.backgroundColor = color;
        marker.style.color = color;
        
        // Add arrow tip based on strand
        if (guide.strand === '+') {
            marker.style.clipPath = 'polygon(0 0, calc(100% - 8px) 0, 100% 50%, calc(100% - 8px) 100%, 0 100%)';
        } else {
            marker.style.clipPath = 'polygon(8px 0, 100% 0, 100% 100%, 8px 100%, 0 50%)';
        }
        
        // Set position and width
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
        
        // Add click handler for scrolling
        marker.addEventListener('click', () => {
            const guideElement = document.querySelector(`#guide-${index + 1}`);
            if (guideElement) {
                guideElement.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'center'
                });
            }
        });

        guideMarkers.appendChild(marker);
    });

    // Add sequence text above the line
    const sequenceText = document.getElementById('sequenceText');
    console.log('Adding sequence text:', sequence);  // Debug log
    // Create evenly spaced characters
    const charWidth = 100 / seqLength;  // Width percentage for each character
    console.log('Character width:', charWidth);  // Debug log
    sequenceText.innerHTML = sequence.split('').map((char, i) => 
        `<span style="position: absolute; left: ${charWidth * i}%">${char}</span>`
    ).join('');
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

// Add event listener for download button
document.addEventListener('DOMContentLoaded', function() {
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            const urlParams = new URLSearchParams(window.location.search);
            const resultId = urlParams.get('resultId');
            if (resultId) {
                window.location.href = `${API_BASE_URL}/api/download/${resultId}`;
            }
        });
    }
});

