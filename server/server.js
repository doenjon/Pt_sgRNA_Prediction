const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const guideGenerationQueue = require('./queue'); 
const path = require('path');
const { pool, initializeDatabase } = require('./config');

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(express.static('public'));

// Initialize database before starting the server
initializeDatabase()
    .then(() => {
        app.listen(port, () => {
            console.log(`Server running at http://localhost:${port}`);
        });
    })
    .catch(error => {
        console.error('Failed to initialize database:', error);
        process.exit(1);
    });

app.post('/api/generate', async (req, res) => {
    console.log('Received request to /api/generate with body:', req.body);

    const resultId = uuidv4(); // Unique ID for the result
    console.log(`Generated resultId: ${resultId}`);

    const { sequence } = req.body;
    if (!sequence) {
        console.error('Sequence is missing in the request body');
        return res.status(400).send('Sequence is required');
    }

    try {
        const job = await guideGenerationQueue.add({ sequence, resultId });
        console.log(`Job submitted from queue with ID: ${job.id} for resultId: ${resultId}`);

        res.json({ resultId }); // Send the resultId back to the client
    } catch (error) {
        console.error('Error adding job to queue:', error);
        res.status(500).send('Failed to queue the guide generation');
    }
});

app.get('/api/results/:resultId', async (req, res) => {
    try {
        const { resultId } = req.params;
        
        // Query to get the job results
        const query = `
            SELECT 
                jobs.input_sequence,
                jobs.result_data,
                jobs.status,
                jobs.created_at
            FROM jobs 
            WHERE jobs.id = $1
        `;
        
        const result = await pool.query(query, [resultId]);
        
        if (result.rows.length === 0) {
            return res.status(404).json({ error: 'Results not found' });
        }

        const job = result.rows[0];

        // Check if job is still processing
        if (job.status !== 'completed') {
            return res.status(202).json({ 
                status: job.status,
                message: 'Results are still processing'
            });
        }

        // Format the response
        const response = {
            inputSequence: job.input_sequence,
            guides: job.result_data.guides.map(guide => ({
                sequence: guide.sequence,
                position: guide.position,
                score: guide.score,
                gcContent: guide.gc_content,
                offTargets: guide.off_targets
            })),
            createdAt: job.created_at
        };

        res.json(response);
    } catch (error) {
        console.error('Error fetching results:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});