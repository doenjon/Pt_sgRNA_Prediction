const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const guideGenerationQueue = require('./queue'); 
const path = require('path');
const { pool, initializeDatabase } = require('./config');

const app = express();
const port = 3000;

app.use(cors({
    origin: '*',  // Be more specific in production
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type']
}));
app.use(bodyParser.json());
app.use(express.static('public'));

// Initialize database before starting the server
initializeDatabase()
    .then(() => {
        app.listen(port, '0.0.0.0', () => {
            console.log('Server running at http://0.0.0.0:3000');
        });
    })
    .catch(error => {
        console.error('Failed to initialize database:', error);
        process.exit(1);
    });

app.post('/api/generate', async (req, res) => {
    console.log('Received request to /api/generate');
    
    const resultId = uuidv4();
    const { sequence, email } = req.body;
    
    if (!sequence) {
        console.error('Sequence is missing in the request body');
        return res.status(400).send('Sequence is required');
    }

    // Clean sequence
    const cleanSequence = sequence.replace(/[^ACTGactg]/g, '').toUpperCase();

    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        
        // Store email with job if provided
        const query = `
            INSERT INTO jobs (id, input_sequence, email, status)
            VALUES ($1, $2, $3, 'pending')
        `;
        await client.query(query, [resultId, cleanSequence, email || null]);

        const job = await guideGenerationQueue.add({ 
            sequence: cleanSequence, 
            resultId,
            email 
        });
        
        await client.query('COMMIT');
        
        console.log(`Job submitted from queue with ID: ${job.id} for resultId: ${resultId}`);
        res.json({ resultId });
    } catch (error) {
        await client.query('ROLLBACK');
        console.error('Error in /api/generate:', error);
        res.status(500).send('Failed to queue the guide generation');
    } finally {
        client.release();
    }
});

app.get('/api/results/:resultId', async (req, res) => {
    try {
        const { resultId } = req.params;
        console.log('Fetching results for:', resultId);

        // Query the jobs table
        const jobQuery = `
            SELECT id, status, email, result_data, input_sequence 
            FROM jobs 
            WHERE id = $1
        `;
        
        const result = await pool.query(jobQuery, [resultId]);
        console.log('Database result:', result.rows[0]); // This log shows the raw DB result

        if (result.rows.length === 0) {
            return res.status(404).json({ error: 'Job not found' });
        }

        const job = result.rows[0];

        // If job is still processing, return appropriate status
        if (job.status === 'pending' || job.status === 'processing') {
            return res.json({
                status: 'processing',
                email: job.email
            });
        }

        // If job is completed, return the full response
        if (job.status === 'completed' && job.result_data) {
            const response = {
                status: 'completed',
                inputSequence: job.input_sequence,
                email: job.email,
                guides: job.result_data.guides,
                summary: job.result_data.summary
            };
            console.log('Sending completed response:', response); // Add this log
            return res.json(response);
        }

        // Handle error state
        return res.status(500).json({ 
            status: 'error',
            error: 'Job failed or invalid state' 
        });

    } catch (error) {
        console.error('Error fetching results:', error);
        res.status(500).json({ error: 'Failed to fetch results' });
    }
});

app.get('/api/download/:resultId', async (req, res) => {
    try {
        const { resultId } = req.params;
        
        const query = `
            SELECT result_data
            FROM jobs 
            WHERE id = $1 AND status = 'completed'
        `;
        
        const result = await pool.query(query, [resultId]);
        
        if (result.rows.length === 0) {
            return res.status(404).json({ error: 'Results not found' });
        }

        // Set headers for file download
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Content-Disposition', `attachment; filename=guide_results_${resultId}.json`);

        // Send the raw result data
        res.json(result.rows[0].result_data);

    } catch (error) {
        console.error('Error downloading results:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});
