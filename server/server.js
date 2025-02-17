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
        console.log('Fetching results for:', resultId); // Add debug log

        const result = await pool.query(
            'SELECT * FROM results WHERE id = $1',
            [resultId]
        );

        if (result.rows.length === 0) {
            return res.status(404).json({ error: 'Results not found' });
        }

        console.log('Found results:', result.rows[0]); // Add debug log
        res.json(result.rows[0]);
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
