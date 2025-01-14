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
    console.log('Request headers:', req.headers);
    console.log('Received request to /api/generate with body:', req.body);

    const resultId = uuidv4();
    console.log(`Generated resultId: ${resultId}`);

    const { sequence } = req.body;
    if (!sequence) {
        console.error('Sequence is missing in the request body');
        return res.status(400).send('Sequence is required');
    }

    try {
        const job = await guideGenerationQueue.add({ sequence, resultId });
        console.log(`Job submitted from queue with ID: ${job.id} for resultId: ${resultId}`);

        res.json({ resultId });
    } catch (error) {
        console.error('Error in /api/generate:', error);
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
            return res.status(202).json({ 
                status: 'processing',
                message: 'Job is being processed'
            });
        }

        const job = result.rows[0];
        console.log('Raw job data from database:', {
            status: job.status,
            result_data: job.result_data,
            has_guides: job.result_data?.guides?.length > 0
        });

        // Log the first guide's complete data
        if (job.result_data?.guides?.length > 0) {
            console.log('First guide complete data:', job.result_data.guides[0]);
        }

        // Check if job is still processing
        if (job.status !== 'completed' || !job.result_data?.guides) {
            return res.status(202).json({ 
                status: job.status,
                message: 'Results are still processing'
            });
        }

        // Format the response for completed jobs
        const response = {
            status: 'completed',
            inputSequence: job.input_sequence,
            guides: job.result_data.guides.map(guide => {
                console.log('Processing guide:', guide);  // Log each guide as we process it
                return {
                    sequence: guide.sequence,
                    position: guide.position,
                    score: Number(guide.sgRNA_Scorer || guide.score),
                    gcContent: guide.gc_content,
                    offTargets: guide.off_targets,
                    strand: guide.strand
                };
            }),
            createdAt: job.created_at
        };

        res.json(response);
    } catch (error) {
        console.error('Error fetching results:', error);
        res.status(500).json({ error: 'Internal server error' });
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
