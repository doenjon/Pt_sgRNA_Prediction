const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const guideGenerationQueue = require('./queue'); 
const path = require('path');
const { pool, initializeDatabase } = require('./config');
const redisClient = guideGenerationQueue.client;

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
        
        // Insert job as 'pending'
        const query = `
            INSERT INTO jobs (id, input_sequence, email, status)
            VALUES ($1, $2, $3, 'pending')
        `;
        await client.query(query, [resultId, cleanSequence, email || null]);

        // Add job to the queue
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

app.get('/api/health', async (req, res) => {
    try {
        // 1. Check database connection
        await pool.query('SELECT 1');

        // 2. Check Redis connection
        await redisClient.ping();

        // 3. Check worker heartbeat
        const heartbeat = await redisClient.get('worker:heartbeat');
        if (!heartbeat) {
            throw new Error('Worker heartbeat not found.');
        }

        const lastHeartbeat = parseFloat(heartbeat);
        const now = Date.now() / 1000; // Convert to seconds
        const timeSinceHeartbeat = now - lastHeartbeat;

        if (timeSinceHeartbeat > 120) { // 2 minutes
            throw new Error(`Worker heartbeat is too old (${Math.round(timeSinceHeartbeat)}s ago).`);
        }

        // All checks passed
        res.status(200).json({
            status: 'ok',
            database: 'connected',
            redis: 'connected',
            worker: 'healthy',
            lastHeartbeat: `${Math.round(timeSinceHeartbeat)}s ago`
        });

    } catch (error) {
        console.error('Health check failed:', error.message);
        res.status(503).json({
            status: 'error',
            message: error.message
        });
    }
});

app.get('/api/results/:resultId', async (req, res) => {
    try {
        const { resultId } = req.params;
        
        const query = `
            SELECT 
                jobs.input_sequence,
                jobs.result_data,
                jobs.status,
                jobs.created_at,
                jobs.email
            FROM jobs 
            WHERE jobs.id = $1
        `;
        
        const result = await pool.query(query, [resultId]);
        
        if (result.rows.length === 0) {
            return res.status(202).json({ 
                status: 'processing',
                message: 'Job is initializing'
            });
        }

        const job = result.rows[0];

        // If we have guide data, normalize the scores
        if (job.result_data?.guides) {
            job.result_data.guides = job.result_data.guides.map(guide => {
                // Get the raw score
                let score = guide.sgRNA_Scorer || guide.score;
                
                // Clamp the score to [-2, 2] range
                score = Math.max(-2, Math.min(2, score));
                
                // Convert to 0-100 range: 
                // -2 maps to 0
                // 0 maps to 50
                // 2 maps to 100
                const normalizedScore = Math.round(((score + 2) / 4) * 100);
                
                return {
                    ...guide,
                    score: normalizedScore
                };
            });
        }

        return res.json({
            status: job.status,
            email: job.email,
            message: job.status === 'processing' ? 'Job is being processed' : undefined,
            guides: job.result_data?.guides || [],
            inputSequence: job.result_data?.inputSequence || job.input_sequence,
            createdAt: job.created_at
        });

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
