const Bull = require('bull');
const { pool } = require('./config');
const { spawn } = require('child_process');
const path = require('path');
const { sendResultsEmail } = require('./services/email');

const guideGenerationQueue = new Bull('guide-generation', {
    redis: {
        host: process.env.REDIS_HOST || 'redis',
        port: process.env.REDIS_PORT || 6379,
    }
});

// Add connection error handling
pool.on('error', (err) => {
    console.error('Unexpected error on idle client', err);
});

// Add job processing logic
guideGenerationQueue.process(async (job) => {
    try {
        const { resultId, sequence, email } = job.data;
        console.log(`Starting to process job ${resultId}`);
        
        // Update job status to processing
        await pool.query(
            'UPDATE jobs SET status = $1 WHERE id = $2',
            ['processing', resultId]
        );
        console.log(`Updated job ${resultId} status to processing`);

        // Send job to Redis queue for Python service
        const redis = guideGenerationQueue.client;
        await redis.lpush('guide_design_queue', JSON.stringify({ sequence, resultId }));

        // Wait for results
        const results = await new Promise((resolve, reject) => {
            const checkResults = async () => {
                const result = await redis.get(`results:${resultId}`);
                if (result) {
                    redis.del(`results:${resultId}`);
                    resolve(JSON.parse(result));
                } else {
                    setTimeout(checkResults, 1000);
                }
            };
            checkResults();
        });

        // When job completes, update the database with results
        const updateResult = await pool.query(
            'UPDATE jobs SET status = $1, result_data = $2 WHERE id = $3 RETURNING *',
            ['completed', results, resultId]
        );
        console.log(`Updated job ${resultId} with results:`, updateResult.rows[0]);

        // Send email if provided
        if (email) {
            try {
                await sendResultsEmail(email, resultId);
            } catch (emailError) {
                console.error('Failed to send results email:', emailError);
                // Continue processing even if email fails
            }
        }

        return { resultId, status: 'completed' };
    } catch (error) {
        console.error('Error processing job:', error);
        
        // Update job status to failed
        await pool.query(
            'UPDATE jobs SET status = $1 WHERE id = $2',
            ['failed', resultId]
        );
        
        throw error;
    }
});

// Log when jobs are added to the queue
guideGenerationQueue.on('waiting', (jobId) => {
    console.log(`Job ${jobId} is waiting to be processed`);
});
  
// Log when jobs are completed
guideGenerationQueue.on('completed', (job, result) => {
    console.log(`Job ${job.id} completed with result:`, result);
});

// Log when jobs fail
guideGenerationQueue.on('failed', (job, err) => {
    console.error(`Job ${job.id} failed with error:`, err);
});

// Log when jobs are active
guideGenerationQueue.on('active', (job) => {
    console.log(`Job ${job.id} has started`);
});

module.exports = guideGenerationQueue;