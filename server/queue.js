const Queue = require('bull');
const { pool } = require('./config');
const { spawn } = require('child_process');
const path = require('path');

const guideGenerationQueue = new Queue('guide-generation', {
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
    const { sequence, resultId } = job.data;
    
    try {
        console.log(`Starting job processing for ${resultId}`);
        
        // Store the job in the database
        await pool.query(
            'INSERT INTO jobs (id, input_sequence, status) VALUES ($1, $2, $3)',
            [resultId, sequence, 'processing']
        );

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

        // Update the job with the results
        await pool.query(
            'UPDATE jobs SET status = $1, result_data = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $3',
            ['completed', results, resultId]
        );

        return { resultId, status: 'completed' };

    } catch (error) {
        console.error('Error processing job:', error);
        
        await pool.query(
            'UPDATE jobs SET status = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2',
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