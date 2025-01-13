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
        
        // First, store the job in the database
        await pool.query(
            'INSERT INTO jobs (id, input_sequence, status) VALUES ($1, $2, $3)',
            [resultId, sequence, 'processing']
        );

        // Run the Python script using spawn instead of exec
        const result = await new Promise((resolve, reject) => {
            console.log('Spawning Python process...');
            
            const pythonProcess = spawn('python3', [
                path.join(process.cwd(), 'generate_guides.py')
            ]);

            let outputData = '';
            let errorData = '';

            pythonProcess.stdout.on('data', (data) => {
                outputData += data.toString();
                console.log('Python stdout:', data.toString());
            });

            pythonProcess.stderr.on('data', (data) => {
                errorData += data.toString();
                console.error('Python stderr:', data.toString());
            });

            pythonProcess.on('error', (error) => {
                console.error('Failed to start Python process:', error);
                reject(error);
            });

            pythonProcess.on('close', (code) => {
                console.log(`Python process exited with code ${code}`);
                if (code !== 0) {
                    reject(new Error(`Python process exited with code ${code}: ${errorData}`));
                } else {
                    resolve(outputData);
                }
            });

            // Write the sequence to stdin
            pythonProcess.stdin.write(sequence);
            pythonProcess.stdin.end();
        });

        console.log('Parsing Python output...');
        const results = JSON.parse(result);

        console.log('Updating database with results...');
        // Update the job with the results
        await pool.query(
            'UPDATE jobs SET status = $1, result_data = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $3',
            ['completed', results, resultId]
        );

        console.log(`Job ${resultId} completed successfully`);
        return { resultId, status: 'completed' };

    } catch (error) {
        console.error('Error processing job:', error);
        
        // Update the job status to failed
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