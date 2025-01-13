const { Pool } = require('pg');

const pool = new Pool({
    user: process.env.POSTGRES_USER,
    host: process.env.DB_HOST,
    database: process.env.POSTGRES_DB,
    password: process.env.POSTGRES_PASSWORD,
    port: 5432
});

async function waitForDatabase(maxAttempts = 10) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            console.log(`Attempting to connect to database (attempt ${attempt}/${maxAttempts})...`);
            await pool.query('SELECT 1');
            console.log('Successfully connected to database');
            return;
        } catch (error) {
            console.log(`Database connection attempt ${attempt} failed:`, error.message);
            if (attempt === maxAttempts) {
                throw new Error('Max connection attempts reached');
            }
            // Wait for 10 seconds before next attempt
            await new Promise(resolve => setTimeout(resolve, 10000));
        }
    }
}

async function initializeDatabase() {
    try {
        // Wait for database to be ready
        await waitForDatabase();

        // Create jobs table if it doesn't exist
        await pool.query(`
            CREATE TABLE IF NOT EXISTS jobs (
                id VARCHAR(255) PRIMARY KEY,
                input_sequence TEXT NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                result_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        `);
        console.log('Database initialized successfully');
    } catch (error) {
        console.error('Error initializing database:', error);
        throw error;
    }
}

module.exports = {
    pool,
    initializeDatabase
}; 