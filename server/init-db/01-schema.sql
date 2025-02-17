CREATE TABLE IF NOT EXISTS jobs (
    id VARCHAR(36) PRIMARY KEY,  -- Using UUID format
    input_sequence TEXT NOT NULL,
    email VARCHAR(255),  -- Add email field
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    result_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_email ON jobs(email);
