CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    input_sequence TEXT NOT NULL,
    email TEXT,
    status TEXT NOT NULL,
    result_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_email ON jobs(email);
