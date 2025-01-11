
const { Pool } = require('pg');

const pool = new Pool({
  user: process.env.POSTGRES_USER || 'myuser',
  host: process.env.DB_HOST || 'db',
  database: process.env.POSTGRES_DB || 'mydatabase',
  password: process.env.POSTGRES_PASSWORD || 'mypassword',
  port: 5432,
});

pool.on('connect', (client) => {
    console.log('Database connection established:', client.connectionParameters.database);
  });

module.exports = pool;