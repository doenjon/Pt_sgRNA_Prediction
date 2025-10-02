const nodemailer = require('nodemailer');

// Create reusable transporter for Amazon SES
const transporter = nodemailer.createTransport({
    host: process.env.SES_HOST,
    port: process.env.SES_PORT,
    auth: {
        user: process.env.SES_USER,
        pass: process.env.SES_PASS
    }
});

// Add this before verify
console.log('Attempting SMTP connection with:', {
    host: process.env.SES_HOST,
    port: process.env.SES_PORT,
    secure: false,
    auth: {
        user: process.env.SES_USER ? 'SET' : 'NOT SET',
        pass: process.env.SES_PASS ? 'SET' : 'NOT SET'
    }
});

// Test the connection with more detailed error logging
transporter.verify((error, success) => {
    if (error) {
        console.log('Email service configuration:', {
            host: process.env.SES_HOST,
            port: process.env.SES_PORT,
            secure: false,
            username: process.env.SES_USER ? '(set)' : '(not set)',
            password: process.env.SES_PASS ? '(set)' : '(not set)'
        });
        console.log('Email service error details:', {
            code: error.code,
            command: error.command,
            response: error.response,
            responseCode: error.responseCode,
            message: error.message,
            stack: error.stack // Add stack trace
        });
    } else {
        console.log('Email service is ready');
    }
});

async function sendResultsEmail(email, resultId) {
    if (!process.env.SES_USER || !process.env.SES_PASS) {
        console.log('Email service not configured - skipping email notification');
        return;
    }

    const resultsUrl = `${process.env.BASE_URL || 'http://localhost'}/results.html?resultId=${resultId}`;
    
    try {
        await transporter.sendMail({
            from: process.env.SES_FROM,
            to: email,
            subject: 'Your CRISPR Guide Results Are Ready',
            html: `
                <h2>Your CRISPR Guide Results Are Ready</h2>
                <p><a href="${resultsUrl}">${resultsUrl}</a></p>
            `
        });
        console.log(`Results email sent to ${email}`);
    } catch (error) {
        console.error('Failed to send email:', error.message);
        // Don't throw the error - allow the application to continue even if email fails
    }
}

module.exports = { sendResultsEmail }; 