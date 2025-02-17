const nodemailer = require('nodemailer');

// Add debug log at the top of the file
console.log('Loading email service with config:', {
    host: process.env.SES_HOST,
    port: process.env.SES_PORT,
    user: process.env.SES_USER,
    pass: process.env.SES_PASS ? '***' : 'not set'
});

// Create reusable transporter for Amazon SES
const transporter = nodemailer.createTransport({
    host: process.env.SES_HOST,
    port: process.env.SES_PORT,
    secure: false,
    auth: {
        user: process.env.SES_USER,
        pass: process.env.SES_PASS
    },
    debug: true,
    logger: true  // Add this for more detailed logs
});

// Test the connection on startup
transporter.verify((error, success) => {
    if (error) {
        console.log('SES connection failed:', error);
        // Log the configuration (but mask the password)
        console.log('SES Config:', {
            host: process.env.SES_HOST,
            port: process.env.SES_PORT,
            user: process.env.SES_USER,
            pass: process.env.SES_PASS ? '***' : 'not set'
        });
    } else {
        console.log('SES is ready to send messages');
    }
});

async function sendResultsEmail(email, resultId) {
    const resultsUrl = `${process.env.BASE_URL || 'http://localhost'}/results.html?resultId=${resultId}`;
    
    try {
        await transporter.sendMail({
            from: process.env.SES_FROM || 'guidedesign@ptcrispr.org',
            to: email,
            subject: 'Your CRISPR Guide Results Are Ready',
            text: `Your guide design results are ready! View them at: ${resultsUrl}`,
            html: `
                <h2>Your CRISPR Guide Results Are Ready</h2>
                <p>Your guide design job has completed. Click the link below to view your results:</p>
                <p><a href="${resultsUrl}">${resultsUrl}</a></p>
            `
        });
        console.log(`Results email sent to ${email}`);
    } catch (error) {
        console.error('Error sending email via SES:', error);
        throw error;
    }
}

module.exports = { sendResultsEmail }; 