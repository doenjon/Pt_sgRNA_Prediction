const nodemailer = require('nodemailer');

// Create reusable transporter for Amazon SES
const transporter = nodemailer.createTransport({
    host: process.env.SES_HOST || 'email-smtp.us-east-1.amazonaws.com',
    port: process.env.SES_PORT ? Number(process.env.SES_PORT) : 587,
    secure: process.env.SES_PORT && process.env.SES_PORT === '465', // true if using port 465
    auth: {
        user: process.env.SES_USER,
        pass: process.env.SES_PASS
    },
    tls: {
        rejectUnauthorized: false
    },
    debug: true // Enable debugging output to help diagnose issues
});

// Test the connection
transporter.verify((error, success) => {
    if (error) {
        console.log('SES connection failed:', error);
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